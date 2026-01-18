"""train_diffusion.py

Training loop for the (MLP-based) conditional diffusion model used in the full
evaluation suite.

Notes:
- The caller is expected to pass `model_save_dir` so checkpoints land inside the
  suite's per-run folder. If `model_save_dir` is not provided, this script falls
  back to the original historical behaviour (`../../models/{run_id}/`) to avoid
  breaking legacy usage.
- Several debug/stability toggles can be enabled via environment variables:
  - `DM_DEBUG=1` enables extra finite checks and anomaly detection.
  - `DM_CLAMP_EPS` clamps the cumulative alpha schedule away from 0/1.
  - `DM_CLIP_GRAD` enables gradient clipping.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset

from diffusion_models import custom_update_bn, get_swa_lr
def train_diffusion_model(
        data, 
        nb_conditions,
        model,  # instance of LatentDiffusionModel
        num_steps, 
        alpha_cumprod, 
        epochs=500, 
        lr=1e-3, 
        batch_size=1024, 
        run_id=None,
        model_save_dir: str = None,
        swa=False,
        swa_start_ratio=0.5,
        swa_lr=0.001,
        lambda_rec=1.0,   # weight for reconstruction loss
        lambda_kl=1e-7,   # weight for KL divergence
        lambda_diff=1.0,  # weight for diffusion (noise) loss
        pretrain_epochs=0,  # number of epochs to pretrain encoder-decoder only
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_variational=True,
        diffusion_space="data",
):
    assert diffusion_space in ["data", "latent"], \
        "diffusion_space must be either 'data' or 'latent'."
    # Debug controls via environment variables (non-breaking for callers)
    DM_DEBUG = os.environ.get("DM_DEBUG", "0") != "0"
    DM_CLAMP_EPS = float(os.environ.get("DM_CLAMP_EPS", "0"))
    DM_SAFE_RECON = os.environ.get("DM_SAFE_RECON", "1") != "0"
    
    if diffusion_space == "data":
        assert pretrain_epochs == 0, \
            "Pretraining is only supported for latent diffusion space."
        use_variational = False
    
    # Define loss functions:
    criterion_recon = nn.MSELoss()
    criterion_diff = nn.MSELoss()
    base_optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Optionally pretrain autoencoder (encoder+decoder) conditioning on data and covariates.
    if pretrain_epochs > 0 and diffusion_space != "data":
        print("Pretraining the conditional autoencoder (encoder+decoder)...")
        model.train()
        # We train only the encoder and decoder. Freeze the diffusion denoiser:
        for p in model.denoiser.parameters():
            p.requires_grad = False
        model.denoiser.eval()
        pretrain_optimizer = optim.AdamW(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=lr)
        # LR scheduler: reduce by factor 0.75 after 50 epochs without improvement
        scheduler = ReduceLROnPlateau(pretrain_optimizer, mode='min', factor=0.75, patience=75, cooldown=0, min_lr=5e-5)
        for epoch in range(pretrain_epochs):
            total_pre_loss = 0.0
            total_pre_loss_rec = 0.0
            total_pre_loss_kl = 0.0
            num_batches = 0
            for idx, (data_batch, cond_batch) in enumerate(DataLoader(
                                            TensorDataset(torch.tensor(data[:, :-nb_conditions], dtype=torch.float32),
                                                          torch.tensor(data[:, -nb_conditions:], dtype=torch.float32)),
                                            batch_size=batch_size, shuffle=True, drop_last=False)):
                data_batch = data_batch.to(device)
                cond_batch = cond_batch.to(device)
                if idx % 2 == 0:
                    # inject noise
                    data_batch = data_batch + torch.randn_like(data_batch) * 0.025
                pretrain_optimizer.zero_grad()
                # Forward: for pretraining, simply compute the reconstruction.
                z, mu, logvar = model.encoder(data_batch, cond_batch)
                # print(z.shape)
                # For a conditional decoder, pass both z and cond.

                # apply some augmentation techniques to the latent space
                # 1. Gaussian noise
                # z = z + torch.randn_like(z) * 0.1
                # # 2. Dropout
                # z = nn.Dropout(0.1)(z)
                # # 3. BatchNorm
                # z = F.batch_norm()  # batchnorm over the latent dimension 

                x_recon = model.decoder(z, cond_batch)
                # print(x_recon.shape, data_batch.shape)
                loss_rec = criterion_recon(x_recon, data_batch)
                if use_variational:
                    loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                else:
                    loss_kl = torch.tensor(0.0).to(device)
                loss = lambda_rec * loss_rec + lambda_kl * loss_kl
                loss.backward()
                pretrain_optimizer.step()
                
                total_pre_loss += loss.item()
                total_pre_loss_rec += lambda_rec * loss_rec.item()
                total_pre_loss_kl += lambda_kl * loss_kl.item()

                num_batches += 1
            
            avg_pre_loss = total_pre_loss / num_batches
            avg_pre_loss_rec = total_pre_loss_rec / num_batches
            avg_pre_loss_kl = total_pre_loss_kl / num_batches

            scheduler.step(avg_pre_loss)

            print(f"[Pretrain] Epoch [{epoch+1}/{pretrain_epochs}] Loss: {avg_pre_loss:.4f} = Rec: {avg_pre_loss_rec:.4f} + KL: {avg_pre_loss_kl:.8f}")
        
        print("Pretraining autoencoder complete.")
        # Unfreeze the denoiser:
        
        for p in model.denoiser.parameters():
            p.requires_grad = True
        model.denoiser.train()
        # for p in model.encoder.parameters():
        #     p.requires_grad = False
        # for p in model.decoder.parameters():
        #     p.requires_grad = False
        # model.encoder.eval()
        # model.decoder.eval()
        # model.denoiser.train()

    # base_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        base_optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': lr * 0.1},
        {'params': model.decoder.parameters(), 'lr': lr * 0.1},
        {'params': model.denoiser.parameters(), 'lr': lr}
        ])

    # Continue with full training.
    # base_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(base_optimizer, swa_lr=swa_lr)
    # LR scheduler: reduce by factor 0.75 after 50 epochs without improvement
    scheduler = ReduceLROnPlateau(base_optimizer, mode='min', factor=0.75, patience=75, cooldown=0, min_lr=1e-6)
    swa_start = int(epochs * swa_start_ratio)

    # Split the data into data and conditioning. (Assume they are the last nb_conditions columns.)
    data_full = data[:, :-nb_conditions]
    cond_data_full = data[:, -nb_conditions:]
    dataset = TensorDataset(torch.tensor(data_full, dtype=torch.float32),
                             torch.tensor(cond_data_full, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    best_loss = np.inf
    best_epoch = 0
    best_swa_loss = np.inf
    best_swa_epoch = 0

    model.to(device)

    def _verify_model_devices(prefix: str = ""):
        if not DM_DEBUG:
            return
        dev_counts = {}
        for name, p in model.named_parameters():
            d = str(p.device)
            dev_counts[d] = dev_counts.get(d, 0) + 1
        for name, b in model.named_buffers():
            d = str(b.device)
            dev_counts[d] = dev_counts.get(d, 0) + 1
        print(f"{prefix} Model param/buffer devices: {dev_counts}")

    _verify_model_devices(prefix="[DM_DEBUG]")

    for epoch in range(epochs):
        if swa and epoch > swa_start:
            current_swa_lr = get_swa_lr(epoch, swa_start, epochs, swa_lr, final_lr=0.00002)
            for param_group in base_optimizer.param_groups:
                param_group['lr'] = current_swa_lr
            print(f"Epoch {epoch}: Adjusted SWA LR = {current_swa_lr:.6f}")
        
        if pretrain_epochs == 0:
            model.train()
        else:
            model.train()
            
        total_loss = 0.0
        total_loss_diff = 0.0
        total_loss_rec = 0.0
        total_loss_kl = 0.0
        swa_total_loss = 0.0
        num_batches = 0

        for idx, (data_batch, cond_batch) in enumerate(dataloader):
            data_batch = data_batch.to(device)
            cond_batch = cond_batch.to(device)

            # (Optional augmentation)
            if idx % 2 == 0:
                data_batch = data_batch + torch.randn_like(data_batch) * 0.025

            # Sample a random timestep for each sample.
            t = torch.randint(low=0, high=num_steps, size=(data_batch.size(0),), device=device).long()
            t_float = t.unsqueeze(-1).float()

            # Use the cumulative schedule for noise injection.
            # Here, alpha_bar_t represents ᾱₜ = ∏ₖ₌₁ᵗ αₖ.
            alpha_bar_t = alpha_cumprod[t].to(device).unsqueeze(-1)  # shape: [batch, 1]
            if DM_CLAMP_EPS > 0.0:
                alpha_bar_t = alpha_bar_t.clamp(DM_CLAMP_EPS, 1.0 - DM_CLAMP_EPS)
            std_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-12))

            # Forward pass (the model’s forward uses the cumulative factor)
            predicted_noise, x_recon, mu, logvar, z, z_noisy = model(data_batch, 
                                                                    t_float, 
                                                                    cond_batch, 
                                                                    alpha_bar_t, 
                                                                    std_t,
                                                                    use_mu=False)
            if DM_DEBUG:
                finite_checks = {
                    'data_batch': torch.isfinite(data_batch).all().item(),
                    'cond_batch': torch.isfinite(cond_batch).all().item(),
                    't_float': torch.isfinite(t_float).all().item(),
                    'alpha_bar_t': torch.isfinite(alpha_bar_t).all().item(),
                    'std_t': torch.isfinite(std_t).all().item(),
                    'predicted_noise': torch.isfinite(predicted_noise).all().item(),
                    'x_recon': torch.isfinite(x_recon).all().item(),
                }
                if not all(finite_checks.values()):
                    print("[DM_DEBUG] Non-finite detected before loss. Checks:", finite_checks)
                    print(f"[DM_DEBUG] alpha_bar_t min/max: {alpha_bar_t.min().item():.3e}/{alpha_bar_t.max().item():.3e}")
                    print(f"[DM_DEBUG] std_t min/max: {std_t.min().item():.3e}/{std_t.max().item():.3e}")
                    print(f"[DM_DEBUG] t range: {t.min().item()}..{t.max().item()} num_steps={num_steps}")
                    _verify_model_devices(prefix="[DM_DEBUG]")
                    # Early stop to surface the issue
                    raise RuntimeError("Non-finite detected prior to loss computation")
            # Recompute the noise that was injected:
            eps = (z_noisy - torch.sqrt(alpha_bar_t) * z) / (std_t + 1e-12)
            loss_diff = criterion_diff(predicted_noise, eps)
            if DM_DEBUG and not torch.isfinite(loss_diff):
                print("[DM_DEBUG] loss_diff is non-finite. Stats:")
                def _stat(tensor, name):
                    t_cpu = tensor.detach().float().cpu()
                    print(f"  {name}: mean={t_cpu.mean():.3e} std={t_cpu.std():.3e} maxabs={t_cpu.abs().max():.3e} has_nan={torch.isnan(t_cpu).any().item()} has_inf={torch.isinf(t_cpu).any().item()}")
                _stat(predicted_noise, 'predicted_noise')
                _stat(eps, 'eps')
                _stat(alpha_bar_t, 'alpha_bar_t')
                _stat(std_t, 'std_t')
                raise RuntimeError("loss_diff became non-finite")

            # z_noisy = torch.sqrt(alpha_bar_t) * z + std_t * eps
            
            if diffusion_space == "latent":
                # if pretrain_epochs > 0:
                #     loss_rec = torch.tensor(0.0).to(device)
                #     # calculate loss rec only for printing
                #     loss_rec_print = criterion_recon(x_recon, data_batch)
                # else:
                loss_rec = criterion_recon(x_recon, data_batch)
                loss_rec_print = criterion_recon(x_recon, data_batch)
            else:
                loss_rec = torch.tensor(0.0).to(device)
                if DM_SAFE_RECON:
                    with torch.no_grad():
                        loss_rec_print = criterion_recon(x_recon, data_batch)
                else:
                    loss_rec_print = criterion_recon(x_recon, data_batch)
                # loss_rec_print = torch.tensor(0.0).to(device)
            
            if use_variational and diffusion_space == "latent":
                if pretrain_epochs > 0:
                    # loss_kl = torch.tensor(0.0).to(device)
                    loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                    loss_kl_print = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                else:
                    loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                    loss_kl_print = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            else:
                loss_kl = torch.tensor(0.0).to(device)
                loss_kl_print = torch.tensor(0.0).to(device)
            
            # if pretrain_epochs > 0:
            #     loss_batch = lambda_diff * loss_diff
            # else:
            loss_batch = lambda_diff * loss_diff + lambda_rec * loss_rec + lambda_kl * loss_kl
            
            base_optimizer.zero_grad()
            if DM_DEBUG:
                torch.autograd.set_detect_anomaly(True)
            loss_batch.backward()
            clip_val = float(os.environ.get("DM_CLIP_GRAD", "0"))
            if clip_val and clip_val > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
            if DM_DEBUG:
                bad_grads = []
                for n, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grads.append(n)
                if bad_grads:
                    print("[DM_DEBUG] Non-finite gradients in:", bad_grads)
                    raise RuntimeError("Non-finite gradients detected")
            base_optimizer.step()
            
            # total_loss += loss_batch.item()
            total_loss_diff += lambda_diff * loss_diff.item()
            # if pretrain_epochs == 0:
            #     total_loss_rec += lambda_rec * loss_rec.item()
            # else:
            total_loss_rec += lambda_rec * loss_rec.item()
            if pretrain_epochs == 0:
                total_loss_kl += lambda_kl * loss_kl.item()
            else:
                total_loss_kl += lambda_kl * loss_kl_print.item()

            num_batches += 1
            
            if swa and epoch > swa_start:
                swa_model.update_parameters(model)
                model.eval()
                with torch.no_grad():
                    predicted_noise_swa, x_recon_swa, mu_swa, logvar_swa, z_swa, z_noisy_swa = swa_model(data_batch, t_float, cond_batch, alpha_bar_t, std_t)
                    eps_swa = (z_noisy_swa - torch.sqrt(alpha_bar_t) * z_swa) / std_t
                    loss_diff_swa = criterion_diff(predicted_noise_swa, eps_swa)

                    loss_rec_swa = criterion_recon(x_recon_swa, data_batch)
                    loss_rec_print_swa = criterion_recon(x_recon_swa, data_batch)
                    
                    if use_variational and diffusion_space == "latent":
                        if pretrain_epochs > 0:
                            loss_kl_swa = -0.5 * torch.mean(torch.sum(1 + logvar_swa - mu_swa.pow(2) - logvar_swa.exp(), dim=1))
                            loss_kl_print_swa = -0.5 * torch.mean(torch.sum(1 + logvar_swa - mu_swa.pow(2) - logvar_swa.exp(), dim=1))
                        else:
                            loss_kl_swa = -0.5 * torch.mean(torch.sum(1 + logvar_swa - mu_swa.pow(2) - logvar_swa.exp(), dim=1))
                            loss_kl_print_swa = -0.5 * torch.mean(torch.sum(1 + logvar_swa - mu_swa.pow(2) - logvar_swa.exp(), dim=1))
                    else:
                        loss_kl_swa = torch.tensor(0.0).to(device)
                        loss_kl_print_swa = torch.tensor(0.0).to(device)

                    loss_swa = lambda_diff * loss_diff_swa + lambda_rec * loss_rec_swa + lambda_kl * loss_kl_swa
                    # loss_swa_save = lambda_diff * loss_diff_swa
                    swa_total_loss += loss_swa.item()
                if pretrain_epochs == 0:
                    model.train()
                else:
                    # model.encoder.eval()
                    # model.decoder.eval()
                    # model.denoiser.train()
                    model.train()
        
        total_loss = total_loss_diff + total_loss_rec + total_loss_kl
        total_schedular_loss = total_loss_diff
        
        avg_loss = total_loss / num_batches
        avg_schedular_loss = total_schedular_loss / num_batches
        avg_loss_diff = total_loss_diff / num_batches
        avg_loss_rec = total_loss_rec / num_batches
        avg_loss_kl = total_loss_kl / num_batches
        avg_swa_loss = (swa_total_loss / num_batches) if (swa and epoch > swa_start) else avg_loss
        avg_swa_loss_save = (swa_total_loss / num_batches) if (swa and epoch > swa_start) else avg_loss_diff
        
        if avg_schedular_loss < best_loss:
            best_loss = avg_schedular_loss
            best_epoch = epoch
            save_dir = model_save_dir if model_save_dir is not None else (f"../../models/{run_id}/" if run_id is not None else "../../models/prototyping/")
            os.makedirs(save_dir, exist_ok=True)
            filename = f"best_model_epoch_{epoch}_steps_{num_steps}.pt"
            # Optionally remove old best files:
            for f in os.listdir(save_dir):
                if f.startswith("best_model_epoch"):
                    os.remove(os.path.join(save_dir, f))
            torch.save(model.state_dict(), os.path.join(save_dir, filename))
            
        if swa and epoch > swa_start and avg_swa_loss_save < best_swa_loss:
            best_swa_loss = avg_swa_loss_save
            best_swa_epoch = epoch
            custom_update_bn(dataloader, swa_model, num_steps, nb_conditions, alpha_cumprod, device=device)
            save_dir = model_save_dir if model_save_dir is not None else (f"../../models/{run_id}/" if run_id is not None else "../../models/prototyping/")
            os.makedirs(save_dir, exist_ok=True)
            filename = f"best_swa_model_epoch_{epoch}_steps_{num_steps}.pt"
            for f in os.listdir(save_dir):
                if f.startswith("best_swa_model_epoch"):
                    os.remove(os.path.join(save_dir, f))
            torch.save(swa_model.state_dict(), os.path.join(save_dir, filename))
        
        if not (swa and epoch > swa_start):
            scheduler.step(avg_schedular_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] Total Loss: {avg_loss:.4f} = Diff: {avg_loss_diff:.4f} + Rec: {avg_loss_rec:.4f} + KL: {avg_loss_kl:.8f}, SWA Loss: {avg_swa_loss:.4f}")
    
    # Return four values if SWA is enabled; otherwise, return only best_loss and best_epoch.
    if swa:
        return best_loss, best_epoch, best_swa_loss, best_swa_epoch
    else:
        return best_loss, best_epoch