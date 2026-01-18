"""modules_2.py (legacy)

Core model components (MLP backbone) for the normative diffusion evaluation suite.

This filename is a legacy internal name kept to preserve backwards compatibility
with previously generated results. New code should import from `diffusion_models.py`,
which re-exports the public symbols from this module.

This module intentionally preserves class/function behavior so the evaluation
suite reproduces the published outputs under `results_full_eval/`.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional Cauchy Activation (unchanged)
class CauchyActivation(nn.Module):
    def __init__(self) -> None:
        super(CauchyActivation, self).__init__()
        self.lambda_1 = nn.Parameter(torch.tensor(1.0))
        self.lambda_2 = nn.Parameter(torch.tensor(1.0))
        self.d = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        # Computes f(x) = λ₁*x/(x²+d²)+λ₂/(x²+d²)
        x2_d2 = x**2 + self.d**2
        return self.lambda_1 * x / x2_d2 + self.lambda_2 / x2_d2

# =================== Conditional Autoencoder Modules ===================
# For the encoder and decoder, we offer two options:
#   "concat" – simply concatenate the condition vector to the input (or latent)
#   or a conditional strategy (using FiLM, hypernet_base, or hypernet_moe) applied at each layer.
#
# ConditionalVariationalEncoder:
class ConditionalVariationalEncoder(nn.Module):
    """
    A VAE-style encoder that conditions on extra covariates.
    Options:
      cond_method = "concat": simply concatenate condition (cond) to the data (x)
                        otherwise: use a conditional MLP where each layer is a conditional layer.
    """
    def __init__(self,
                 input_dim, 
                 cond_dim, 
                 latent_dim, 
                 hidden_dims=[64,64],
                 batch_norm=True, 
                 activation="ReLU",
                 use_dropout=False, 
                 dropout_rate=0.1,
                 use_variational=True,
                 use_conditioning=True,
                 cond_method="concat",
                 nb_experts=4,
                 use_fan=False,
                 dp_ratio=0.25,
                 ) -> None:
        super(ConditionalVariationalEncoder, self).__init__()
        assert cond_method in ["concat", "film", "hypernet_base", "hypernet_moe"], \
            "Invalid cond_method. Options: 'concat', 'film', 'hypernet_base', or 'hypernet_moe'."
        self.use_variational = use_variational
        self.cond_method = cond_method
        self.act = activation
        self.activation = getattr(nn, activation)() if activation != "cauchy" else CauchyActivation()
        self.use_cond = use_conditioning
        layers = []
        if self.use_cond:
            if self.cond_method == "concat":
                total_input_dim = input_dim + cond_dim
                prev_dim = total_input_dim
                for h in hidden_dims:
                    if use_fan:
                        layers.append(FANLinearLayer(
                            prev_dim, 
                            h, 
                            dp_ratio=dp_ratio, 
                            bias=True, 
                            activation=activation,
                            batch_norm=batch_norm,
                            use_dropout=use_dropout,
                            dropout_rate=dropout_rate,
                        ))
                    else:
                        layers.append(nn.Linear(prev_dim, h))
                    if batch_norm and not use_fan:
                        layers.append(nn.BatchNorm1d(h))
                    if use_dropout and not use_fan:
                        layers.append(nn.Dropout(dropout_rate))
                    if not use_fan:
                        layers.append(self.activation)
                    prev_dim = h
            else:
                # For conditional strategies, process x alone through a stack of conditional layers.
                prev_dim = input_dim + cond_dim
                for h in hidden_dims:
                    if self.cond_method == "film":
                        if use_fan:
                            layers.append(FAN_FiLMLinearLayer(
                                prev_dim, 
                                h, 
                                cond_dim, 
                                dp_ratio=dp_ratio,
                                bias=True,
                                batch_norm=batch_norm,
                                activation=activation,
                                use_dropout=use_dropout,
                                dropout_rate=dropout_rate,
                            ))
                        else:
                            layers.append(FiLMLinearLayer(
                                prev_dim, 
                                h, 
                                cond_dim, 
                                bias=True,
                                batch_norm=batch_norm,
                            ))
                    elif self.cond_method == "hypernet_base":
                        if use_fan:
                            layers.append(FANHyperNetLinearLayer(
                                prev_dim, 
                                h, 
                                cond_dim, 
                                hidden_size=128,
                                dp_ratio=dp_ratio,
                                activation=activation,
                                batch_norm=False,
                            ))
                        else:
                            layers.append(HyperNetLinearLayer(
                                prev_dim, 
                                h, 
                                cond_dim,
                                activation=activation,
                                batch_norm=False,
                            ))
                        if batch_norm:
                            layers.append(nn.BatchNorm1d(h))
                    elif self.cond_method == "hypernet_moe":
                        if use_fan:
                            layers.append(FANMoE_HyperNetLinearLayer(
                                prev_dim, 
                                h, 
                                cond_dim, 
                                num_experts=nb_experts,
                                dp_ratio=dp_ratio, 
                                activation=activation,
                                batch_norm=False,
                            ))
                        else:
                            layers.append(MoE_HyperNetLinearLayer(
                                prev_dim, 
                                h, 
                                cond_dim, 
                                num_experts=nb_experts, 
                                activation=activation,
                                batch_norm=False,
                            ))
                        if batch_norm:
                            layers.append(nn.BatchNorm1d(h))
                    else:
                        raise ValueError("Invalid cond_method.")
                    
                    # if batch_norm:
                    #     layers.append(nn.BatchNorm1d(h))

                    if use_dropout and not use_fan:
                        layers.append(nn.Dropout(dropout_rate))
                    if not use_fan:
                        layers.append(self.activation)
                    prev_dim = h
        else:
            prev_dim = input_dim
            for h in hidden_dims:
                if use_fan:
                    layers.append(FANLinearLayer(
                        prev_dim, 
                        h, 
                        dp_ratio=dp_ratio, 
                        bias=True, 
                        activation=activation,
                        batch_norm=batch_norm,
                        use_dropout=use_dropout,
                        dropout_rate=dropout_rate,
                    ))
                else:
                    layers.append(nn.Linear(prev_dim, h))
                if batch_norm and not use_fan:
                    layers.append(nn.BatchNorm1d(h))
                if use_dropout and not use_fan:
                    layers.append(nn.Dropout(dropout_rate))
                if not use_fan:
                    layers.append(self.activation)
                prev_dim = h


        self.net = nn.Sequential(*layers)
        if self.use_variational:
            self.fc_mu = nn.Linear(prev_dim, latent_dim)
            self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        else:
            self.fc = nn.Linear(prev_dim, latent_dim)

        self._init_weights()

    def _init_weights(self, init_type: str = "kaiming_normal_") -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_normal_":
                    if self.act == "SELU":
                        nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
                    nn.init.kaiming_normal_(layer.weight)
                else:
                    getattr(nn.init, init_type)(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, FiLMLinearLayer):
                if init_type == "kaiming_normal_":
                    if self.act == "SELU":
                        nn.init.kaiming_normal_(layer.linear.weight, nonlinearity="linear")
                    nn.init.kaiming_normal_(layer.linear.weight)
                else:
                    getattr(nn.init, init_type)(layer.linear.weight)
                if layer.linear.bias is not None:
                    nn.init.zeros_(layer.linear.bias)
            elif isinstance(layer, MoE_HyperNetLinearLayer):
                nn.init.normal_(layer.expert_weights, mean=0.0, std=1e-4)
                nn.init.zeros_(layer.expert_biases)
                for m in layer.gating.modules():
                    if isinstance(m, nn.Linear):
                        if self.act == "SELU":
                            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            elif isinstance(layer, HyperNetLinearLayer):
                nn.init.kaiming_normal_(layer.base_weight)
                nn.init.zeros_(layer.base_bias)
                for m in layer.hypernet.modules():
                    if isinstance(m, nn.Linear):
                        if self.act == "SELU":
                            nn.init.kaiming_normal_(m.weight, nonlinearity = "linear")
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)     

        # init the weights of the mu and logvar layers
        if self.use_variational:
            nn.init.kaiming_normal_(self.fc_mu.weight)
            nn.init.kaiming_normal_(self.fc_logvar.weight)
            nn.init.zeros_(self.fc_mu.bias)
            nn.init.zeros_(self.fc_logvar.bias)
        else:
            nn.init.kaiming_normal_(self.fc.weight)
            nn.init.zeros_(self.fc.bias) 
    
    def reparameterize(self, mu, logvar, use_mu=False):
        if use_mu:
            return mu
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, cond, use_mu=False):
        if self.use_cond:
            x_in = torch.cat([x, cond], dim=1)
            if self.cond_method == "concat":
                # x_in = torch.cat([x, cond], dim=1)
                h = self.net(x_in)
            else:
                h = x_in
                for layer in self.net:
                    if isinstance(layer, (
                        FiLMLinearLayer, 
                        HyperNetLinearLayer, 
                        MoE_HyperNetLinearLayer,
                        FANHyperNetLinearLayer,
                        FANMoE_HyperNetLinearLayer,
                        FAN_FiLMLinearLayer,
                    )):
                        h = layer(h, cond)
                    else:
                        h = layer(h)
        else:
            h = self.net(x)
        
        if self.use_variational:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar, use_mu=use_mu)
            # regularise the latent space
            # z = z / torch.norm(z, dim=1, keepdim=True)
            return z, mu, logvar
        else:
            z = self.fc(h)
            # z = z / torch.norm(z, dim=1, keepdim=True)
            return z, None, None

# ConditionalLatentDecoder:
class ConditionalLatentDecoder(nn.Module):
    """
    A decoder that conditions on covariates.
    Options:
      cond_method = "concat": simply concatenates the latent z with cond at the input.
                        Otherwise, uses conditional layers.
    """
    def __init__(self,
                 latent_dim, 
                 cond_dim, 
                 output_dim,
                 hidden_dims=[64,64],
                 batch_norm=True,
                 activation="ReLU",
                 use_dropout=False,
                 dropout_rate=0.1,
                 cond_method="concat",
                 nb_experts=4,
                 use_fan=False,
                 dp_ratio=0.25,
    ) -> None:
        super(ConditionalLatentDecoder, self).__init__()
        self.cond_method = cond_method
        self.act = activation
        self.activation = getattr(nn, activation)() if activation != "cauchy" else CauchyActivation()
        layers = []
        if self.cond_method == "concat":
            total_input_dim = latent_dim + cond_dim
            prev_dim = total_input_dim
            for h in hidden_dims:
                if use_fan:
                    layers.append(FANLinearLayer(
                        prev_dim, 
                        h, 
                        dp_ratio=dp_ratio, 
                        bias=True, 
                        activation=activation,
                        batch_norm=batch_norm,
                        use_dropout=use_dropout,
                        dropout_rate=dropout_rate,
                    ))
                else:
                    layers.append(nn.Linear(prev_dim, h))
                    if batch_norm:
                        layers.append(nn.BatchNorm1d(h))
                    if use_dropout:
                        layers.append(nn.Dropout(dropout_rate))
                    layers.append(self.activation)
                prev_dim = h
        else:
            prev_dim = latent_dim + cond_dim
            for h in hidden_dims:
                if self.cond_method == "film":
                    if use_fan:
                        layers.append(FAN_FiLMLinearLayer(
                            prev_dim, 
                            h, 
                            cond_dim, 
                            dp_ratio=dp_ratio,
                            bias=True,
                            batch_norm=batch_norm,
                            activation=activation,
                            use_dropout=use_dropout,
                            dropout_rate=dropout_rate,
                        ))
                    else:
                        layers.append(FiLMLinearLayer(prev_dim, h, cond_dim, bias=True, batch_norm=batch_norm))
                elif self.cond_method == "hypernet_base":
                    if use_fan:
                        layers.append(FANHyperNetLinearLayer(
                            prev_dim, 
                            h, 
                            cond_dim, 
                            hidden_size=128,
                            dp_ratio=dp_ratio,
                            activation=activation,
                            batch_norm=False,
                        ))
                    else:
                        layers.append(HyperNetLinearLayer(
                            prev_dim, 
                            h, 
                            cond_dim,
                            activation=activation,
                            batch_norm=False,
                        ))
                    if batch_norm:
                        layers.append(nn.BatchNorm1d(h))
                elif self.cond_method == "hypernet_moe":
                    if use_fan:
                        layers.append(FANMoE_HyperNetLinearLayer(
                            prev_dim, 
                            h, 
                            cond_dim, 
                            num_experts=nb_experts,
                            dp_ratio=dp_ratio, 
                            activation=activation,
                            batch_norm=False,
                        ))
                    else:
                        layers.append(MoE_HyperNetLinearLayer(
                            prev_dim, 
                            h, 
                            cond_dim, 
                            num_experts=nb_experts, 
                            activation=activation,
                            batch_norm=False,
                        ))
                    if batch_norm:
                        layers.append(nn.BatchNorm1d(h))
                else:
                    raise ValueError("Invalid cond_method.")
                
                # if batch_norm:
                #     layers.append(nn.BatchNorm1d(h))

                if use_dropout and not use_fan:
                    layers.append(nn.Dropout(dropout_rate))
                if not use_fan:
                    layers.append(self.activation)
                prev_dim = h
        
        if self.cond_method == "concat":
            layers.append(nn.Linear(prev_dim, output_dim))
        elif self.cond_method == "film":
            layers.append(FiLMLinearLayer(prev_dim, output_dim, cond_dim, bias=True, batch_norm=False))
        elif self.cond_method == "hypernet_base":
            layers.append(HyperNetLinearLayer(
                prev_dim, 
                output_dim, 
                cond_dim,
                activation=activation,
                batch_norm=False,
        ))
        elif self.cond_method == "hypernet_moe":
            layers.append(MoE_HyperNetLinearLayer(
                prev_dim, 
                output_dim, 
                cond_dim, 
                num_experts=nb_experts, 
                activation=activation,
                batch_norm=False,
        ))
        self.decoder = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self, init_type: str = "kaiming_normal_") -> None:
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_normal_":
                    if self.act == "SELU":
                        nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
                    nn.init.kaiming_normal_(layer.weight)
                else:
                    getattr(nn.init, init_type)(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, FiLMLinearLayer):
                if init_type == "kaiming_normal_":
                    if self.act == "SELU":
                        nn.init.kaiming_normal_(layer.linear.weight, nonlinearity="linear")
                    nn.init.kaiming_normal_(layer.linear.weight)
                else:
                    getattr(nn.init, init_type)(layer.linear.weight)
                if layer.linear.bias is not None:
                    nn.init.zeros_(layer.linear.bias)
            elif isinstance(layer, MoE_HyperNetLinearLayer):
                nn.init.normal_(layer.expert_weights, mean=0.0, std=1e-4)
                nn.init.zeros_(layer.expert_biases)
                for m in layer.gating.modules():
                    if isinstance(m, nn.Linear):
                        if self.act == "SELU":
                            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            elif isinstance(layer, HyperNetLinearLayer):
                nn.init.kaiming_normal_(layer.base_weight)
                nn.init.zeros_(layer.base_bias)
                for m in layer.hypernet.modules():
                    if isinstance(m, nn.Linear):
                        if self.act == "SELU":
                            nn.init.kaiming_normal_(m.weight, nonlinearity = "linear")
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
    
    def forward(self, z, cond):
        z_in = torch.cat([z, cond], dim=1)
        if self.cond_method == "concat":
            return self.decoder(z_in)
        else:
            h = z_in
            for layer in self.decoder:
                if isinstance(layer, (
                    FiLMLinearLayer, 
                    HyperNetLinearLayer, 
                    MoE_HyperNetLinearLayer,
                    FANHyperNetLinearLayer,
                    FANMoE_HyperNetLinearLayer,
                    FAN_FiLMLinearLayer,
                )):
                    h = layer(h, cond)
                else:
                    h = layer(h)
            return h

# The conditional versions of HyperNetLinearLayer, MoE_HyperNetLinearLayer, FiLMGenerator, and FiLMLinearLayer remain unchanged.
# (They are as defined in your current code blocks.)

# =================== Updated Diffusion Model with Option for Data V.s. Latent Diffusion ===================
class GeneralDiffusionModel(nn.Module):
    """
    A general diffusion model that can operate in latent space OR directly on data space.
    
    If diffusion_space=="latent":
      - A conditional encoder maps x and cond to a latent z (with optional variational reparameterization).
      - Noise is added to z with a schedule, a denoiser MLP (conditional) predicts the noise,
        and a conditional decoder maps the denoised latent back to x.
    
    If diffusion_space=="data":
      - NO encoder or decoder are used.
      - Noise is added directly to x (data space) using the same schedule,
        and a conditional denoiser (with input dim = data_dim+1+cond_dim and output dim = data_dim)
        predicts the noise. The denoised data is computed by inverting the noising step.
    """
    def __init__(
            self, 
            data_dim: int, 
            latent_dim: int, 
            cond_dim: int,
            denoiser_kwargs: dict,
            vae_kwargs: dict,
            diffusion_space: str = "latent",  # either "latent" or "data"
            combine_t_cond: bool = True,
        ) -> None:
        super(GeneralDiffusionModel, self).__init__()
        assert diffusion_space in ["latent", "data"], "diffusion_space must be 'latent' or 'data'"
        self.diffusion_space = diffusion_space

        if self.diffusion_space == "latent":

            self.latent_dim = latent_dim
            self.cond_dim = cond_dim

            self.encoder = ConditionalVariationalEncoder(
                input_dim=data_dim,
                cond_dim=cond_dim,
                latent_dim=latent_dim,
                hidden_dims=vae_kwargs.get("encoder_hidden_dims", [256,128]),
                batch_norm=vae_kwargs.get("batch_norm", True),
                activation=vae_kwargs.get("activation", "ReLU"),
                use_dropout=vae_kwargs.get("use_dropout", True),
                dropout_rate=vae_kwargs.get("dropout_rate", 0.1),
                use_variational=vae_kwargs.get("use_variational", True),
                use_conditioning=vae_kwargs.get("encoder_conditioning", True),
                cond_method=vae_kwargs.get("vae_cond_method", "concat"),
                nb_experts=vae_kwargs.get("nb_experts", 4),
                use_fan=vae_kwargs.get("use_fan", False),
                dp_ratio=vae_kwargs.get("dp_ratio", 0.25),
            )
            self.decoder = ConditionalLatentDecoder(
                latent_dim=latent_dim,
                cond_dim=cond_dim,
                output_dim=data_dim,
                hidden_dims=vae_kwargs.get("decoder_hidden_dims", [128,256]),
                batch_norm=vae_kwargs.get("batch_norm", True),
                activation=vae_kwargs.get("activation", "ReLU"),
                use_dropout=vae_kwargs.get("use_dropout", False),
                dropout_rate=vae_kwargs.get("dropout_rate", 0.1),
                cond_method=vae_kwargs.get("vae_cond_method", "concat"),
                nb_experts=vae_kwargs.get("nb_experts", 4),
                use_fan=vae_kwargs.get("use_fan", False),
                dp_ratio=vae_kwargs.get("dp_ratio", 0.25),
            )

            # Build a denoiser for latent space. Its input dim = latent_dim + 1 + cond_dim,
            # and its output dim = latent_dim.
            self.denoiser = LatentDenoiser(
                latent_dim=latent_dim,
                output_dim=latent_dim,
                cond_dim=cond_dim,
                combine_t_cond=combine_t_cond,
                **denoiser_kwargs
            )
        else:  # diffusion_space=="data"
            # For data-space diffusion, we do not build an encoder/decoder.
            # Instead, we build a denoiser that works directly on data.
            # Its input dim = data_dim + 1 + cond_dim, and output dim = data_dim.
            self.denoiser = LatentDenoiser(
                latent_dim=data_dim,
                output_dim=data_dim,
                cond_dim=cond_dim,
                combine_t_cond=combine_t_cond,
                **denoiser_kwargs
            )

    def forward(
            self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
            cond: torch.Tensor,
            alpha_bar_t: torch.Tensor, 
            std_t: torch.Tensor, 
            use_mu: bool = False,
    ) -> torch.Tensor:
        if self.diffusion_space == "latent":
            z, mu, logvar = self.encoder(x, cond, use_mu=use_mu)

            eps = torch.randn_like(z)
            z_noisy = torch.sqrt(alpha_bar_t) * z + std_t * eps
            
            predicted_noise = self.denoiser(z_noisy, t, cond)
            
            x_recon = self.decoder(z, cond)
            return predicted_noise, x_recon, mu, logvar, z, z_noisy
        else:
            debug = os.environ.get("DM_DEBUG", "0") != "0"
            eps = torch.randn_like(x)
            if debug:
                def _stat(tensor):
                    tc = tensor.detach().float().cpu()
                    return float(tc.mean()), float(tc.std()), float(tc.abs().max()), bool(torch.isnan(tc).any()), bool(torch.isinf(tc).any())
                mx, sx, mxmax, xnan, xinf = _stat(x)
                ma, sa, mamax, anan, ainf = _stat(alpha_bar_t)
                ms, ss, msmax, snan, sinf = _stat(std_t)
                me, se, memax, enam, einf = _stat(eps)
                print(f"[DM_DEBUG][GDM] x mean={mx:.3e} std={sx:.3e} nan={xnan}; alpha_bar_t mean={ma:.3e} min={float(alpha_bar_t.min().cpu()):.3e} max={float(alpha_bar_t.max().cpu()):.3e}; std_t mean={ms:.3e} min={float(std_t.min().cpu()):.3e} max={float(std_t.max().cpu()):.3e}; eps mean={me:.3e} std={se:.3e} nan={enam}")
            x_noisy = torch.sqrt(alpha_bar_t) * x + std_t * eps
            if debug and (not torch.isfinite(x_noisy).all()):
                xnc = x_noisy.detach().float().cpu()
                print(f"[DM_DEBUG][GDM] x_noisy non-finite. mean={float(xnc.mean()):.3e} std={float(xnc.std()):.3e} maxabs={float(xnc.abs().max()):.3e} has_nan={bool(torch.isnan(xnc).any())} has_inf={bool(torch.isinf(xnc).any())}")
                raise RuntimeError("x_noisy became non-finite in GeneralDiffusionModel")
            predicted_noise = self.denoiser(x_noisy, t, cond)
            x_recon = (x_noisy - std_t * predicted_noise) / torch.sqrt(alpha_bar_t)
            return predicted_noise, x_recon, None, None, x, x_noisy


class HyperNetLinearLayer(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            cond_dim: int, 
            hidden_size: int=128,
            activation: str="ReLU",
            batch_norm: bool=False,
    ) -> None:
        super(HyperNetLinearLayer, self).__init__()
        self.activation = getattr(nn, activation)() if activation != "cauchy" else CauchyActivation()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_weight = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        self.base_bias   = nn.Parameter(torch.zeros(output_dim))
        self.hypernet = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            self.activation,
            nn.Linear(hidden_size, input_dim * output_dim + output_dim)
        )
        if batch_norm:
            self.bn = nn.BatchNorm1d(input_dim * output_dim + output_dim)
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        dynamic_params = self.hypernet(cond)
        if hasattr(self, 'bn'):
            dynamic_params = self.bn(dynamic_params)
        batch_size = x.size(0)
        weight_delta = dynamic_params[:, :self.input_dim * self.output_dim].view(batch_size, self.input_dim, self.output_dim)
        bias_delta   = dynamic_params[:, self.input_dim * self.output_dim:].view(batch_size, self.output_dim)
        W = self.base_weight.unsqueeze(0) + weight_delta
        b = self.base_bias.unsqueeze(0) + bias_delta
        x = x.unsqueeze(1)
        out = torch.bmm(x, W).squeeze(1) + b
        return out

class MoE_HyperNetLinearLayer(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            cond_dim: int, 
            num_experts: int = 4, 
            activation: str = "ReLU",
            batch_norm: bool = False,
    ) -> None:
        super(MoE_HyperNetLinearLayer, self).__init__()
        self.activation = getattr(nn, activation)() if activation != "cauchy" else CauchyActivation()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.expert_weights = nn.Parameter(torch.randn(num_experts, input_dim, output_dim) * 1e-4)
        self.expert_biases  = nn.Parameter(torch.zeros(num_experts, output_dim))
        layers = []
        layers.append(nn.Linear(cond_dim, num_experts * 4))
        layers.append(self.activation)
        layers.append(nn.Linear(num_experts * 4, num_experts))
        self.gating = nn.Sequential(*layers)
        if batch_norm:
            self.bn = nn.BatchNorm1d(num_experts)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gating_logits = self.gating(cond)
        if hasattr(self, 'bn'):
            gating_logits = self.bn(gating_logits)
        gating_weights = F.softmax(gating_logits, dim=1)
        dynamic_weight = torch.einsum('be,eio->bio', gating_weights, self.expert_weights)
        dynamic_bias = torch.einsum('be,eo->bo', gating_weights, self.expert_biases)
        x_unsqueezed = x.unsqueeze(1)
        out = torch.bmm(x_unsqueezed, dynamic_weight).squeeze(1) + dynamic_bias
        return out

class FiLMGenerator(nn.Module):
    def __init__(self, input_dim: int, num_features: int, bias: bool = True):
        super(FiLMGenerator, self).__init__()
        self.generator = nn.Linear(input_dim, num_features * 2, bias=bias)
    def forward(self, x: torch.Tensor):
        film_params = self.generator(x)
        gamma, beta = film_params.chunk(2, dim=1)
        return gamma, beta

class FiLMLinearLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cond_dim: int, bias: bool = True, batch_norm: bool = False):
        super(FiLMLinearLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.film_generator = FiLMGenerator(cond_dim, output_dim, bias=bias)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Debug: Print input stats
        #if torch.isnan(x).any():
        #    print("[FiLMLinearLayer] NaN in input x to linear")
        #if torch.isnan(cond).any():
        #    print("[FiLMLinearLayer] NaN in cond to film_generator")
        #print(f"[FiLMLinearLayer] x stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}, min={x.min().item():.6f}, max={x.max().item():.6f}")
        #print(f"[FiLMLinearLayer] cond stats: mean={cond.mean().item():.6f}, std={cond.std().item():.6f}, min={cond.min().item():.6f}, max={cond.max().item():.6f}")
        debug = os.environ.get("DM_DEBUG", "0") != "0"
        x_in = x
        x = self.linear(x)
        # Optional BN disable via env to debug stability
        disable_bn = os.environ.get("DM_DISABLE_FILM_BN", "0") != "0"
        if self.batch_norm and not disable_bn:
            x = self.bn(x)
        #if torch.isnan(x).any():
        #    print("[FiLMLinearLayer] NaN after linear (and BN if used)")
        #print(f"[FiLMLinearLayer] x after linear stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}, min={x.min().item():.6f}, max={x.max().item():.6f}")
        gamma, beta = self.film_generator(cond)
        # Optional clamp on FiLM parameters for stability experiments
        clamp_val_str = os.environ.get("DM_FILM_CLAMP", "")
        if clamp_val_str:
            try:
                vmax = float(clamp_val_str)
                if vmax > 0:
                    gamma = torch.clamp(gamma, -vmax, vmax)
                    beta = torch.clamp(beta, -vmax, vmax)
            except Exception:
                pass
        #if torch.isnan(gamma).any():
        #    print("[FiLMLinearLayer] NaN in gamma")
        #if torch.isnan(beta).any():
        #    print("[FiLMLinearLayer] NaN in beta")
        #print(f"[FiLMLinearLayer] gamma stats: mean={gamma.mean().item():.6f}, std={gamma.std().item():.6f}, min={gamma.min().item():.6f}, max={gamma.max().item():.6f}")
        #print(f"[FiLMLinearLayer] beta stats: mean={beta.mean().item():.6f}, std={beta.std().item():.6f}, min={beta.min().item():.6f}, max={beta.max().item():.6f}")
        out = x * (1 + gamma) + beta
        if debug and (not torch.isfinite(out).all()):
            def _stat(t, name):
                tc = t.detach().float().cpu()
                return {
                    'mean': float(tc.mean()),
                    'std': float(tc.std()),
                    'max_abs': float(tc.abs().max()),
                    'has_nan': bool(torch.isnan(tc).any()),
                    'has_inf': bool(torch.isinf(tc).any()),
                    'shape': list(tc.shape),
                }
            print("[DM_DEBUG][FiLMLinearLayer] Non-finite output. Stats:")
            print("  x_in:", _stat(x_in, 'x_in'))
            print("  linear(x):", _stat(x, 'x'))
            print("  gamma:", _stat(gamma, 'gamma'))
            print("  beta:", _stat(beta, 'beta'))
            raise RuntimeError("FiLMLinearLayer produced non-finite output")
        #if torch.isnan(out).any():
        #    print("[FiLMLinearLayer] NaN in output after FiLM transformation")
        #print(f"[FiLMLinearLayer] output stats: mean={out.mean().item():.6f}, std={out.std().item():.6f}, min={out.min().item():.6f}, max={out.max().item():.6f}")
        return out

# Updated Latent Denoiser remains essentially as before.
class LatentDenoiser(nn.Module):
    """
    The denoiser acts in latent space. It takes the noised latent (z_noisy),
    the current timestep t, and the conditioning vector cond.
    In this design, we use one of several conditioning methods (“hypernet_base”, “hypernet_moe”, or “film”)
    on a multi‐layer MLP. The input to the denoiser is the concatenation of z_noisy, t, and cond.
    """
    def __init__(self, latent_dim: int, output_dim: int,
                 nb_units: list = [256,256,256],
                 bias: bool = True,
                 use_dropout: bool = False,
                 dropout_rate: float = 0.1,
                 weight_init: str = "kaiming_normal_",
                 activation: str = "ReLU",
                 cond_method: str = "hypernet_base",
                 nb_experts: int = 4,
                 cond_dim: int = None,
                 batch_norm: bool = False,
                 use_fan: bool = False,
                 dp_ratio: float = 0.25,
                 combine_t_cond: bool = True,
    ) -> None:
        
        super(LatentDenoiser, self).__init__()
        assert cond_method in ["hypernet_base", "hypernet_moe", "film"]
        self.act = activation
        self.activation = getattr(nn, activation)() if activation != "cauchy" else CauchyActivation()
        self.layers = nn.ModuleList()
        # The input is the concatenation of z_noisy (latent_dim), t (1 dim) and cond (cond_dim).
        layer_in_dim = latent_dim + 1 + cond_dim
        len_hidden = len(nb_units)
        self.combine_t_cond = combine_t_cond
        if self.combine_t_cond:
            cond_dim = cond_dim + 1
        for idx, hidden_dim in enumerate(nb_units):
            if cond_method == "hypernet_base":
                if use_fan:
                    self.layers.append(FANHyperNetLinearLayer(
                        layer_in_dim, 
                        hidden_dim, 
                        cond_dim, 
                        hidden_size=128,
                        dp_ratio=dp_ratio,
                        activation=activation,
                        batch_norm=False,
                    ))
                else:
                    self.layers.append(HyperNetLinearLayer(layer_in_dim, hidden_dim, cond_dim))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(hidden_dim))
            
            elif cond_method == "hypernet_moe":
                if use_fan:
                    self.layers.append(FANMoE_HyperNetLinearLayer(
                        layer_in_dim, 
                        hidden_dim, 
                        cond_dim, 
                        num_experts=nb_experts,
                        dp_ratio=dp_ratio, 
                        activation=activation,
                        batch_norm=False,
                    ))
                else:
                    self.layers.append(MoE_HyperNetLinearLayer(
                        layer_in_dim, 
                        hidden_dim, 
                        cond_dim, 
                        num_experts=nb_experts, 
                        activation=activation,
                        batch_norm=False,
                        ))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(hidden_dim))
            
            elif cond_method == "film":
                # if use_fan and not the last hidden layer
                if use_fan and idx < len_hidden - 1:
                    self.layers.append(FAN_FiLMLinearLayer(
                        layer_in_dim, 
                        hidden_dim, 
                        cond_dim, 
                        dp_ratio=dp_ratio,
                        bias=bias,
                        batch_norm=True,
                        activation=activation,
                        use_dropout=use_dropout,
                        dropout_rate=dropout_rate,
                    ))
                else:
                    self.layers.append(FiLMLinearLayer(layer_in_dim, hidden_dim, cond_dim, bias=bias, batch_norm=True))
            
            else:
                if use_fan:
                    self.layers.append(FANLinearLayer(
                        layer_in_dim, 
                        hidden_dim, 
                        dp_ratio=dp_ratio, 
                        bias=bias, 
                        activation=activation,
                        batch_norm=batch_norm,
                        use_dropout=use_dropout,
                        dropout_rate=dropout_rate,
                    ))
                else:
                    self.layers.append(nn.Linear(layer_in_dim, hidden_dim, bias=bias))
                if batch_norm and not use_fan:
                    self.layers.append(nn.BatchNorm1d(hidden_dim))
            if use_dropout and not use_fan:
                self.layers.append(nn.Dropout(dropout_rate))
            if not use_fan:
                self.layers.append(self.activation)
            layer_in_dim = hidden_dim
        # Final layer to predict the noise in latent space.
        if cond_method == "hypernet_base":
            self.layers.append(HyperNetLinearLayer(layer_in_dim, output_dim, cond_dim))
        elif cond_method == "hypernet_moe":
            self.layers.append(MoE_HyperNetLinearLayer(
                layer_in_dim, 
                output_dim, 
                cond_dim, 
                num_experts=nb_experts, 
                activation=activation,
                batch_norm=False,
                ))
        elif cond_method == "film":
            self.layers.append(FiLMLinearLayer(layer_in_dim, output_dim, cond_dim, bias=bias, batch_norm=False))
        else:
            self.layers.append(nn.Linear(layer_in_dim, output_dim, bias=bias))

        self._init_weights(weight_init)
    
    def forward(self, z_noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t = t.type_as(z_noisy)
        cond = cond.type_as(z_noisy)
        debug = os.environ.get("DM_DEBUG", "0") != "0"
        if debug:
            def _stat(tensor):
                tc = tensor.detach().float().cpu()
                return float(tc.mean()), float(tc.std()), float(tc.abs().max()), bool(torch.isnan(tc).any()), bool(torch.isinf(tc).any())
            mz, sz, mzmax, znan, zinf = _stat(z_noisy)
            mt, st, mtmax, tnan, tinf = _stat(t)
            mc, sc, mcmax, cnan, cinf = _stat(cond)
            print(f"[DM_DEBUG] Inputs to denoiser: z_noisy mean={mz:.3e} std={sz:.3e} maxabs={mzmax:.3e} nan={znan} inf={zinf}; t mean={mt:.3e} std={st:.3e} nan={tnan}; cond mean={mc:.3e} std={sc:.3e} nan={cnan}")
        x = torch.cat([z_noisy, t, cond], dim=1)
        if debug and (not torch.isfinite(x).all()):
            xc = x.detach().float().cpu()
            print(f"[DM_DEBUG] Non-finite right after concat: mean={float(xc.mean()):.3e} std={float(xc.std()):.3e} maxabs={float(xc.abs().max()):.3e} has_nan={bool(torch.isnan(xc).any())} has_inf={bool(torch.isinf(xc).any())}")
            raise RuntimeError("Non-finite after concat in LatentDenoiser")
        if self.combine_t_cond:
            cond = torch.cat([t, cond], dim=1)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, (
                FiLMLinearLayer, 
                HyperNetLinearLayer, 
                MoE_HyperNetLinearLayer,
                FANHyperNetLinearLayer,
                FANMoE_HyperNetLinearLayer,
                FAN_FiLMLinearLayer,
            )):
                x = layer(x, cond)
                if debug and (not torch.isfinite(x).all()):
                    # surface the problematic layer
                    x_cpu = x.detach().float().cpu()
                    print(f"[DM_DEBUG] Non-finite after conditional layer #{idx} ({type(layer).__name__}) | mean={x_cpu.mean():.3e} std={x_cpu.std():.3e} maxabs={x_cpu.abs().max():.3e}")
                    raise RuntimeError("Non-finite in LatentDenoiser conditional layer")
            else:
                x = layer(x)
                if debug and (not torch.isfinite(x).all()):
                    x_cpu = x.detach().float().cpu()
                    print(f"[DM_DEBUG] Non-finite after layer #{idx} ({type(layer).__name__}) | mean={x_cpu.mean():.3e} std={x_cpu.std():.3e} maxabs={x_cpu.abs().max():.3e}")
                    raise RuntimeError("Non-finite in LatentDenoiser layer")
        return x
    
    def _init_weights(self, init_type: str = "kaiming_normal_") -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if init_type == "kaiming_normal_":
                    if self.act == "SELU":
                        nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
                    nn.init.kaiming_normal_(layer.weight)
                else:
                    getattr(nn.init, init_type)(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, FiLMLinearLayer):
                if init_type == "kaiming_normal_":
                    if self.act == "SELU":
                        nn.init.kaiming_normal_(layer.linear.weight, nonlinearity="linear")
                    nn.init.kaiming_normal_(layer.linear.weight)
                else:
                    getattr(nn.init, init_type)(layer.linear.weight)
                if layer.linear.bias is not None:
                    nn.init.zeros_(layer.linear.bias)
            elif isinstance(layer, MoE_HyperNetLinearLayer):
                nn.init.normal_(layer.expert_weights, mean=0.0, std=1e-4)
                nn.init.zeros_(layer.expert_biases)
                for m in layer.gating.modules():
                    if isinstance(m, nn.Linear):
                        if self.act == "SELU":
                            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            elif isinstance(layer, HyperNetLinearLayer):
                nn.init.kaiming_normal_(layer.base_weight)
                nn.init.zeros_(layer.base_bias)
                for m in layer.hypernet.modules():
                    if isinstance(m, nn.Linear):
                        if self.act == "SELU":
                            nn.init.kaiming_normal_(m.weight, nonlinearity = "linear")
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
    
#################################
# Functions for Custom BN Update and SWA LR
#################################

def custom_update_bn(
        loader: torch.utils.data.DataLoader, 
        model: nn.Module, 
        num_steps: int, 
        cond_dim: int, 
        alpha_cumprod: torch.Tensor,
        device: torch.device = None,
) -> None:
    """
    Custom BatchNorm updater that supports additional arguments in the forward pass.
    Inputs:
    - `loader`: DataLoader, should yield inputs and conditional data
    - `model`: Your model with BatchNorm layers
    - `num_steps`: Number of diffusion steps (for generating `t`)
    - `cond_dim`: Size of the conditional input vector
    - `device`: Device for model and data

    This customized version of `update_bn` will loop through the data,
    construct required inputs (`t`, `cond`), and call the forward method.
    """
    model.train()
    momenta = {}
    cond_zeros = torch.zeros(cond_dim, dtype=torch.float32, device=device)

    # Find all BatchNorm layers and reset their stats
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return  # No BatchNorm layers to update

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None

    for input_data, cond_data in loader:
        input_data = input_data.to(device)
        cond_data = cond_data.to(device)

        # Generate random diffusion step indices for the batch
        # t = torch.randint(0, num_steps, (input_data.size(0),), device=device).float().unsqueeze(-1)

        # Sample a batch of random time steps
        t = torch.randint(low=0, high=num_steps, size=(input_data.size(0),), device=device).long()
        t_float = t.unsqueeze(-1).float()
        
        # Get diffusion schedule parameters from your precomputed schedule
        alpha_t = alpha_cumprod[t].to(device).unsqueeze(-1) # shape (batch, 1)
        std_t   = torch.sqrt(1.0 - alpha_t)

        # Run forward pass
        model(input_data, t_float, cond_data, alpha_t, std_t)

        # # Run the forward pass (adjust for your model structure)
        # model(input_data, t, cond_data)

    # Restore original BatchNorm momentum values
    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]

    model.train(was_training)



# Define a function to calculate the oscillating SWA LR with final decay
def get_swa_lr(epoch, swa_start, epochs, base_swa_lr, final_lr):
    """
    Compute the SWA learning rate for a given epoch.
    
    Args:
    - epoch: Current training epoch.
    - swa_start: Epoch when SWA begins.
    - epochs: Total number of epochs.
    - base_swa_lr: Initial SWA learning rate to oscillate around.
    - final_lr: Learning rate value to decay to in the last 3% of epochs.
    
    Returns:
    - lr: Computed learning rate for the current epoch.
    """
    if epoch < swa_start:  # Before SWA starts
        return base_swa_lr
    
    swa_epochs = epochs - swa_start
    final_decay_start = epochs - int(0.05 * epochs)  # Final 3% of epochs
    # oscillation_period = (swa_epochs - final_decay_start) // 4  # Adjustable: How many epochs per oscillation

    if epoch >= final_decay_start:  # Final decay phase
        # Linearly decay during the last 3% of epochs
        decay_progress = (epoch - final_decay_start) / (epochs - final_decay_start)
        return base_swa_lr * (1 - decay_progress) + final_lr * decay_progress
    else:  # Oscillating phase
        # Sinusoidal oscillation between 50% and 150% of base_swa_lr
        oscillation_period = swa_epochs // 4  # Adjustable: How many epochs per oscillation
        oscillation = 0.5 * math.sin(2 * math.pi * (epoch - swa_start) / oscillation_period) + 1
        return base_swa_lr * oscillation
    

class FANLinearLayer(nn.Module):
    def __init__(
            self, 
            in_features, 
            out_features, 
            dp_ratio=0.25, 
            bias=True, 
            activation="ReLU",
            batch_norm=False,
            use_dropout=False,
            dropout_rate=0.1,
            ) -> None:
        super(FANLinearLayer, self).__init__()
        self.activation = getattr(nn, activation)() if activation != "cauchy" else CauchyActivation()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_norm = batch_norm
        self.use_dropout = use_dropout
        # dₚ is the number of "periodic" units (for both cosine and sine)
        self.dp = max(1, int(out_features * dp_ratio))
        # The rest of the output dimensions come from the nonperiodic branch.
        self.d_non = out_features - 2 * self.dp
        # Periodic branch: one weight matrix used for both cos and sin.
        self.weight_period = nn.Parameter(torch.randn(in_features, self.dp) * math.sqrt(2.0 / in_features))
        # Nonperiodic branch:
        if self.d_non > 0:
            self.weight_non = nn.Parameter(torch.randn(in_features, self.d_non) * math.sqrt(2.0 / in_features))
            self.batch_norm_non = nn.BatchNorm1d(self.d_non) if batch_norm else None
            self.dropout_non = nn.Dropout(dropout_rate) if use_dropout else None

        else:
            self.weight_non = None
            self.bias_non = None   
        if bias:
            self.bias_non = nn.Parameter(torch.zeros(self.d_non))
        else:
            self.register_parameter('bias_non', None)

        # # The nonlinearity for the nonperiodic branch (as in Eq. 9)
        # self.activation = activation() if isinstance(activation, type) else activation

    def forward(self, x):
        # Compute x @ Wₚ then take cosine and sine for the periodic part
        periodic = torch.matmul(x, self.weight_period)  # shape: [B, dp]
        periodic_cos = torch.cos(periodic)
        periodic_sin = torch.sin(periodic)
        if self.d_non > 0:
            non_periodic = F.linear(x, self.weight_non.t(), self.bias_non)
            if self.batch_norm:
                non_periodic = self.batch_norm_non(non_periodic)
            if self.use_dropout:
                non_periodic = self.dropout_non(non_periodic)
            non_periodic = self.activation(non_periodic)
            out = torch.cat([periodic_cos, periodic_sin, non_periodic], dim=-1)
        else:
            out = torch.cat([periodic_cos, periodic_sin], dim=-1)
        return out




class FAN_FiLMLinearLayer(nn.Module):
    def __init__(
            self, 
            in_features, 
            out_features, 
            cond_dim, 
            dp_ratio=0.25, 
            bias=True, 
            batch_norm=False, 
            activation="ReLU",
            use_dropout=False,
            dropout_rate=0.1,
    ) -> None:
        super(FAN_FiLMLinearLayer, self).__init__()
        self.fan_linear = FANLinearLayer(
            in_features, 
            out_features, 
            dp_ratio=dp_ratio, 
            bias=bias, 
            activation=activation,
            batch_norm=batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            )
        # We use the same FiLMGenerator as in your original code.
        self.film_generator = FiLMGenerator(cond_dim, out_features, bias=bias)
        # self.batch_norm = batch_norm
        # if self.batch_norm:
        #     self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, cond):
        x = self.fan_linear(x)
        # if self.batch_norm:
        #     x = self.bn(x)
        gamma, beta = self.film_generator(cond)
        return x * (1 + gamma) + beta


class FANHyperNetLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, cond_dim, hidden_size=128, dp_ratio=0.25, activation=nn.ReLU(), batch_norm=False):
        super(FANHyperNetLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dp = max(1, int(out_features * dp_ratio))
        self.d_non = out_features - 2 * self.dp

        # Base parameters for FAN branch:
        self.base_weight_period = nn.Parameter(torch.randn(in_features, self.dp) * math.sqrt(2.0 / in_features))
        if self.d_non > 0:
            self.base_weight_non = nn.Parameter(torch.randn(in_features, self.d_non) * math.sqrt(2.0 / in_features))
            self.base_bias_non = nn.Parameter(torch.zeros(self.d_non))
        else:
            self.base_weight_non = None
            self.base_bias_non = None
        
        # Build hypernet to output deltas for both branches.
        # Total number of dynamic parameters:
        total_params = in_features * self.dp
        if self.d_non > 0:
            total_params += in_features * self.d_non + self.d_non

        self.hypernet = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            activation() if isinstance(activation, type) else activation,
            nn.Linear(hidden_size, total_params)
        )
        if batch_norm:
            self.bn = nn.BatchNorm1d(total_params)
        self.activation = activation() if isinstance(activation, type) else activation

    def forward(self, x, cond):
        batch_size = x.size(0)
        delta = self.hypernet(cond)  # shape: [B, total_params]
        if hasattr(self, 'bn'):
            delta = self.bn(delta)
        # Unpack dynamic parameters.
        idx = 0
        delta_weight_period = delta[:, idx: idx + self.in_features * self.dp].view(batch_size, self.in_features, self.dp)
        idx += self.in_features * self.dp
        # Dynamic weight for periodic part.
        weight_period = self.base_weight_period.unsqueeze(0) + delta_weight_period  # shape: [B, in_features, dp]

        # Compute periodic output: for each sample, x[i] (1 x in_features) @ weight_period[i] (in_features x dp)
        x_expanded = x.unsqueeze(1)  # [B, 1, in_features]
        periodic = torch.bmm(x_expanded, weight_period).squeeze(1)  # [B, dp]
        periodic_cos = torch.cos(periodic)
        periodic_sin = torch.sin(periodic)
        
        if self.d_non > 0:
            delta_weight_non = delta[:, idx: idx + self.in_features * self.d_non].view(batch_size, self.in_features, self.d_non)
            idx += self.in_features * self.d_non
            delta_bias_non = delta[:, idx: idx + self.d_non].view(batch_size, self.d_non)
            weight_non = self.base_weight_non.unsqueeze(0) + delta_weight_non
            bias_non = self.base_bias_non.unsqueeze(0) + delta_bias_non
            non_periodic = torch.bmm(x_expanded, weight_non).squeeze(1) + bias_non  # [B, d_non]
            non_periodic = self.activation(non_periodic)
            out = torch.cat([periodic_cos, periodic_sin, non_periodic], dim=1)
        else:
            out = torch.cat([periodic_cos, periodic_sin], dim=1)
        return out


class FANMoE_HyperNetLinearLayer(nn.Module):
    """
    FANMoE_HyperNetLinearLayer implements a conditional linear transformation where the
    linear mapping is replaced by a FAN‐style transformation. The output dimension is partitioned
    into a periodic branch (which outputs both cosine and sine features) plus an (optional) nonperiodic branch.

    Here, several experts (num_experts) are maintained.
    A hypernetwork applied to the conditioning input produces dynamic delta parameters per expert,
    and a gating branch yields a softmax weight over experts. The final output is the weighted sum over experts.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
            (Note: The FAN transformation produces 2*d_p + d_non output units,
            where d_p = max(1, int(out_features * dp_ratio)) and d_non = out_features - 2*d_p.)
        cond_dim (int): Dimension of the conditioning vector.
        num_experts (int, optional): Number of experts. Default: 4.
        dp_ratio (float, optional): Ratio (default 0.25) of output units reserved for periodic branch.
        hidden_size (int, optional): Hidden layer size for the hypernetwork. Default: 128.
        activation (callable or nn.Module, optional): Activation function; default ReLU.
        batch_norm (bool, optional): Whether to use BatchNorm in the hypernet and gating network.
    """
    def __init__(
            self, 
            in_features, 
            out_features, 
            cond_dim, 
            num_experts=4, 
            dp_ratio=0.25,
            hidden_size=64, 
            activation="ReLU", 
            batch_norm=False):
        super(FANMoE_HyperNetLinearLayer, self).__init__()
        self.activation = getattr(nn, activation)() if activation != "cauchy" else CauchyActivation()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        # Determine dimensions for periodic branch: d_p for frequency encoding.
        self.dp = max(1, int(out_features * dp_ratio))
        # The nonperiodic branch (if any) covers the rest.
        self.d_non = out_features - 2 * self.dp
        
        # Base parameters for periodic branch.
        # Each expert holds a weight matrix mapping [in_features x dp]
        self.expert_base_weight_period = nn.Parameter(
            torch.randn(num_experts, in_features, self.dp) * math.sqrt(2.0 / in_features)
        )
        
        # For the nonperiodic branch (if any)
        if self.d_non > 0:
            self.expert_base_weight_non = nn.Parameter(
                torch.randn(num_experts, in_features, self.d_non) * math.sqrt(2.0 / in_features)
            )
            self.expert_base_bias_non = nn.Parameter(
                torch.zeros(num_experts, self.d_non)
            )
        else:
            self.expert_base_weight_non = None
            self.expert_base_bias_non = None
            
        # Determine total dynamic delta parameters per expert.
        total_params_per_expert = in_features * self.dp
        if self.d_non > 0:
            total_params_per_expert += in_features * self.d_non + self.d_non
        self.total_params_per_expert = total_params_per_expert
        total_dynamic_params = self.num_experts * total_params_per_expert
        
        # Hypernetwork: maps the conditioning vector to dynamic delta parameters
        self.hypernet = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            self.activation,
            nn.Linear(hidden_size, total_dynamic_params)
        )
        if batch_norm:
            self.bn = nn.BatchNorm1d(total_dynamic_params)
        
        # Gating network: maps the conditioning vector to gating weights over experts.
        self.gating = nn.Sequential(
            nn.Linear(cond_dim, num_experts * 3),
            self.activation,
            nn.Linear(num_experts * 3, num_experts)
        )
        if batch_norm:
            self.gn = nn.BatchNorm1d(num_experts)

    def forward(self, x, cond):
        batch_size = x.size(0)
        # >>> Added assert to ensure dimension compatibility.
        assert x.size(1) == self.in_features, f"FANMoE_HyperNetLinearLayer expected input dim={self.in_features} but got {x.size(1)}"
        # <<<
        
        # Generate dynamic delta parameters via the hypernetwork.
        delta = self.hypernet(cond)  # shape: [B, num_experts * total_params_per_expert]
        if hasattr(self, 'bn'):
            delta = self.bn(delta)
        delta = delta.view(batch_size, self.num_experts, self.total_params_per_expert)
        
        # Split delta for periodic branch:
        delta_period = delta[:, :, :self.in_features * self.dp]
        delta_period = delta_period.view(batch_size, self.num_experts, self.in_features, self.dp)
        
        # For the nonperiodic branch, if applicable.
        if self.d_non > 0:
            start = self.in_features * self.dp
            delta_non = delta[:, :, start: start + self.in_features * self.d_non]
            delta_non = delta_non.view(batch_size, self.num_experts, self.in_features, self.d_non)
            start2 = start + self.in_features * self.d_non
            delta_bias_non = delta[:, :, start2: start2 + self.d_non]
        
        # Add the dynamic deltas to the base parameters.
        # Periodic branch.
        weight_period = self.expert_base_weight_period.unsqueeze(0) + delta_period  # [B, num_experts, in_features, dp]
        # Compute FAN periodic transformation via einsum.
        periodic_expert = torch.einsum('bi,bnip->bnp', x, weight_period)  # [B, num_experts, dp]
        periodic_cos = torch.cos(periodic_expert)
        periodic_sin = torch.sin(periodic_expert)
        
        if self.d_non > 0:
            weight_non = self.expert_base_weight_non.unsqueeze(0) + delta_non  # [B, num_experts, in_features, d_non]
            bias_non = self.expert_base_bias_non.unsqueeze(0) + delta_bias_non  # [B, num_experts, d_non]
            nonperiodic = torch.einsum('bi,bnip->bnp', x, weight_non) + bias_non  # [B, num_experts, d_non]
            nonperiodic = self.activation(nonperiodic)
        
        # Compute gating weights.
        gating_logits = self.gating(cond)  # [B, num_experts]
        if hasattr(self, 'gn'):
            gating_logits = self.gn(gating_logits)
        gating_weights = F.softmax(gating_logits, dim=1)  # [B, num_experts]
        gating_weights_exp = gating_weights.unsqueeze(-1)  # [B, num_experts, 1]
        
        # Aggregate experts.
        periodic_cos_comb = torch.sum(periodic_cos * gating_weights_exp, dim=1)  # [B, dp]
        periodic_sin_comb = torch.sum(periodic_sin * gating_weights_exp, dim=1)  # [B, dp]
        if self.d_non > 0:
            nonperiodic_comb = torch.sum(nonperiodic * gating_weights_exp, dim=1)  # [B, d_non]
            out = torch.cat([periodic_cos_comb, periodic_sin_comb, nonperiodic_comb], dim=-1)
        else:
            out = torch.cat([periodic_cos_comb, periodic_sin_comb], dim=-1)
            
        return out