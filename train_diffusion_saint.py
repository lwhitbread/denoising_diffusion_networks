"""train_diffusion_saint.py
Utility wrapper that plugs the SAINT-based denoiser into the existing
`train_diffusion_model` loop.
"""
from __future__ import annotations

from typing import Optional
import torch

from train_diffusion import train_diffusion_model  # noqa: F401 – lives in the same directory
from saint_tabular import SAINTDiffusionModel


def build_saint_diffusion_model(
    data_dim: int,
    cond_dim: int,
    saint_kwargs: Optional[dict] = None,
    device: torch.device | str = "cpu",
) -> SAINTDiffusionModel:
    model = SAINTDiffusionModel(
        data_dim=data_dim,
        cond_dim=cond_dim,
        saint_kwargs=saint_kwargs,
    )
    return model.to(device)


def train_diffusion_saint_model(
    data,
    nb_conditions: int,
    num_steps: int,
    alpha_cumprod: torch.Tensor,
    saint_kwargs: Optional[dict] = None,
    model: SAINTDiffusionModel | None = None,
    **train_kwargs,
):
    device = train_kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if model is None:
        model = build_saint_diffusion_model(
            data_dim=data.shape[1] - nb_conditions,
            cond_dim=nb_conditions,
            saint_kwargs=saint_kwargs,
            device=device,
        )
    best_loss, best_epoch, *rest = train_diffusion_model(
        data=data,
        nb_conditions=nb_conditions,
        model=model,
        num_steps=num_steps,
        alpha_cumprod=alpha_cumprod,
        **train_kwargs,
    )
    return best_loss, best_epoch, model
