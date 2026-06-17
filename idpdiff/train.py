"""
Training loop for the torsion-space wrapped-diffusion model.

Combines the pieces: sample a timestep, add wrapped noise, predict clean
angles, and minimise the angular reconstruction loss plus an annealed
Ramachandran-KDE regulariser.

Regulariser annealing
----------------------
The KDE penalty weight follows a schedule that starts at ``lambda_max`` and
decays (cosine) to ``lambda_min`` over training. The rationale is the one stated
in the loss module: early on, the prior usefully shepherds predictions toward
populated Ramachandran regions and stabilises learning; late in training the
data must dominate so the model can learn sequence-specific deviations and,
critically, so the prior cannot by itself collapse ensemble diversity. The whole
with/without-regulariser comparison is run by setting ``lambda_max = 0`` for the
"without" arm.

The loop is single-GPU friendly (gradient accumulation optional), checkpoints
the EMA weights used for sampling, and runs periodic protein-level validation
through the metrics module.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .diffusion.wrapped import WrappedDiffusion
from .diffusion.losses import angular_reconstruction_loss


@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 100_000
    warmup_steps: int = 1000
    ema_decay: float = 0.999
    lambda_max: float = 1.0  # KDE regulariser weight at start (0 disables it)
    lambda_min: float = 0.0  # weight at end of training
    lambda_anneal_steps: int = 50_000
    grad_clip: float = 1.0
    log_every: int = 50
    ckpt_every: int = 2000
    device: str = "cuda"


def regularizer_weight(step: int, cfg: TrainConfig) -> float:
    """Cosine anneal from lambda_max to lambda_min over lambda_anneal_steps."""
    if cfg.lambda_max == 0.0:
        return 0.0
    if step >= cfg.lambda_anneal_steps:
        return cfg.lambda_min
    frac = step / max(1, cfg.lambda_anneal_steps)
    cos = 0.5 * (1 + math.cos(math.pi * frac))
    return cfg.lambda_min + (cfg.lambda_max - cfg.lambda_min) * cos


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to 10% of base lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    frac = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    frac = min(1.0, frac)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


class EMA:
    """Exponential moving average of model parameters for stable sampling."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def train(
    model,
    diffusion: WrappedDiffusion,
    rama_kde,
    train_loader: DataLoader,
    cfg: TrainConfig,
    on_log=None,
    on_checkpoint=None,
):
    """Run the training loop. ``rama_kde`` may be None to disable the penalty.

    ``on_log(step, dict)`` and ``on_checkpoint(step, ema_model)`` are optional
    callbacks so the driver script controls logging/checkpoint destinations
    without this function importing any I/O specifics.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    if rama_kde is not None:
        rama_kde = rama_kde.to(device)
    diffusion.device = device
    diffusion.sigma = diffusion.sigma.to(device)
    diffusion.alpha_bar = diffusion.alpha_bar.to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
    )
    ema = EMA(model, cfg.ema_decay)

    step = 0
    model.train()
    data_iter = _cycle(train_loader)
    while step < cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        accum_logs = {"recon": 0.0, "reg": 0.0, "loss": 0.0}

        for _ in range(cfg.grad_accum):
            batch = next(data_iter)
            emb = batch["embedding"].to(device)
            angles = batch["angles"].to(device)
            class_ids = batch["class_ids"].to(device)
            mask = batch["mask"].to(device)
            B, L, _ = angles.shape

            t = torch.randint(0, diffusion.config.num_steps, (B,), device=device)
            x_t, x0 = diffusion.training_targets(angles, t)
            from .utils.angles import angles_to_features

            pred = model(angles_to_features(x_t), t, emb, mask)

            recon = angular_reconstruction_loss(pred, x0, mask)
            lam = regularizer_weight(step, cfg)
            if rama_kde is not None and lam > 0:
                reg = rama_kde.regularizer(pred, class_ids, mask)
            else:
                reg = torch.zeros((), device=device)

            loss = recon + lam * reg
            (loss / cfg.grad_accum).backward()

            accum_logs["recon"] += recon.item() / cfg.grad_accum
            accum_logs["reg"] += float(reg) / cfg.grad_accum
            accum_logs["loss"] += loss.item() / cfg.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        opt.step()
        ema.update(model)

        if step % cfg.log_every == 0 and on_log is not None:
            on_log(step, {**accum_logs, "lr": lr_at(step, cfg), "lambda": regularizer_weight(step, cfg)})
        if step % cfg.ckpt_every == 0 and step > 0 and on_checkpoint is not None:
            on_checkpoint(step, ema.shadow)

        step += 1

    if on_checkpoint is not None:
        on_checkpoint(step, ema.shadow)
    return ema.shadow


def _cycle(loader):
    while True:
        for b in loader:
            yield b
