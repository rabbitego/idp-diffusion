"""
Transformer denoiser for torsion-space diffusion.

The network is an x0-predictor: it receives the noised torsion features, the
diffusion timestep, and per-residue sequence conditioning, and outputs a
prediction of the clean torsion features for every residue.

Architecture
------------
* **Inputs fused per residue.** The 4-d noised (cos, sin) features are linearly
  projected to the model width and *added* to a projection of the per-residue
  ESM embedding. Conditioning is injected per position (never pooled to a single
  global vector) because torsion angles are local quantities -- a residue's
  (phi, psi) depends mostly on its own identity and immediate neighbours, and a
  pooled vector would throw that locality away.
* **Timestep conditioning via AdaLN.** The sinusoidally embedded timestep is
  mapped to per-layer scale/shift parameters that modulate each block's
  layer-norm (adaptive layer norm, as in DiT). This is a clean, well-tested way
  to tell every layer "how noisy" the input is without burning a sequence slot.
* **Relative-position-friendly encoder.** A standard pre-norm Transformer
  encoder with sinusoidal absolute positions. For the chain lengths typical of
  PED entries this is sufficient; the code is structured so the attention block
  can be swapped for a rotary/relative variant without touching the rest.
* **Padding mask respected throughout.** Variable-length chains are right-padded
  and a key-padding mask is threaded into attention so padded positions never
  contribute.

The model is intentionally modest in size by default (a few M parameters) so it
trains comfortably on a single consumer GPU and does not overfit the small PED
corpus; width/depth are config-driven for scaling up later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Standard sinusoidal embedding of integer timesteps -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def sinusoidal_positions(length: int, dim: int, device) -> torch.Tensor:
    """Absolute sinusoidal positional encodings -> (1, length, dim)."""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
    )
    args = pos * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb.unsqueeze(0)


@dataclass
class ModelConfig:
    seq_embed_dim: int = 1280  # ESM-2 650M / ESM-1b width; projected down
    width: int = 384
    depth: int = 6
    n_heads: int = 6
    ffn_mult: int = 4
    dropout: float = 0.1
    feature_dim: int = 4  # (cos phi, sin phi, cos psi, sin psi)


class AdaLNBlock(nn.Module):
    """Pre-norm Transformer block with adaptive layer norm from the timestep."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.width, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            cfg.width, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.width, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.width, cfg.width * cfg.ffn_mult),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.width * cfg.ffn_mult, cfg.width),
        )
        # Produce 6 modulation vectors (scale/shift/gate for attn and ffn).
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cfg.width, 6 * cfg.width))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond, key_padding_mask):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.ada(cond).chunk(6, dim=-1)
        shift_a, scale_a, gate_a = (v.unsqueeze(1) for v in (shift_a, scale_a, gate_a))
        shift_f, scale_f, gate_f = (v.unsqueeze(1) for v in (shift_f, scale_f, gate_f))

        h = self.norm1(x) * (1 + scale_a) + shift_a
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + gate_a * attn_out

        h = self.norm2(x) * (1 + scale_f) + shift_f
        x = x + gate_f * self.ffn(h)
        return x


class TorsionDenoiser(nn.Module):
    """x0-prediction denoiser: (features, t, seq_embed, mask) -> features."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feat_proj = nn.Linear(cfg.feature_dim, cfg.width)
        self.seq_proj = nn.Linear(cfg.seq_embed_dim, cfg.width)
        self.t_embed = nn.Sequential(
            nn.Linear(cfg.width, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width)
        )
        self.blocks = nn.ModuleList([AdaLNBlock(cfg) for _ in range(cfg.depth)])
        self.out_norm = nn.LayerNorm(cfg.width, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(cfg.width, 2 * cfg.width))
        nn.init.zeros_(self.out_ada[-1].weight)
        nn.init.zeros_(self.out_ada[-1].bias)
        self.head = nn.Linear(cfg.width, cfg.feature_dim)

    def forward(
        self,
        features: torch.Tensor,  # (B, L, 4)
        t: torch.Tensor,  # (B,)
        seq_embedding: torch.Tensor,  # (B, L, seq_embed_dim)
        mask: torch.Tensor,  # (B, L) True at real residues
    ) -> torch.Tensor:
        B, L, _ = features.shape
        x = self.feat_proj(features) + self.seq_proj(seq_embedding)
        x = x + sinusoidal_positions(L, self.cfg.width, x.device)

        t_emb = timestep_embedding(t, self.cfg.width)
        cond = self.t_embed(t_emb)  # (B, width)

        key_padding_mask = ~mask  # True where padded -> ignored by attention

        for block in self.blocks:
            x = block(x, cond, key_padding_mask)

        shift, scale = self.out_ada(cond).chunk(2, dim=-1)
        x = self.out_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.head(x)  # (B, L, 4)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
