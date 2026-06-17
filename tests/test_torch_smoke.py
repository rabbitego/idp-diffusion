"""End-to-end smoke test for the torch-dependent stack.

Run this on a machine with torch installed (no GPU required -- it is sized to
finish in seconds on CPU). It verifies that the model, wrapped diffusion, losses,
KDE regulariser, and a single training step all fit together and that sampling
produces angles of the right shape on the torus. It does NOT check scientific
quality -- that is what the real training run and ``sample_and_eval.py`` are for.

    python -m pytest tests/test_torch_smoke.py -v
    # or
    python tests/test_torch_smoke.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _skip_if_no_torch():
    if not HAVE_TORCH:
        print("torch not installed -- skipping torch smoke tests")
        sys.exit(0)


def test_forward_and_sample_shapes():
    _skip_if_no_torch()
    from idpdiff.models.denoiser import TorsionDenoiser, ModelConfig
    from idpdiff.diffusion.wrapped import WrappedDiffusion, DiffusionConfig
    from idpdiff.utils.angles import angles_to_features, wrap_to_pi

    B, L, D = 2, 16, 64
    cfg = ModelConfig(seq_embed_dim=D, width=64, depth=2, n_heads=4)
    model = TorsionDenoiser(cfg).eval()
    diff = WrappedDiffusion(DiffusionConfig(num_steps=20), device="cpu")

    emb = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    angles = (torch.rand(B, L, 2) * 2 * math.pi) - math.pi

    t = torch.randint(0, 20, (B,))
    pred = model(angles_to_features(angles), t, emb, mask)
    assert pred.shape == (B, L, 4)

    with torch.no_grad():
        sampled = diff.sample(model, emb, mask, L)
    assert sampled.shape == (B, L, 2)
    assert torch.all(sampled <= math.pi + 1e-4) and torch.all(sampled >= -math.pi - 1e-4)


def test_forward_noise_concentration():
    """Forward kernel: low t stays near x0, high t approaches uniform."""
    _skip_if_no_torch()
    from idpdiff.diffusion.wrapped import WrappedDiffusion, DiffusionConfig
    from idpdiff.utils.angles import angular_difference

    diff = WrappedDiffusion(DiffusionConfig(num_steps=1000), device="cpu")
    x0 = (torch.rand(4000, 1, 2) * 2 * math.pi) - math.pi
    lo = diff.q_sample(x0, torch.zeros(4000, dtype=torch.long))
    hi = diff.q_sample(x0, torch.full((4000,), 999, dtype=torch.long))
    d_lo = angular_difference(lo, x0).abs().mean().item()
    d_hi = angular_difference(hi, x0).abs().mean().item()
    assert d_lo < d_hi
    assert d_lo < 0.2  # nearly clean at t=0


def test_single_training_step_decreases_loss_on_toy():
    """One model should be able to overfit a single tiny batch quickly."""
    _skip_if_no_torch()
    from idpdiff.models.denoiser import TorsionDenoiser, ModelConfig
    from idpdiff.diffusion.wrapped import WrappedDiffusion, DiffusionConfig
    from idpdiff.diffusion.losses import angular_reconstruction_loss
    from idpdiff.utils.angles import angles_to_features

    torch.manual_seed(0)
    B, L, D = 4, 8, 32
    cfg = ModelConfig(seq_embed_dim=D, width=64, depth=2, n_heads=4, dropout=0.0)
    model = TorsionDenoiser(cfg).train()
    diff = WrappedDiffusion(DiffusionConfig(num_steps=50), device="cpu")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    emb = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    angles = (torch.rand(B, L, 2) * 2 * math.pi) - math.pi

    first, last = None, None
    for it in range(50):
        t = torch.randint(0, 50, (B,))
        x_t, x0 = diff.training_targets(angles, t)
        pred = model(angles_to_features(x_t), t, emb, mask)
        loss = angular_reconstruction_loss(pred, x0, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it == 0:
            first = loss.item()
        last = loss.item()
    assert last < first  # learning happened


def test_kde_regularizer_finite_and_differentiable():
    _skip_if_no_torch()
    from idpdiff.diffusion.losses import (
        RamachandranKDE, RamachandranKDEConfig, RAMA_CLASSES,
    )
    samples = {
        name: (torch.rand(500, 2) * 2 * math.pi - math.pi) for name in RAMA_CLASSES
    }
    kde = RamachandranKDE.from_angle_samples(samples, RamachandranKDEConfig(grid_size=36))
    feats = torch.randn(2, 10, 4, requires_grad=True)
    class_ids = torch.zeros(2, 10, dtype=torch.long)
    mask = torch.ones(2, 10, dtype=torch.bool)
    reg = kde.regularizer(feats, class_ids, mask)
    assert torch.isfinite(reg)
    reg.backward()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()


if __name__ == "__main__":
    _skip_if_no_torch()
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} torch smoke tests passed")
