#!/usr/bin/env python
"""Train the torsion-space wrapped-diffusion model.

Runs one arm of the central experiment. Use ``--lambda-max 0`` for the
"without Ramachandran regularisation" ablation arm and a positive value for the
"with" arm; everything else held identical. This single flag is what makes the
headline comparison clean.

Usage
-----
    python scripts/train_model.py \
        --dataset artifacts/dataset.pt \
        --rama-kde artifacts/rama_kde.pt \
        --out-dir runs/with_reg \
        --lambda-max 1.0 \
        --esm-model esm2_t33_650M_UR50D

Add ``--mock-esm`` to run end-to-end without downloading ESM weights (uses the
deterministic MockEmbedder; for pipeline checks only, not for real results).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idpdiff.data.torsions import TorsionEntry
from idpdiff.data.dataset import (
    TorsionDataset, entries_to_conformers, collate, Conformer,
)
from idpdiff.data.embeddings import ESMEmbedder, MockEmbedder, ESM_WIDTHS
from idpdiff.diffusion.wrapped import WrappedDiffusion, DiffusionConfig
from idpdiff.diffusion.losses import RamachandranKDE
from idpdiff.models.denoiser import TorsionDenoiser, ModelConfig
from idpdiff.train import train, TrainConfig


def load_entries(dataset_path, split):
    blob = torch.load(dataset_path, map_location="cpu", weights_only=False)
    ids = set(blob["splits"][split])
    entries = [
        TorsionEntry(d["entry_id"], d["sequence"], d["angles"], d["class_ids"])
        for d in blob["entries"]
        if d["entry_id"] in ids
    ]
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rama-kde", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lambda-max", type=float, default=1.0)
    ap.add_argument("--lambda-anneal-steps", type=int, default=50_000)
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--mock-esm", action="store_true")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--num-diffusion-steps", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    # Data
    entries = load_entries(args.dataset, "train")
    conformers = entries_to_conformers(entries)
    print(f"train conformers: {len(conformers)} from {len(entries)} proteins")

    if args.mock_esm:
        embedder = MockEmbedder(dim=ESM_WIDTHS.get(args.esm_model, 1280))
    else:
        embedder = ESMEmbedder(args.esm_model, device=args.device)

    dataset = TorsionDataset(conformers, embedder)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, drop_last=True,
    )

    # Model + diffusion
    model_cfg = ModelConfig(
        seq_embed_dim=embedder.dim, width=args.width, depth=args.depth
    )
    model = TorsionDenoiser(model_cfg)
    print(f"model parameters: {model.num_parameters()/1e6:.2f}M")

    diffusion = WrappedDiffusion(
        DiffusionConfig(num_steps=args.num_diffusion_steps), device=args.device
    )

    rama_kde = None
    if args.rama_kde and args.lambda_max > 0:
        rama_kde = RamachandranKDE.load(args.rama_kde)
        print(f"loaded Ramachandran KDE from {args.rama_kde}")

    train_cfg = TrainConfig(
        lr=args.lr, batch_size=args.batch_size, max_steps=args.max_steps,
        lambda_max=args.lambda_max, lambda_anneal_steps=args.lambda_anneal_steps,
        device=args.device,
    )

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_fh = open(log_path, "a")

    def on_log(step, d):
        d = {"step": step, **d}
        log_fh.write(json.dumps(d) + "\n")
        log_fh.flush()
        print(f"step {step}: loss={d['loss']:.4f} recon={d['recon']:.4f} "
              f"reg={d['reg']:.4f} lambda={d['lambda']:.3f}")

    def on_checkpoint(step, ema_model):
        path = os.path.join(args.out_dir, f"ema_step{step}.pt")
        torch.save({"model": ema_model.state_dict(), "model_cfg": model_cfg.__dict__}, path)
        # also keep a 'latest' pointer
        torch.save({"model": ema_model.state_dict(), "model_cfg": model_cfg.__dict__},
                   os.path.join(args.out_dir, "ema_latest.pt"))
        print(f"  checkpoint -> {path}")

    ema_model = train(model, diffusion, rama_kde, loader, train_cfg, on_log, on_checkpoint)
    log_fh.close()
    print("training complete.")


if __name__ == "__main__":
    main()
