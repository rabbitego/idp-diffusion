#!/usr/bin/env python
"""Build per-residue-class Ramachandran KDE tables from PED structures.

CRITICAL methodological note baked into this script: the KDE is built from the
*same disordered-protein source* used for training (PED entries, or a curated
disordered-region set), NOT from a folded-protein Ramachandran map. IDP ensembles
are PPII-enriched and shifted relative to folded proteins, so seeding the
regulariser with folded statistics would bias the model away from the very
landscape it is meant to learn. This is a deliberate, documented choice.

Usage
-----
    python scripts/build_ramachandran_kde.py \
        --ped-dir /path/to/ped_structures \
        --out artifacts/rama_kde.pt \
        --bandwidth-deg 15

Run this where the PED files and dependencies (biotite) are available.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idpdiff.constants import RAMA_CLASSES
from idpdiff.data.torsions import extract_entry
from idpdiff.diffusion.losses import RamachandranKDE, RamachandranKDEConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ped-dir", required=True, help="dir of PED .pdb/.cif files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bandwidth-deg", type=float, default=15.0)
    ap.add_argument("--grid-size", type=int, default=72)
    ap.add_argument("--glob", default="*.pdb")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.ped_dir, args.glob)))
    if not paths:
        raise SystemExit(f"no files matching {args.glob} in {args.ped_dir}")

    # Collect angle samples per residue class across the whole corpus.
    samples = {name: [] for name in RAMA_CLASSES}
    for p in paths:
        entry = extract_entry(p)
        # angles: (n_models, L, 2); class_ids: (L,)
        for m in range(entry.n_models):
            ang = entry.angles[m]  # (L, 2)
            for i in range(entry.length):
                a = ang[i]
                if np.isnan(a).any():
                    continue
                cls = RAMA_CLASSES[entry.class_ids[i]]
                samples[cls].append(a)
        print(f"processed {entry.entry_id}: {entry.n_models} models, L={entry.length}")

    angles_by_class = {
        name: (torch.tensor(np.array(v), dtype=torch.float32) if v else torch.empty(0, 2))
        for name, v in samples.items()
    }
    for name in RAMA_CLASSES:
        print(f"  class {name}: {len(samples[name])} (phi,psi) samples")

    cfg = RamachandranKDEConfig(bandwidth_deg=args.bandwidth_deg, grid_size=args.grid_size)
    kde = RamachandranKDE.from_angle_samples(angles_by_class, cfg)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kde.save(args.out)
    print(f"saved KDE -> {args.out}")


if __name__ == "__main__":
    main()
