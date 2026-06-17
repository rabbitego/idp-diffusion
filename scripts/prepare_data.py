#!/usr/bin/env python
"""Extract torsion angles from PED structures and build protein-level splits.

Produces a single ``.pt`` cache holding all parsed entries plus the
train/val/test split (by protein, optionally by sequence cluster) so training
and evaluation read identical, reproducible data.

Usage
-----
    python scripts/prepare_data.py \
        --ped-dir /path/to/ped_structures \
        --out artifacts/dataset.pt \
        [--clusters artifacts/clusters.tsv]   # entry_id<TAB>cluster_id

The optional clusters file (produced offline by e.g. MMseqs2) keeps homologous
proteins on the same side of the split. Run where PED files + biotite exist.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idpdiff.data.torsions import extract_entry
from idpdiff.data.dataset import make_protein_level_splits


def load_clusters(path):
    if not path:
        return None
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0]] = int(parts[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ped-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="*.pdb")
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.ped_dir, args.glob)))
    if not paths:
        raise SystemExit(f"no files in {args.ped_dir}")

    entries = []
    for p in paths:
        try:
            e = extract_entry(p)
            entries.append(e)
            print(f"parsed {e.entry_id}: {e.n_models} models, L={e.length}")
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {p}: {exc}")

    clusters = load_clusters(args.clusters)
    splits = make_protein_level_splits(
        entries, args.val_frac, args.test_frac, clusters, args.seed
    )
    print(
        f"splits -> train {len(splits['train'])}, "
        f"val {len(splits['val'])}, test {len(splits['test'])} entries"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "sequence": e.sequence,
                    "angles": e.angles,
                    "class_ids": e.class_ids,
                }
                for e in entries
            ],
            "splits": splits,
        },
        args.out,
    )
    print(f"saved dataset -> {args.out}")


if __name__ == "__main__":
    main()
