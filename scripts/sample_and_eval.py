#!/usr/bin/env python
"""Sample ensembles from a trained model and evaluate against held-out PED.

Generates, per held-out protein, an ensemble of the same size as the reference,
then computes the full metric suite (per-residue phi-psi JSD, Rg/Ree/asphericity
Wasserstein distances, diversity, clash rate) and writes a results table. Also
runs the statistical-coil baseline on the same proteins for direct comparison.

Usage
-----
    python scripts/sample_and_eval.py \
        --dataset artifacts/dataset.pt \
        --checkpoint runs/with_reg/ema_latest.pt \
        --rama-kde artifacts/rama_kde.pt \
        --out runs/with_reg/eval.json \
        [--mock-esm]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idpdiff.data.torsions import TorsionEntry
from idpdiff.data.embeddings import ESMEmbedder, MockEmbedder, ESM_WIDTHS
from idpdiff.diffusion.wrapped import WrappedDiffusion, DiffusionConfig
from idpdiff.diffusion.losses import RamachandranKDE
from idpdiff.models.denoiser import TorsionDenoiser, ModelConfig
from idpdiff.eval.metrics import summarize, per_residue_jsd
from idpdiff.eval.baselines import StatisticalCoilModel
from idpdiff.validation.reconstruct import reconstruct_backbone, clash_count


def load_split_entries(dataset_path, split):
    blob = torch.load(dataset_path, map_location="cpu")
    ids = set(blob["splits"][split])
    return [
        TorsionEntry(d["entry_id"], d["sequence"], d["angles"], d["class_ids"])
        for d in blob["entries"] if d["entry_id"] in ids
    ]


@torch.no_grad()
def sample_for_entry(model, diffusion, embedder, entry, n_conf, device):
    emb = torch.from_numpy(embedder.embed(entry.sequence)).float().unsqueeze(0)
    emb = emb.repeat(n_conf, 1, 1).to(device)
    L = entry.length
    mask = torch.ones(n_conf, L, dtype=torch.bool, device=device)
    angles = diffusion.sample(model, emb, mask, L)
    return angles.cpu().numpy()


def mean_clash_rate(angles):
    rates = []
    for k in range(angles.shape[0]):
        coords = reconstruct_backbone(angles[k, :, 0], angles[k, :, 1])
        rates.append(clash_count(coords[:, 1, :]))
    return float(np.mean(rates))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--rama-kde", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--mock-esm", action="store_true")
    ap.add_argument("--max-conf", type=int, default=200,
                    help="cap conformers per protein for speed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = TorsionDenoiser(model_cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = WrappedDiffusion(DiffusionConfig(), device=args.device)

    if args.mock_esm:
        embedder = MockEmbedder(dim=model_cfg.seq_embed_dim)
    else:
        embedder = ESMEmbedder(args.esm_model, device=args.device)

    coil = None
    if args.rama_kde:
        coil = StatisticalCoilModel(RamachandranKDE.load(args.rama_kde))

    entries = load_split_entries(args.dataset, args.split)
    print(f"evaluating on {len(entries)} held-out proteins")

    results = {}
    for e in entries:
        n_ref = min(e.n_models, args.max_conf)
        ref = e.angles[:n_ref]

        gen = sample_for_entry(model, diffusion, embedder, e, n_ref, args.device)
        model_metrics = summarize(gen, ref)
        model_metrics["clash_rate"] = mean_clash_rate(gen)
        model_metrics["ref_clash_rate"] = mean_clash_rate(ref)

        entry_result = {"model": model_metrics}

        if coil is not None:
            coil_gen = coil.sample_ensemble(e.sequence, n_ref)
            coil_metrics = summarize(coil_gen, ref)
            coil_metrics["clash_rate"] = mean_clash_rate(coil_gen)
            entry_result["statistical_coil"] = coil_metrics

        results[e.entry_id] = entry_result
        print(f"{e.entry_id}: model JSD={model_metrics['jsd_mean']:.3f} "
              f"Rg-W={model_metrics['rg_wasserstein']:.2f}A "
              f"clash={model_metrics['clash_rate']:.2f}"
              + (f" | coil JSD={entry_result['statistical_coil']['jsd_mean']:.3f}"
                 if coil else ""))

    # Aggregate
    def agg(metric, sub="model"):
        vals = [r[sub][metric] for r in results.values() if sub in r]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_proteins": len(entries),
        "model": {m: agg(m, "model") for m in
                  ["jsd_mean", "rg_wasserstein", "ree_wasserstein",
                   "asphericity_wasserstein", "gen_diversity", "clash_rate"]},
    }
    if coil is not None:
        summary["statistical_coil"] = {m: agg(m, "statistical_coil") for m in
                                       ["jsd_mean", "rg_wasserstein", "ree_wasserstein",
                                        "asphericity_wasserstein", "gen_diversity", "clash_rate"]}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "per_protein": results}, fh, indent=2)
    print(f"\nsaved -> {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
