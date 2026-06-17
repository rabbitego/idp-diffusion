"""Generate a small synthetic dataset for smoke-testing the training pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from idpdiff.data.torsions import _residue_class_ids

rng = np.random.default_rng(42)
amino = list("ACDEFGHIKLMNPQRSTVWY")
entries = []
for i in range(20):
    L = int(rng.integers(15, 40))
    seq = "".join(rng.choice(amino) for _ in range(L))
    n_models = int(rng.integers(5, 15))
    angles = rng.uniform(-np.pi, np.pi, (n_models, L, 2)).astype(np.float64)
    angles[:, 0, 0] = np.nan
    angles[:, -1, 1] = np.nan
    class_ids = _residue_class_ids(seq)
    entries.append({
        "entry_id": f"PED{i:04d}",
        "sequence": seq,
        "angles": angles,
        "class_ids": class_ids,
    })

ids = [e["entry_id"] for e in entries]
splits = {"train": ids[:14], "val": ids[14:17], "test": ids[17:]}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts", "dataset.pt")
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({"entries": entries, "splits": splits}, out)
print(f"Saved: {len(entries)} proteins | train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
