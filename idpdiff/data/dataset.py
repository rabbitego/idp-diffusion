"""
Dataset, collation, and protein-level splitting.

The single most important methodological point in this file: **splits are made
at the level of whole proteins (PED entries), never at the level of individual
conformers.** A random per-conformer split would place different members of the
same protein's ensemble in both train and test, leaking the answer and inflating
every metric. The point of sequence conditioning is generalisation to unseen
sequences, so the test set must contain proteins the model never saw.

``make_protein_level_splits`` goes one step further and supports grouping by a
caller-supplied cluster id (e.g. from MMseqs2 at some sequence-identity
threshold), so sequence-similar proteins are kept together on the same side of
the split. The clustering itself is done offline by ``scripts/cluster_*`` and
passed in; this module just respects the grouping.

A training example is one conformer: (sequence embedding, angles for that
conformer, residue-class ids, mask). Because all conformers of an entry share a
sequence, the embedding is computed once per entry and broadcast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .torsions import TorsionEntry


@dataclass
class Conformer:
    entry_id: str
    sequence: str
    angles: np.ndarray  # (L, 2) radians, NaN at undefined termini
    class_ids: np.ndarray  # (L,)


def entries_to_conformers(entries: list[TorsionEntry]) -> list[Conformer]:
    """Flatten entries into per-conformer training examples."""
    conformers: list[Conformer] = []
    for e in entries:
        for m in range(e.n_models):
            conformers.append(Conformer(e.entry_id, e.sequence, e.angles[m], e.class_ids))
    return conformers


def make_protein_level_splits(
    entries: list[TorsionEntry],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    cluster_of: dict[str, int] | None = None,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Split PED entry ids into train/val/test by protein (or cluster).

    Returns a dict with keys 'train', 'val', 'test' mapping to lists of
    entry_ids. If ``cluster_of`` is given (entry_id -> cluster id), whole
    clusters are assigned together so homologous proteins do not straddle the
    split boundary.
    """
    rng = np.random.default_rng(seed)
    ids = [e.entry_id for e in entries]

    if cluster_of is None:
        groups = {eid: [eid] for eid in ids}
    else:
        groups = {}
        for eid in ids:
            groups.setdefault(cluster_of.get(eid, eid), []).append(eid)

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    n = len(group_keys)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n)))

    test_keys = group_keys[:n_test]
    val_keys = group_keys[n_test : n_test + n_val]
    train_keys = group_keys[n_test + n_val :]

    def expand(keys):
        out = []
        for k in keys:
            out.extend(groups[k])
        return out

    return {"train": expand(train_keys), "val": expand(val_keys), "test": expand(test_keys)}


class TorsionDataset(Dataset):
    """Per-conformer dataset that yields tensors ready for the collator.

    The embedder is queried lazily and cached by the embedder itself, so the
    dataset stays light. ``max_length`` truncates pathologically long chains;
    shorter chains are padded in the collator.
    """

    def __init__(self, conformers: list[Conformer], embedder, max_length: int = 256):
        self.conformers = conformers
        self.embedder = embedder
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.conformers)

    def __getitem__(self, idx: int):
        c = self.conformers[idx]
        L = min(len(c.sequence), self.max_length)
        seq = c.sequence[:L]
        emb = self.embedder.embed(seq)[:L]  # (L, D)
        angles = c.angles[:L]  # (L, 2)
        class_ids = c.class_ids[:L]  # (L,)

        # Defined-angle mask: exclude termini / missing atoms (NaN angles).
        defined = ~np.isnan(angles).any(axis=-1)  # (L,)
        # Replace NaN with 0 so tensors are finite; mask zeroes their loss.
        angles = np.nan_to_num(angles, nan=0.0)

        return {
            "embedding": torch.from_numpy(emb).float(),
            "angles": torch.from_numpy(angles).float(),
            "class_ids": torch.from_numpy(class_ids).long(),
            "defined": torch.from_numpy(defined),
            "length": L,
        }


def collate(batch: list[dict]) -> dict:
    """Right-pad a batch to the max length and build the padding/defined masks.

    ``mask`` is True only where a position is both a real (non-pad) residue and
    has a defined angle, so it is the single mask the loss and metrics consume.
    """
    B = len(batch)
    Lmax = max(item["length"] for item in batch)
    D = batch[0]["embedding"].shape[-1]

    emb = torch.zeros(B, Lmax, D)
    angles = torch.zeros(B, Lmax, 2)
    class_ids = torch.zeros(B, Lmax, dtype=torch.long)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)

    for i, item in enumerate(batch):
        L = item["length"]
        emb[i, :L] = item["embedding"]
        angles[i, :L] = item["angles"]
        class_ids[i, :L] = item["class_ids"]
        mask[i, :L] = item["defined"]

    return {
        "embedding": emb,
        "angles": angles,
        "class_ids": class_ids,
        "mask": mask,
    }
