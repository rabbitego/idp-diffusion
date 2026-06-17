"""
Sequence conditioning via ESM embeddings, with on-disk caching.

We use per-residue representations from a pretrained ESM model (ESM-2 650M by
default; ESM-1b is supported by changing the model name) as conditioning for the
denoiser. ESM-2 is a strict, drop-in upgrade over ESM-1b and is recommended
unless a specific comparison to ESM-1b is wanted.

Two honest caveats are baked into the docstring so they are not forgotten:

* ESM was trained predominantly on structured proteins, so for low-complexity
  and disordered sequences the embeddings carry less structural signal than for
  globular domains. They still provide useful per-residue amino-acid identity
  and local sequence context, which is what the local torsion task mostly needs,
  so this is a limitation to be aware of, not a reason to avoid them.
* Embeddings are expensive, so we cache them per sequence (keyed by a hash of
  the sequence) and reuse across conformers of the same PED entry -- every model
  in an ensemble shares one sequence and therefore one embedding.

This module degrades gracefully: if ``fair-esm`` / ``torch.hub`` weights are not
available (e.g. in an offline environment), :class:`ESMEmbedder` raises a clear
message, and a deterministic :class:`MockEmbedder` is provided so the rest of
the pipeline (and the test suite) can run without network access.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

# ESM-2 650M has width 1280, matching ModelConfig.seq_embed_dim by default.
DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D"
ESM_WIDTHS = {
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t30_150M_UR50D": 640,
    "esm1b_t33_650M_UR50S": 1280,
}


def sequence_hash(seq: str, model_name: str) -> str:
    return hashlib.sha1(f"{model_name}:{seq}".encode()).hexdigest()[:16]


class ESMEmbedder:
    """Wraps a pretrained ESM model to produce per-residue embeddings.

    Lazily loads weights on first use. Results are cached to ``cache_dir`` as
    ``.npy`` files keyed by sequence hash.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ESM_MODEL,
        cache_dir: str = ".esm_cache",
        device: str | torch.device = "cpu",
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = torch.device(device)
        os.makedirs(cache_dir, exist_ok=True)
        self._model = None
        self._batch_converter = None
        self._repr_layer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import esm  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ESMEmbedder needs fair-esm (`pip install fair-esm`) and network "
                "access to download weights. Use MockEmbedder for offline dev."
            ) from exc
        model, alphabet = torch.hub.load("facebookresearch/esm:main", self.model_name)
        self._model = model.eval().to(self.device)
        self._batch_converter = alphabet.get_batch_converter()
        # final transformer layer index
        self._repr_layer = int(self.model_name.split("_")[1][1:])

    @property
    def dim(self) -> int:
        return ESM_WIDTHS.get(self.model_name, 1280)

    def embed(self, seq: str) -> np.ndarray:
        """Return (L, dim) per-residue embedding for one sequence, cached."""
        key = sequence_hash(seq, self.model_name)
        cache_path = os.path.join(self.cache_dir, f"{key}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)

        self._ensure_loaded()
        _, _, tokens = self._batch_converter([("seq", seq)])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            out = self._model(tokens, repr_layers=[self._repr_layer])
        rep = out["representations"][self._repr_layer][0]
        # strip BOS/EOS -> (L, dim)
        rep = rep[1 : len(seq) + 1].cpu().numpy().astype(np.float32)
        np.save(cache_path, rep)
        return rep


class MockEmbedder:
    """Deterministic offline stand-in for ESM, for development and tests.

    Produces a reproducible per-residue embedding from a hash of each residue's
    identity and position. It carries no biological signal -- it exists purely so
    the pipeline runs end-to-end without downloading weights. Swap in
    :class:`ESMEmbedder` for real experiments.
    """

    def __init__(self, dim: int = 1280, seed: int = 0):
        self._dim = dim
        self.seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, seq: str) -> np.ndarray:
        out = np.empty((len(seq), self._dim), dtype=np.float32)
        for i, aa in enumerate(seq):
            h = int(hashlib.sha1(f"{self.seed}:{aa}:{i % 64}".encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2 ** 32))
            out[i] = rng.standard_normal(self._dim).astype(np.float32)
        return out
