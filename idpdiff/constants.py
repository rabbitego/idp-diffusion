"""Shared constants with no heavy dependencies.

Kept torch-free so the data-parsing and geometry layers (which are pure numpy)
can import residue-class definitions without pulling in torch. The diffusion
losses re-export these names for backward compatibility.
"""

from __future__ import annotations

# Residue-type classes for residue-specific Ramachandran densities.
RAMA_CLASSES = ("general", "glycine", "proline", "pre_proline")
RAMA_CLASS_TO_ID = {name: i for i, name in enumerate(RAMA_CLASSES)}
