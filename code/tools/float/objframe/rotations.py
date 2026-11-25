# Origin: the model authors' object-frame HUG2 code, commit 44ad0250779db5366c938d2c71f0a5467e007770, file src/utils/transform_utils.py; MIT licence (the authors' notice: LICENSES/hug2obj-source-LICENSE.txt).
# Changes in this copy: none (the file is the original, byte for byte, below this header).
# Provenance (sha256 of the original and of this file): PROVENANCE.json next to this file.
"""Transformation utilities."""

import torch
from roma import special_gramschmidt


def six_d_to_rotation_matrix(six_d: torch.Tensor) -> torch.Tensor:
    """Convert 6D representation to 3x3 rotation matrix using Gram-Schmidt."""
    six_d = six_d.reshape(*six_d.shape[:-1], 3, 2)
    return special_gramschmidt(six_d)
