# Origin: the model authors' object-frame HUG2 code, commit 44ad0250779db5366c938d2c71f0a5467e007770, file src/utils/data_keys.py; MIT licence (the authors' notice: LICENSES/hug2obj-source-LICENSE.txt).
# Changes in this copy: '(whole file)' -> "rewritten: the package's data files and mano_models_folder(); PROJECT_ROOT kept; OUTPUTS_FOLDER and MANO_RIGHT_MESH_FACES_FILE dropped (read by none of the ten modules)".
# Provenance (sha256 of the original and of this file): PROVENANCE.json next to this file.
"""Core file names of the object-frame model: the package's own data files and the place of the user's MANO model."""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR                                    # a relative norm-stats path of a config is read below the package

# the two small data files of the model (kept under data/)
MANO_RIGHT_SHAPE_FILE = PACKAGE_DIR / "data" / "mano_rhand_shape.npy"
NORM_STATS_FILE = PACKAGE_DIR / "data" / "norm_stats.json"


def mano_models_folder() -> Path:
    """The folder that holds models/MANO_RIGHT.pkl (the user's own MANO download): $WF_MANO_ROOT, else $WF_ASSETS/mano."""
    root = os.environ.get("WF_MANO_ROOT")
    if not root:
        assets = os.environ.get("WF_ASSETS")
        if not assets:
            raise RuntimeError("the MANO model folder is not set: set WF_MANO_ROOT (the folder that holds models/MANO_RIGHT.pkl) or WF_ASSETS")
        root = str(Path(assets) / "mano")
    if not (Path(root) / "models" / "MANO_RIGHT.pkl").is_file():
        raise RuntimeError(f"the MANO model is missing: {Path(root) / 'models' / 'MANO_RIGHT.pkl'} (README.md section 3.0)")
    return Path(root)
