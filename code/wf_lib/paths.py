"""Where things are. Nothing in this release hard-codes a machine path: the asset root comes from `--assets`, the environment
variable WF_ASSETS, or the file wf_config.json in the release folder (see wf_config.example.json).

Release folder (this repository folder):
  code/          the code the wf commands import (tools/, rl/, grasp_runtime/, protocols/, wf_lib/, wf.py)
  reference/     test vectors with expected results (small files, in git)
  asset_manifests/<part>.sha256   the sha256 of every file of each asset part

Asset root (received separately, see ASSETS.md), the layout below $WF_ASSETS:
  evaluator/  assets/                         part `core`   (CPU rollouts)
  mano/                                       part `mano`
  hand_models/  retargeter/                   part `hands`  (Sharpa / Wuji hand protocols)
  ckpt/hug1  ckpt/hug2  ckpt/dumbbell50k      the HUG checkpoints
  hf/                                         the DINOv2 weights in Hugging Face cache layout (HF_HOME)
  hug_source/                                 git checkout of the public HUG repository at the pinned commit (or the folder named by
                                              $WF_HUG_SOURCE, for a read-only asset root)
  inputs/                                     stored input pictures (decode and DSRL read-out)
  prep/<name>/                                DSRL training data folders
  batches/<batch>/                            our production jobs files and stored results (+ INDEX.json)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[2]
CODE = RELEASE / "code"
REFERENCE = RELEASE / "reference"
MANIFESTS = RELEASE / "asset_manifests"
CONFIG_FILE = RELEASE / "wf_config.json"

HUG_SOURCE_URL = "https://github.com/KevinyWu/hug.git"
HUG_SOURCE_COMMIT = "8d1c52d4c24bfae5a369e32e3f134f5601a02630"
DINOV2_REPO = "facebook/dinov2-with-registers-base"
DINOV2_REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"

# decode priors: name -> (checkpoint path below the asset root, expected sha256, Euler steps), as in code/protocols/mano-1111-p1.json
PRIORS = {
    "hug2": ("ckpt/hug2/hug_100k.pt", "3c714b76a61ba3cdb93bec198c98fa9ecb4807c1fa20cce08870d0aaf092180d", 10),
    "hug1": ("ckpt/hug1/hug_full.safetensors", "515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae", 50),
    "dumbbell50k": ("ckpt/dumbbell50k/hug_dumbbell_50k.pt", "cacad83b6dc40b1b20673863e174f95e39fb58e015e1310af500ef911e97ef5b", 10),
}


def config() -> dict:
    if CONFIG_FILE.is_file():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def assets_root(cli=None) -> Path:
    """The asset root: --assets, then $WF_ASSETS, then wf_config.json {"assets": ...}."""
    value = cli or os.environ.get("WF_ASSETS") or config().get("assets")
    if not value:
        raise SystemExit("wf: no asset root. Pass --assets PATH, set WF_ASSETS, or write wf_config.json (see wf_config.example.json).")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"wf: the asset root {root} is not a directory")
    return root


class Layout:
    """The paths below one asset root."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.evaluator = self.root / "evaluator"
        self.mano = self.root / "mano"
        self.objects = self.root / "assets"                       # a job's sim_assets_dir is relative to this folder
        self.hand_models = self.root / "hand_models"
        self.retargeter = self.root / "retargeter"
        self.ckpt = self.root / "ckpt"
        self.hf = self.root / "hf"
        self.hug_source = Path(os.environ["WF_HUG_SOURCE"]).expanduser().resolve() if os.environ.get("WF_HUG_SOURCE") else self.root / "hug_source"
        self.inputs = self.root / "inputs"
        self.input_store = self.root / "inputs/input_store"
        self.prep = self.root / "prep"

    def checkpoint(self, prior: str) -> tuple:
        rel, sha, steps = PRIORS[prior]
        return self.root / rel, sha, steps
