# Third-party code and code derived from it in this folder

This project's own code and records are released under the **MIT licence** ([LICENSE.txt](LICENSE.txt), Copyright (c) 2026 the wuji-floating authors). Files that are not only ours keep the licence of their source:

| Path | Origin | Licence |
|---|---|---|
| `code/tools/float/collab_native.py` | the floating-MANO adapter around the HUG-Bench evaluator, copied byte for byte from `campaigns/mano1111/native.py` of the mano-1111 task branch (commit 02aeadcc) of the project's shared development repository, which is not public; it imports the evaluator's modules unchanged; its sha256 is part of every protocol digest | MIT: this project's code (LICENSE.txt); the evaluator it imports is MIT ([LICENSES/HUG-Bench-evaluator-MIT.txt](LICENSES/HUG-Bench-evaluator-MIT.txt), the evaluator's own `LICENSE`, also shipped as `evaluator/LICENSE` in the asset part `core`) |
| `code/tools/float_wuji/wuji_eval.py`, `code/tools/float_hands/hand_eval.py` | this project's evaluators that put a robot hand into the HUG-Bench evaluator harness (the frozen October evaluator and its generalisation); they import the evaluator's modules unchanged | MIT: this project's code on the MIT-licensed evaluator harness (both notices apply: LICENSE.txt and LICENSES/HUG-Bench-evaluator-MIT.txt) |
| `code/tools/float/decode.py` | in-process HUG decoding; the body of `prepared_scene` is kept statement for statement equal to `hug_finetune/scene.py::prepared_scene` of the same mano-1111 task branch (the function of its sampling service); it imports the public HUG source (github.com/KevinyWu/hug, MIT) | MIT: this project's code (LICENSE.txt); the HUG source it imports is MIT (`licences/hug-LICENSE.txt` in the asset part `licences`) |
| `code/tools/dsrl_port/` (`dsrl.py`, `sb3_parts.py`, `hug2_adapter.py`) | a line-by-line port of DSRL-NA (github.com/ajwagen/dsrl @8d21b9c; the update code is in its submodule github.com/ajwagen/stable-baselines3-dsrl @10e5d311); `DIFF.md` lists every change | MIT: the submodule's licence, shipped unchanged with its copyright notice as `LICENSE_stable-baselines3-dsrl`; the DSRL repository itself has no top-level licence file |
| `code/rl/sim/multihand/` | this project's multihand retargeter (fits a robot hand to HUG's landmarks); the hand models it loads are in asset part `hands` (Apache-2.0, see ASSETS.md) | MIT (this project's code) |
| everything else under `code/`, `configs/`, `reference/` | this project's code and records | MIT (`LICENSE.txt`) |

The asset tarballs (not in git) carry the licence texts their parts need in the part `licences` (`licences/README.txt` is the index, with the attribution CC-BY-4.0 requires); [ASSETS.md](ASSETS.md) and `ASSET_TERMS.txt` give the licence of every part.

Python packages are installed from their own indexes (requirements/*.txt) and are not redistributed here: MuJoCo is Apache-2.0, chumpy is MIT, and manotorch is GPL
according to its package metadata (it is installed from its public repository only).
