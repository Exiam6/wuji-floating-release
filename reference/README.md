# Reference vectors (small files, kept in git)

The verification suite (`python code/wf.py verify`) compares what your machine computes with the records stored here. Nothing in this folder is an asset in
the sense of ASSETS.md: the jobs files hold decoded grasps and camera poses, the expected files hold our records. The object folders the jobs name
(`sim_assets_dir`) come from the asset bundle's `core` part.

## `cpu/` (levels L1 and L1h; bit for bit)

| File | Rows | What |
|---|---:|---|
| `jobs_ref200.npz` | 200 | HUG2 grasps (set a, sample 0) for the first 10 configurations of each of 20 HUG-Bench objects, from one of our experiments (EX285 seed 1) |
| `expected_ref200_mano-1111-p1.npz` | 200 | our labels (112 of 200 succeed) and the evaluator's other record fields. The labels were rolled on host A, re-rolled on host B and host C, and again on host A through this release: all equal. The other fields come from host B's re-roll; host C's re-roll and this release on host A equal them in every field |
| `expected_ref200_mano-1111-p2.npz` | 200 | the same grasps under p2 (92 of 200 succeed); rolled by this release on one host (not an independent record); equal across Python 3.10 and 3.12. The p2 rule itself is anchored by `expected_w3_mano-1111-p2.npz` |
| `jobs_w3_220.npz` | 220 | one dumbbell census sample per view (the prior alternates hug2 / hug1 / dumbbell50k) |
| `expected_w3_<protocol>.npz` | 220 | our records of those grasps under `mano-1111-p1`, `-p2`, `-p2s-dzm1`, `-p2-dzm1`, `-p2-dzm2`, `-p2-dzp1`, rolled by our repository code on one host |
| `jobs_bench2000.npz`, `expected_bench2000_mano-1111-p1.npz` | 2,000 | the throughput set (samples 1 to 10 of the same configurations): `wf.py verify --levels L1 --set bench2000 --out DIR` runs it, and the speed of your CPU can be read from the run |
| `hands/jobs_{ref200,pilot64}_hands.npz`, `hands/expected_*_{sharpa,wuji1hr}-float-p{1,2}.npz` | 200 / 64 | the robot-hand protocols (the grasps do not depend on the hand) |
| `wuji1/jobs_ref200_wuji1.npz`, `wuji1/expected_ref200_wuji1-float-p1.npz` | 200 | Wuji Hand 1 in the frozen October evaluator |
| `INDEX.json` | | for each file: where it came from and its sha256 |

The HUG-Bench files (`jobs_ref200.npz` with `expected_ref200_mano-1111-p1.npz` and `expected_ref200_mano-1111-p2.npz`, `jobs_bench2000.npz` with `expected_bench2000_mano-1111-p1.npz`) are also the
known-answer pairs (canary bank, `wf_lib/canary.py`, README section 9.2) that jobs on HUG-Bench objects draw their canaries from, next to the dumbbell census pairs of the asset part `batches`.

The `source` and `run_info` text fields inside the expected files were rewritten to remove our machine names and paths; every array is bit-equal to the original
(the builder asserts it) and `wf.py rollout compare` ignores both fields.

## `decode/` (level L2; within a tolerance)

Six pictures of the dumbbell (one per view group of the census; `input_set.json`, `input_store/`) and `expected_<prior>.npz`: what HUG2, the dumbbell 50k prior and HUG1
decoded from them on 2026-11-18 on an NVIDIA GeForce RTX 3080 (torch 2.9.1+cu128), the first 32 codebook noise rows each (192 grasps per prior). `PROVENANCE.json`
lists the checkpoints' sha256 and the decoder settings. GPU kernels are not bit-identical across GPU models, so the check allows 1e-3 (the project's census re-decode
tolerance; the measured difference between an RTX 3080 and an RTX A6000 is below 2e-5, see docs/SMOKE_TEST.md).

## `dsrl/` (level L3)

`mini_prep/` is a small training set cut from the project's HUG1 dumbbell prep (8 training views, 256 labelled executions, 512 pool draws per view, the states of 11 views) and
`mini_config.json` the authors' settings with 20 calls. `expected_mini_train.json` is the record of that training on one RTX A6000 (the loss and Q-value summary after
each call and the end-of-training diagnostics). The check compares them within the tolerance printed by `wf.py dsrl check`.

`dsrl/mini_sweep/` (revision r7a) is the reference vector of the sweep path: 3 of the shipped training objects (55 views, 1,760 labelled executions, copied bit for bit from the asset part `prep_ex338_a_p2`) in the layout of a census shard
(`prep/`: states, actions, a noise pool stored once with 64 rows per view and tiled by index to the registered 4,096, manifest with every file's sha256), `train_ids.txt`, `forbidden_ids.json` (the 20 held-out test objects) and `config.json` (the registered recipe, 25 calls).
`wf.py dsrl check` trains it through the whole list-file path and compares it with `expected_mini_sweep.json` (`dsrl/mini_sweep/expected_mini_sweep.json`, the record of that training on one RTX A6000; the check reports T6 as skipped if the file is missing), and
trains the same data from a tiled copy of the pool on your machine: the two must be bit-identical.

## `census/` (an input set for batches)

`dumbbell_census220_readout_sets.json` is the dumbbell census input set of the asset part `inputs` (220 views, the same order, every field of every unit unchanged) with a `readout_set` per unit: 0 = the 56 training
views, 1 = the 164 held-out views of the EX330 / EX331 tables. `dsrl readout --sets 1` reads out the held-out views only, `decode run --units` takes any subset. The 164 held-out ids come from the `heldout` flags of our
census metadata (the sha256 of the sorted ids joined by newlines is in the file's `note`). A batch group names it as `input_set: "reference/census/dumbbell_census220_readout_sets.json"` (the runner looks in the release first, then in the asset root).
