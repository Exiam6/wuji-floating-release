> Note (revision r3): this record belongs to the first release (revision r1). The section numbers it quotes (README sections 3 to 10) are those of the r1 README; in r3 the unit of work is the issued batch (README section 5), job types are numbered GPU first, and returns are section 9.

# Smoke test of the release on a clean machine (2026-11-20)

Two runs on our reference host A. **Run 2 belongs to this release:** the assets exactly as the recipients get them (tarballs, without the MANO model), the README's own commands, the MANO file placed by hand as README section 3.0 says. **Run 1** (the end of this file) was made on the first release commit, with the asset bundle copied as a folder that still held the MANO model; where run 2 repeats a check, run 2 stands.

## Run 2: the delivery form (tarballs, MANO placed by the recipient)

**What was tested:** the release folder (`SHA256SUMS` digest `8674fd78cb932959f40f1e1aaec2106f6ce599e7c63b2a0622c53fd60b5e8313`, 135 files) and both delivery sets as they come out of our packaging step: the rollout-only set (4 tarballs) and the full set (13 tarballs), rsynced to host A, unpacked into empty asset roots with the README's commands, the MANO file copied in by hand (a stand-in for the file downloaded from the Max Planck Institute: the same file, sha256 checked). `nice -n 19`, at most two CPU workers, GPU 0 only. The main run (every table below except the CPython 3.12 one and the public-pieces table) was made on that digest; the CPython 3.12 run and the public-pieces check were made afterwards on `SHA256SUMS` digest `5a565ecf2bf20df8be8cb0e3294d9c142b7268ca411ed77b1b80fd66cb0b4b00`, which differs from it by one line of code: the release check ignores the `.git` folder, so that a clone of the repository verifies in place (`code/wf_lib/manifest.py`). The committed folder differs from those states only by README text (sections 3.0, 4, 8, 9 and 10) and `docs/SMOKE_TEST.md`, and by `SHA256SUMS`.

### The tarballs and the MANO file (README sections 3.0 and 3.1, commands as written there)

| Step | Result |
|---|---|
| Rollout-only set, `sha256sum -c SHA256SUMS` in the downloaded folder | 4 of 4 tarballs `OK` |
| Unpack with `for t in *.tar; do tar -xf "$t" -C $ASSETS; done` | 1,846 files, 309 MB, 0.3 s (the files had just been copied, so they came from the page cache); no MANO file in it |
| `sha256sum -c --quiet --ignore-missing asset_manifests/SHA256SUMS.all` in the asset root | pass |
| MANO file missing: `wf.py assets verify --parts mano` (with the system Python 3.12 started with `-S -I`, i.e. no packages at all), `assets verify`, `verify --levels L0,L1`, `rollout run`, `decode run`, `dsrl readout` | every one stops with a message that names the missing path, the registration URL, the archive, the file inside it and the expected sha256 (exit 1); `verify` stops once, before any level runs |
| A wrong file at the MANO path | `assets verify --parts mano` prints the file's sha256 and the expected one and says where to get the right file (exit 1) |
| The file placed as README 3.0 step 3 says | `sha256sum` prints `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767`; `assets verify --parts mano`: pass |
| Full set, `sha256sum -c SHA256SUMS` | 13 of 13 tarballs `OK`; unpacked in 5.1 s (page cache): 4,580 files, 5.8 GiB, no MANO file |
| `sha256sum -c --quiet asset_manifests/SHA256SUMS.all` in the full asset root (no `--ignore-missing`) | pass, 4,580 files |
| `assets fetch-hug-source` before and after the MANO file is in place | before: `pass: false`, and the message that says to place the file and run it again; after: `pass: true`, MANO linked into the HUG source |
| `assets verify` (default: every part present, and mano) on the full asset root | pass for all 14 parts, the bundle note says that no `SHA256SUMS` is at the asset root (unpacked tarballs do not carry it) |


### CPU, rollout-only asset root (CPython 3.10.20, 2 workers)

`wf.py verify --levels L0,L1,L1h,L1b --workers 2`

Result: **PASS**, 41 checks in 1211 s. Release `SHA256SUMS` digest `8674fd78cb932959f40f1e1aaec2106f6ce599e7c63b2a0622c53fd60b5e8313`. CPython 3.10.20, mujoco 3.10.0, numpy 2.2.6, torch 2.9.1+cpu.

| Check | Status | Labels equal | Every field equal | Rollouts per second | Note |
|---|---|---|---|---|---|
| L0 release folder | pass_bitwise |  |  |  |  |
| L0 asset part batches | pass_bitwise |  |  |  |  |
| L0 asset part core | pass_bitwise |  |  |  |  |
| L0 asset part hands | pass_bitwise |  |  |  |  |
| L0 asset part mano | pass_bitwise |  |  |  |  |
| l1_ref200_mano-1111-p1 | pass_bitwise | 200 of 200 | yes | 5.98 |  |
| l1_ref200_mano-1111-p2 | pass_bitwise | 200 of 200 | yes | 6.069 |  |
| l1_w3_mano-1111-p1 | pass_bitwise | 220 of 220 | yes | 4.835 |  |
| l1_w3_mano-1111-p2 | pass_bitwise | 220 of 220 | yes | 5.185 |  |
| l1_w3_mano-1111-p2s-dzm1 | pass_bitwise | 220 of 220 | yes | 4.666 |  |
| l1_w3_mano-1111-p2-dzm1 | pass_bitwise | 220 of 220 | yes | 4.581 |  |
| l1_w3_mano-1111-p2-dzm2 | pass_bitwise | 220 of 220 | yes | 4.077 |  |
| l1_w3_mano-1111-p2-dzp1 | pass_bitwise | 220 of 220 | yes | 6.335 |  |
| l1b_ex330_ex331_census220_mano-1111-p1 | pass_bitwise | 100 of 100 | yes | 3.764 |  |
| l1b_ex330_ex331_census220_mano-1111-p2 | pass_bitwise | 100 of 100 | yes | 3.783 |  |
| l1b_ex330_ex331_census220_mano-1111-p2s-dzm1 | pass_bitwise | 100 of 100 | yes | 3.655 |  |
| l1b_ex330_ex331_census220_mano-1111-p2-dzm1 | pass_bitwise | 100 of 100 | yes | 3.563 |  |
| l1b_ex330_ex331_census220_mano-1111-p2-dzm2 | pass_bitwise | 100 of 100 | yes | 3.299 |  |
| l1b_ex330_ex331_census220_mano-1111-p2-dzp1 | pass_bitwise | 100 of 100 | yes | 4.395 |  |
| l1b_ex305_wuji_dumbbell_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 4.817 |  |
| l1b_ex307_wuji_dumbbell_new_views_part1_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 4.923 |  |
| l1b_ex307_wuji_dumbbell_new_views_part2_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 4.826 |  |
| l1b_ex308_wuji_steering_readout_wuji1-float-p1 | pass_near | 100 of 100 | no | 4.698 | 1 job differs, largest difference 8.6e-05 |
| l1b_ex318_wuji_multiobject_part1_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 7.054 |  |
| l1b_ex318_wuji_multiobject_part2_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 7.761 |  |
| l1b_ex318_wuji_multiobject_part3_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 8.051 |  |
| l1b_ex318_wuji_multiobject_part4_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 7.584 |  |
| l1b_ex327_hands_pop9216_sharpa-float-p1 | pass_bitwise | 100 of 100 | yes | 3.472 |  |
| l1b_ex327_hands_pop9216_sharpa-float-p2 | pass_bitwise | 100 of 100 | yes | 3.81 |  |
| l1b_ex327_hands_pop9216_wuji1hr-float-p1 | pass_bitwise | 100 of 100 | yes | 2.933 |  |
| l1b_ex327_hands_pop9216_wuji1hr-float-p2 | pass_bitwise | 100 of 100 | yes | 2.97 |  |
| l1b_ex327_hands_pop9216_mano-1111-p2 | pass_bitwise | 100 of 100 | yes | 6.569 |  |
| l1h_ref200_sharpa-float-p1 | pass_bitwise | 200 of 200 | yes | 3.418 |  |
| l1h_pilot64_sharpa-float-p1 | pass_bitwise | 64 of 64 | yes | 1.23 |  |
| l1h_ref200_sharpa-float-p2 | pass_bitwise | 200 of 200 | yes | 3.795 |  |
| l1h_pilot64_sharpa-float-p2 | pass_bitwise | 64 of 64 | yes | 1.18 |  |
| l1h_ref200_wuji1hr-float-p1 | pass_bitwise | 200 of 200 | yes | 3.969 |  |
| l1h_pilot64_wuji1hr-float-p1 | pass_bitwise | 64 of 64 | yes | 1.85 |  |
| l1h_ref200_wuji1hr-float-p2 | pass_bitwise | 200 of 200 | yes | 4.238 |  |
| l1h_pilot64_wuji1hr-float-p2 | pass_bitwise | 64 of 64 | yes | 1.696 |  |
| l1h_ref200_wuji1-float-p1 | pass_bitwise | 200 of 200 | yes | 8.934 |  |

### L0 over every part of the full set

`wf.py verify --levels L0`

Result: **PASS**, 15 checks in 4 s. Release `SHA256SUMS` digest `8674fd78cb932959f40f1e1aaec2106f6ce599e7c63b2a0622c53fd60b5e8313`. CPython 3.10.20, mujoco 3.10.0, numpy 2.2.6, torch 2.9.1+cpu.

| Check | Status | Labels equal | Every field equal | Rollouts per second | Note |
|---|---|---|---|---|---|
| L0 release folder | pass_bitwise |  |  |  |  |
| L0 asset part batches | pass_bitwise |  |  |  |  |
| L0 asset part core | pass_bitwise |  |  |  |  |
| L0 asset part dinov2 | pass_bitwise |  |  |  |  |
| L0 asset part dumbbell50k | pass_bitwise |  |  |  |  |
| L0 asset part hands | pass_bitwise |  |  |  |  |
| L0 asset part hug1 | pass_bitwise |  |  |  |  |
| L0 asset part hug2 | pass_bitwise |  |  |  |  |
| L0 asset part inputs | pass_bitwise |  |  |  |  |
| L0 asset part licences | pass_bitwise |  |  |  |  |
| L0 asset part mano | pass_bitwise |  |  |  |  |
| L0 asset part prep_ex299_hug1 | pass_bitwise |  |  |  |  |
| L0 asset part prep_ex315_hug2 | pass_bitwise |  |  |  |  |
| L0 asset part prep_ex324_a | pass_bitwise |  |  |  |  |
| L0 asset part prep_ex324_b | pass_bitwise |  |  |  |  |

### GPU 0: decode and DSRL-NA

`CUDA_VISIBLE_DEVICES=0 wf.py verify --levels L0,L2,L3`

Result: **PASS**, 12 checks in 35 s. Release `SHA256SUMS` digest `8674fd78cb932959f40f1e1aaec2106f6ce599e7c63b2a0622c53fd60b5e8313`. CPython 3.10.20, mujoco 3.10.0, numpy 2.2.6, torch 2.9.1+cu128.

| Check | Status | Labels equal | Every field equal | Rollouts per second | Note |
|---|---|---|---|---|---|
| L0 release folder | pass_bitwise |  |  |  |  |
| L0 asset part dinov2 | pass_bitwise |  |  |  |  |
| L0 asset part dumbbell50k | pass_bitwise |  |  |  |  |
| L0 asset part hug1 | pass_bitwise |  |  |  |  |
| L0 asset part hug2 | pass_bitwise |  |  |  |  |
| L0 asset part mano | pass_bitwise |  |  |  |  |
| L2 decode hug2 | pass_tolerance |  |  |  | max difference of the 99-D output 3.0e-06, of the landmarks 5.5e-07 m (NVIDIA RTX A6000) |
| L2 decode hug1 | pass_tolerance |  |  |  | max difference of the 99-D output 3.0e-06, of the landmarks 1.3e-07 m (NVIDIA RTX A6000) |
| L2 decode dumbbell50k | pass_tolerance |  |  |  | max difference of the 99-D output 8.2e-06, of the landmarks 6.0e-07 m (NVIDIA RTX A6000) |
| L3 T1_terminal_backup_skip | pass_bitwise |  |  |  | terminal-backup skip is exact: every parameter bit-identical to the authors' full backup after 3 train() calls |
| L3 T3_noise_bounds | pass_bitwise |  |  |  | the deterministic noise lies in [-bound, bound]; the noise critic's input is noise / bound |
| L3 T4_mini_training | pass_tolerance |  |  |  | 20 calls on the reference mini prep; worst relative difference to the reference 0 (tolerance 0.02) |

### CPU, rollout-only asset root (CPython 3.12.3, 2 workers)

`wf.py verify --levels L0,L1 --workers 2`

Result: **PASS**, 11 checks in 341 s. Release `SHA256SUMS` digest `5a565ecf2bf20df8be8cb0e3294d9c142b7268ca411ed77b1b80fd66cb0b4b00`. CPython 3.12.3, mujoco 3.10.0, numpy 2.2.6, torch 2.9.1+cpu.

| Check | Status | Labels equal | Every field equal | Rollouts per second | Note |
|---|---|---|---|---|---|
| L0 release folder | pass_bitwise |  |  |  |  |
| L0 asset part core | pass_bitwise |  |  |  |  |
| L0 asset part mano | pass_bitwise |  |  |  |  |
| l1_ref200_mano-1111-p1 | pass_bitwise | 200 of 200 | yes | 5.934 |  |
| l1_ref200_mano-1111-p2 | pass_bitwise | 200 of 200 | yes | 6.151 |  |
| l1_w3_mano-1111-p1 | pass_bitwise | 220 of 220 | yes | 4.888 |  |
| l1_w3_mano-1111-p2 | pass_bitwise | 220 of 220 | yes | 5.252 |  |
| l1_w3_mano-1111-p2s-dzm1 | pass_bitwise | 220 of 220 | yes | 4.692 |  |
| l1_w3_mano-1111-p2-dzm1 | pass_bitwise | 220 of 220 | yes | 4.634 |  |
| l1_w3_mano-1111-p2-dzm2 | pass_bitwise | 220 of 220 | yes | 4.028 |  |
| l1_w3_mano-1111-p2-dzp1 | pass_bitwise | 220 of 220 | yes | 6.334 |  |

### The four released DSRL prep folders load and train (full asset root, GPU 0, `dsrl train --calls 3`)

| Config | Wall time | Peak card memory | Peak host memory |
|---|---|---|---|
| `configs/dsrl_ex299_hug1.json` | 5.9 s | 1.0 GiB | 1.6 GB |
| `configs/dsrl_ex315_hug2.json` | 11.2 s | 1.5 GiB | 2.0 GB |
| `configs/dsrl_ex324_a.json` | 26.9 s | 3.4 GiB | 4.1 GB |
| `configs/dsrl_ex324_b.json` | 7.0 s | 1.5 GiB | 2.0 GB |

The configs pin the prep manifests by sha256 (`prep_manifest_sha256`), every file named in a manifest is hashed again at load, and all four runs passed those checks: this is the check of the
scrubbed prep folders (host names and absolute paths removed; every array unchanged except the pools' `host` arrays; README section 10). Three calls are a load test, not a training result.

### The three job types end to end (full asset root, GPU 0 and one CPU worker) and the returns procedure

| Flow | Result |
|---|---|
| `decode run --prior hug1` (6 reference pictures, 32 draws) -> `decode make-jobs` -> `rollout run --protocol mano-1111-p1` | 192 of 192 grasps decoded and rolled; 134 succeed (`success_native`); decode 8.4 s, peak card memory 1.6 GiB |
| `dsrl train` (reference mini prep, seed 3) -> `dsrl readout` -> `rollout run` | read-out of 6 inputs x 3 arms = 18 grasps, rolled; 9 succeed |
| `rollout slice` on `batches/ex308_wuji_steering_readout` | 40 rows picked of 2,016 |
| README section 8: `pack-results`, `tar -cf`, `sha256sum`, unpack, `sha256sum -c SHA256SUMS` inside | 150 files in the results folder; the unpacked folder verifies |

### Public pieces only, plus the user's own MANO file (README section 2; empty asset root, GPU 0)

| Step | Result |
|---|---|
| `assets fetch-hug-source` without the MANO file | `pass: false` with the message, the clone itself done (commit `8d1c52d4`) |
| MANO file placed, then `fetch-hug-source`, `fetch-dinov2`, `fetch-hug1` | all `pass: true` (DINOv2 22 s, HUG1 13 s, sha256 matched); `assets verify --parts hug1,dinov2,mano`: pass |
| `decode run --prior hug1` on the six reference pictures (8 draws) | 48 of 48 grasps finite, 1.6 s, model load 3.7 s, peak card memory 1.0 GiB |

### What the run showed

1. **The rollout-only set is enough for every CPU job:** all 41 checks passed on an asset root made from its four tarballs plus the MANO file (the 200 HUG-Bench grasps under p1 and p2 and the 220 dumbbell grasps under six MANO protocols bit for bit; the robot-hand protocols bit for bit; 18 of 19 production-batch spot checks bit for bit and one `pass_near`, the same single Wuji 1 job as in run 1 (`hug2/s3/stoch/s_a300`, differing by at most 8.6e-5 mm in six diagnostics, all labels equal)).
2. **The GPU results of run 1 reproduce from the tarballs:** decode within 8.2e-6 of the stored RTX 3080 decode, the DSRL port's exactness tests bit for bit, the 20-call training reproduces its record.
3. **A missing or wrong MANO file is caught before any work starts, with a message a recipient can act on** (first table).

### Not tested here

- The real download of the MANO archive from the Max Planck Institute (it needs a registered account): the stand-in was the same file, copied in, with its sha256 checked; the archive's name and layout (`mano_v1_2.zip`, `models/MANO_RIGHT.pkl`) are as the HUG source's README describes them (`mano_v*_*/` with a `models` folder) and as we remember the download page; the sha256 is the check.
- The cloud-drive upload and download and the GitHub invitation and clone (the tarballs were copied with rsync).
- A CPU outside the AMD Zen family (an Intel Xeon reproduced the labels but not every diagnostic: expect `labels_only`), GPUs other than an RTX 3080 and an RTX A6000, the 1,000-call trainings and the full production batches.


## Run 1: the first release commit (asset bundle copied as a folder, with the MANO model inside)

Kept for the record; run 2 repeats the checks that matter.


**What was tested:** the release folder (as committed) plus the asset bundle, copied to our reference host A over the network, with virtual environments built from `requirements/*.txt`. Constraints: `nice -n 19`, at most two CPU workers, GPU 0 only (never the other cards). Host A: AMD Ryzen Threadripper PRO 5975WX (64 threads, shared with other users' jobs), Ubuntu 24.04, NVIDIA driver 570, GPU 0 an RTX A6000.

| Environment | Python | Built from | Size |
|---|---|---|---|
| CPU | 3.10.20 | `requirements/cpu.txt` + `cpu_hands.txt`, README 3.2 (`pip check`: no broken requirements) | 1.6 GB |
| CPU | 3.12.3 | the same files | 1.6 GB |
| GPU | 3.10.20 | `requirements/gpu.txt` + chumpy, README 3.4 (`pip check`: no broken requirements) | 7.4 GB |

**Release folder tested:** 133 files, `SHA256SUMS` digest `45a4d551f03ba202254e67d071948d92508b1df82b62bc171b9bc00b7e367c9f`. The committed folder differs from it only by `docs/SMOKE_TEST.md`, the README (section 9) and `SHA256SUMS`, which were written after the run (a checksum comparison of the two folders lists exactly these three files); `verify --levels L0` and the 200-grasp rollout checks under p1 and p2 were repeated on the committed bytes, with the same result.

**Asset bundle:** copied with rsync (6.1 GB, 4,581 files); `sha256sum -c SHA256SUMS` passed (SHA256SUMS digest `f015781a0fcc7240c0d0aa587a1c62da756263b18775195220e419a1d39c5a72`); `wf.py assets verify` of all 14 parts and of the bundle digest passed.

#### CPU, MANO protocols and production batches (CPython 3.10.20, 2 workers)

`wf.py verify --levels L0,L1,L1b --workers 2`

Result: **PASS**, 32 checks in 789 s. Release `SHA256SUMS` digest `45a4d551f03ba202254e67d071948d92508b1df82b62bc171b9bc00b7e367c9f`. CPython 3.10.20, mujoco 3.10.0, numpy 2.2.6, torch 2.9.1+cpu.

| Check | Status | Labels equal | Every field equal | Rollouts per second | Note |
|---|---|---|---|---|---|
| L0 asset bundle SHA256SUMS | pass_bitwise |  |  |  |  |
| L0 release folder | pass_bitwise |  |  |  |  |
| L0 asset part batches | pass_bitwise |  |  |  |  |
| L0 asset part core | pass_bitwise |  |  |  |  |
| L0 asset part mano | pass_bitwise |  |  |  |  |
| l1_ref200_mano-1111-p1 | pass_bitwise | 200 of 200 | yes | 6.087 |  |
| l1_ref200_mano-1111-p2 | pass_bitwise | 200 of 200 | yes | 6.265 |  |
| l1_w3_mano-1111-p1 | pass_bitwise | 220 of 220 | yes | 4.938 |  |
| l1_w3_mano-1111-p2 | pass_bitwise | 220 of 220 | yes | 5.255 |  |
| l1_w3_mano-1111-p2s-dzm1 | pass_bitwise | 220 of 220 | yes | 4.794 |  |
| l1_w3_mano-1111-p2-dzm1 | pass_bitwise | 220 of 220 | yes | 4.668 |  |
| l1_w3_mano-1111-p2-dzm2 | pass_bitwise | 220 of 220 | yes | 4.036 |  |
| l1_w3_mano-1111-p2-dzp1 | pass_bitwise | 220 of 220 | yes | 6.488 |  |
| l1b_ex330_ex331_census220_mano-1111-p1 | pass_bitwise | 100 of 100 | yes | 3.803 |  |
| l1b_ex330_ex331_census220_mano-1111-p2 | pass_bitwise | 100 of 100 | yes | 3.682 |  |
| l1b_ex330_ex331_census220_mano-1111-p2s-dzm1 | pass_bitwise | 100 of 100 | yes | 3.59 |  |
| l1b_ex330_ex331_census220_mano-1111-p2-dzm1 | pass_bitwise | 100 of 100 | yes | 3.623 |  |
| l1b_ex330_ex331_census220_mano-1111-p2-dzm2 | pass_bitwise | 100 of 100 | yes | 3.282 |  |
| l1b_ex330_ex331_census220_mano-1111-p2-dzp1 | pass_bitwise | 100 of 100 | yes | 4.353 |  |
| l1b_ex305_wuji_dumbbell_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 4.714 |  |
| l1b_ex307_wuji_dumbbell_new_views_part1_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 4.788 |  |
| l1b_ex307_wuji_dumbbell_new_views_part2_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 4.753 |  |
| l1b_ex308_wuji_steering_readout_wuji1-float-p1 | pass_near | 100 of 100 | no | 4.597 | 1 job differs, largest difference 8.6e-05 |
| l1b_ex318_wuji_multiobject_part1_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 6.897 |  |
| l1b_ex318_wuji_multiobject_part2_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 7.711 |  |
| l1b_ex318_wuji_multiobject_part3_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 8.109 |  |
| l1b_ex318_wuji_multiobject_part4_wuji1-float-p1 | pass_bitwise | 100 of 100 | yes | 7.506 |  |
| l1b_ex327_hands_pop9216_sharpa-float-p1 | pass_bitwise | 100 of 100 | yes | 3.423 |  |
| l1b_ex327_hands_pop9216_sharpa-float-p2 | pass_bitwise | 100 of 100 | yes | 3.83 |  |
| l1b_ex327_hands_pop9216_wuji1hr-float-p1 | pass_bitwise | 100 of 100 | yes | 2.967 |  |
| l1b_ex327_hands_pop9216_wuji1hr-float-p2 | pass_bitwise | 100 of 100 | yes | 2.987 |  |
| l1b_ex327_hands_pop9216_mano-1111-p2 | pass_bitwise | 100 of 100 | yes | 6.513 |  |

#### CPU, robot-hand protocols (CPython 3.10.20, 2 workers)

`wf.py verify --levels L1h --workers 2`

Result: **PASS**, 9 checks in 424 s. Release `SHA256SUMS` digest `45a4d551f03ba202254e67d071948d92508b1df82b62bc171b9bc00b7e367c9f`. CPython 3.10.20, mujoco 3.10.0, numpy 2.2.6, torch 2.9.1+cpu.

| Check | Status | Labels equal | Every field equal | Rollouts per second | Note |
|---|---|---|---|---|---|
| l1h_ref200_sharpa-float-p1 | pass_bitwise | 200 of 200 | yes | 3.413 |  |
| l1h_pilot64_sharpa-float-p1 | pass_bitwise | 64 of 64 | yes | 1.195 |  |
| l1h_ref200_sharpa-float-p2 | pass_bitwise | 200 of 200 | yes | 3.743 |  |
| l1h_pilot64_sharpa-float-p2 | pass_bitwise | 64 of 64 | yes | 1.16 |  |
| l1h_ref200_wuji1hr-float-p1 | pass_bitwise | 200 of 200 | yes | 3.847 |  |
| l1h_pilot64_wuji1hr-float-p1 | pass_bitwise | 64 of 64 | yes | 1.79 |  |
| l1h_ref200_wuji1hr-float-p2 | pass_bitwise | 200 of 200 | yes | 4.087 |  |
| l1h_pilot64_wuji1hr-float-p2 | pass_bitwise | 64 of 64 | yes | 1.686 |  |
| l1h_ref200_wuji1-float-p1 | pass_bitwise | 200 of 200 | yes | 8.61 |  |

#### CPU, MANO protocols (CPython 3.12.3, 2 workers)

`wf.py verify --levels L0,L1 --workers 2`

Result: **PASS**, 12 checks in 347 s. Release `SHA256SUMS` digest `45a4d551f03ba202254e67d071948d92508b1df82b62bc171b9bc00b7e367c9f`. CPython 3.12.3, mujoco 3.10.0, numpy 2.2.6, torch 2.9.1+cpu.

| Check | Status | Labels equal | Every field equal | Rollouts per second | Note |
|---|---|---|---|---|---|
| L0 asset bundle SHA256SUMS | pass_bitwise |  |  |  |  |
| L0 release folder | pass_bitwise |  |  |  |  |
| L0 asset part core | pass_bitwise |  |  |  |  |
| L0 asset part mano | pass_bitwise |  |  |  |  |
| l1_ref200_mano-1111-p1 | pass_bitwise | 200 of 200 | yes | 5.926 |  |
| l1_ref200_mano-1111-p2 | pass_bitwise | 200 of 200 | yes | 6.047 |  |
| l1_w3_mano-1111-p1 | pass_bitwise | 220 of 220 | yes | 4.797 |  |
| l1_w3_mano-1111-p2 | pass_bitwise | 220 of 220 | yes | 5.21 |  |
| l1_w3_mano-1111-p2s-dzm1 | pass_bitwise | 220 of 220 | yes | 4.701 |  |
| l1_w3_mano-1111-p2-dzm1 | pass_bitwise | 220 of 220 | yes | 4.496 |  |
| l1_w3_mano-1111-p2-dzm2 | pass_bitwise | 220 of 220 | yes | 3.948 |  |
| l1_w3_mano-1111-p2-dzp1 | pass_bitwise | 220 of 220 | yes | 6.139 |  |

#### GPU 0: decode and DSRL-NA

`CUDA_VISIBLE_DEVICES=0 wf.py verify --levels L0,L2,L3`

Result: **PASS**, 13 checks in 33 s. Release `SHA256SUMS` digest `45a4d551f03ba202254e67d071948d92508b1df82b62bc171b9bc00b7e367c9f`. CPython 3.10.20, mujoco 3.10.0, numpy 2.2.6, torch 2.9.1+cu128.

| Check | Status | Labels equal | Every field equal | Rollouts per second | Note |
|---|---|---|---|---|---|
| L0 asset bundle SHA256SUMS | pass_bitwise |  |  |  |  |
| L0 release folder | pass_bitwise |  |  |  |  |
| L0 asset part dinov2 | pass_bitwise |  |  |  |  |
| L0 asset part dumbbell50k | pass_bitwise |  |  |  |  |
| L0 asset part hug1 | pass_bitwise |  |  |  |  |
| L0 asset part hug2 | pass_bitwise |  |  |  |  |
| L0 asset part mano | pass_bitwise |  |  |  |  |
| L2 decode hug2 | pass_tolerance |  |  |  | max difference of the 99-D output 3.0e-06, of the landmarks 5.5e-07 m (NVIDIA RTX A6000) |
| L2 decode hug1 | pass_tolerance |  |  |  | max difference of the 99-D output 3.0e-06, of the landmarks 1.3e-07 m (NVIDIA RTX A6000) |
| L2 decode dumbbell50k | pass_tolerance |  |  |  | max difference of the 99-D output 8.2e-06, of the landmarks 6.0e-07 m (NVIDIA RTX A6000) |
| L3 T1_terminal_backup_skip | pass_bitwise |  |  |  | terminal-backup skip is exact: every parameter bit-identical to the authors' full backup after 3 train() calls |
| L3 T3_noise_bounds | pass_bitwise |  |  |  | the deterministic noise lies in [-bound, bound]; the noise critic's input is noise / bound |
| L3 T4_mini_training | pass_tolerance |  |  |  | 20 calls on the reference mini prep; worst relative difference to the reference 0 (tolerance 0.02) |

### The three job types end to end (RTX A6000 and one CPU worker)

| Step | Command (abridged) | Result |
|---|---|---|
| decode | `decode run --prior hug1 --inputs reference/decode/input_set.json --store reference/decode/input_store --draws 32` | 6 pictures x 32 draws = 192 grasps, all finite, 3.6 s (model load 4.1 s), peak card memory 1,598 MiB, landmark convention error 1e-4 mm |
| jobs, rollout | `decode make-jobs ...`, `rollout run --protocol mano-1111-p1 --workers 1` | 192 jobs, 192 rolled, 134 successes, 65 s |
| train | `dsrl train --config reference/dsrl/mini_config.json --seed 3` | 20 calls in 6.6 s, peak card memory 843 MiB |
| read-out, rollout | `dsrl readout --train-out ...` (no `--config`), then `rollout run` | 3 arms x 6 inputs = 18 jobs; `readout.json` records seed 3, the seed of the training run; 18 rolled, 9 successes |
| slice | `rollout slice --jobs ex308_jobs.npz --expected ... --n 40` | 40 of 2,016 rows |
| pack-results | `pack-results --folder <decode output>` | `RETURN.json` and `SHA256SUMS` over 3 files |

### What the run showed

1. **CPU physics, MANO and hand protocols: bit for bit.** Every L1 and L1h check reproduced the stored reference records in every label and every other field, in CPython 3.10.20 and in
   CPython 3.12.3 (same numbers, same fields). The reference records were made on other machines and with an older interpreter, so this is the cross-host, cross-version evidence
   behind the README's statement. Speed at two workers: 4.0 to 6.5 rollouts per second for the MANO protocols, 3.4 to 4.1 for Sharpa and Wuji 1 hr on 200 grasps (the fit is part of every rollout), 8.6 for
   `wuji1-float-p1`.
2. **Production batches (L1b): 18 of 19 spot checks bit for bit, one `pass_near`.** 100 rows spread over each batch were rolled on host A and compared with our stored results. The Wuji 1 batches
   (EX305, EX307, EX308, EX318) were stored from host C (AMD EPYC 7542, Zen 2): seven of the eight spot checks match in every field; in EX308, one job (`hug2/s3/stoch/s_a300`) differs from host C's stored
   result in six diagnostic fields (object rise, penetration depth) by at most 8.6e-5 mm, with its label and the other 99 jobs equal. That is floating-point noise between CPU models in one rollout
   of 800; it is why `compare` has the status `pass_near` (labels all equal, at most 2 % of the jobs differ, by at most 1e-3). The first run of this test flagged it as `labels_only`
   and also flagged the stored EX327 MANO p2 file, whose four trace-derived fields are all zero because that run had the trace off; the shipped copy of that file no longer stores those four fields.
3. **GPU decode: within 1.1e-5 of a different GPU model.** The six reference pictures decoded on an RTX A6000 differ from the stored decode made on an RTX 3080 by at most 8.2e-6 in the 99-D output and 6e-7 m in the
   landmarks (HUG2 3.0e-6, HUG1 3.0e-6, dumbbell 50k 8.2e-6); the same GPU twice gives identical decodes (difference 0). Decoding 192 rows takes 1.7 to 3.1 s, the peak card memory is 1.6 GB.
4. **DSRL-NA: exact where it can be, repeatable where it cannot.** The port's terminal-backup skip is bit-identical to the authors' full backup over 101 tensors; the noise bounds hold. Twenty calls of
   training on the mini prep take 6.6 s on the A6000 (peak card memory 0.84 GB), reproduce the stored record bit for bit on the same GPU in three repeats, and a TF32 stress run
   (a far larger numerical perturbation than another GPU model) moves the worst compared number by 0.19 %, against a tolerance of 2 %.
5. **The three job types, end to end.** `decode run` (HUG1, six pictures, 32 draws) -> `decode make-jobs` -> `rollout run` rolled 192 grasps; `dsrl train --seed 3` -> `dsrl readout` (arms own, stoch, normrand)
   -> `rollout run` rolled 18 grasps and recorded seed 3 (the read-out takes the seed from the training run); `rollout slice` and `pack-results` ran.
6. **Public pieces from scratch** (tested earlier into an empty asset root, before the bundle was copied): `assets fetch-hug-source` cloned github.com/KevinyWu/hug at the pinned commit and linked the MANO file,
   `assets fetch-dinov2` downloaded the pinned DINOv2 revision, `assets fetch-hug1` downloaded the HUG1 weights and their sha256 matched.
7. **Before this run** an independent read of the release found and we fixed: a stale `SHA256SUMS`, the read-out ignoring the training seed (`normrand` would have used seed 1), output folders created before a failing
   set-up step, `verify` refusing a second run into the same `--out`, unvalidated `--levels` / `--prior` values, machine names left in two reference files and a branch name in the protocol files, and several wrong numbers and
   cross-references in the README.

### Not tested here

- A CPU outside the AMD Zen family (we know one Intel Xeon reproduced the labels but not every diagnostic, so it would report `labels_only`); a GPU other than an RTX 3080 and an RTX A6000; an operating system other than Ubuntu.
- The full production batches (only 100 rows of each, for time); the `fetch-*` downloads against the bundle copy (they were tested into an empty root).
- The 1,000-call DSRL trainings and the read-outs of the three large preps (`prep_ex315_hug2`, `prep_ex324_a`, `prep_ex324_b`): their files are checked against their manifests, and the 20-call run exercises the same code.

### Commands

```bash
H=<smoke-test folder>; R=$H/release; WF=$R/code/wf.py; export WF_ASSETS=$H/assets_bundle      # the asset bundle copied with rsync
cd $WF_ASSETS && sha256sum -c --quiet SHA256SUMS                                                # 4,581 files
$H/venv310/bin/python $WF assets link-hands; $H/venv310/bin/python $WF assets fetch-hug-source
$H/venv310/bin/python $WF assets verify --parts core,mano,hands,licences,inputs,hug1,hug2,dumbbell50k,dinov2,batches,prep_ex299_hug1,prep_ex315_hug2,prep_ex324_a,prep_ex324_b
nice -n 19 $H/venv310/bin/python $WF verify --levels L0,L1,L1b --workers 2 --out out/cpu310
nice -n 19 $H/venv310/bin/python $WF verify --levels L1h --workers 2 --out out/hands310
CUDA_VISIBLE_DEVICES=0 nice -n 19 $H/venvgpu310/bin/python $WF verify --levels L0,L2,L3 --out out/gpu
nice -n 19 $H/venv312/bin/python $WF verify --levels L0,L1 --workers 2 --out out/cpu312
# the job-type flows: decode run / make-jobs / rollout run; dsrl train / readout / rollout run; rollout slice; pack-results (as in the table above)
```
