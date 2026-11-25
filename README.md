# wuji-floating release (2026-11-25, revision r8 = r7a + the object-frame prior): run our floating-hand grasp experiments on your own GPUs and CPUs

**Starting from nothing? Follow [SETUP.md](SETUP.md)** (in Chinese: [SETUP_zh.md](SETUP_zh.md)): it takes a fresh cluster account to a verified machine and to the first batch, with the expected output of every command.
For the first verification run there are two shorter documents: the one-page checklist [QUICKSTART.md](QUICKSTART.md) (Chinese: [QUICKSTART_zh.md](QUICKSTART_zh.md)) and the work order for an AI agent that operates your cluster, [QUICKSTART_AGENT.md](QUICKSTART_AGENT.md) (every command, its expected output and what to do on each failure).
This README is the reference behind them.

**Revision r8 (2026-11-25): a second grasp prior.** r8 is r7a plus the object-frame HUG2 prior (prior ids `hug2obj_50k` and `hug2obj`; [docs/HUG2OBJ.md](docs/HUG2OBJ.md)). Nothing of r7a changes: the old prior ids, their pins, the protocols, the reference results and every r7a batch run as before (the steps below that name the tag `wf-20261123-r7a` are r7a's; for r8 use the tag `wf-20261125-r8`, or let an r8 batch make the folder from your r7a folder). The model's code is part of this repository (package `code/tools/float/objframe/`, the model authors' MIT-licensed code integrated as our own package, provenance in its `PROVENANCE.json`). **A run with the new prior needs one data part that is in neither this repository nor the asset tarballs: the checkpoint part (`ckpt/hug2obj/hug_objpcl_aug_50k.pt`, the model authors' weights, sha256 c4ef2014...). It is delivered with the batch folder that uses it, `batches/ex397_b8/` of the project's dataset repository (file `ex397_ckpt_hug2obj_50k.tar`, listed with its sha256 in that folder's `PARTS.txt`); until it is uploaded it is not published. Without it a command with `--prior hug2obj_50k` stops at once with one line that names the missing checkpoint file.**

This folder lets you run the simulation experiments of the wuji-grasplift floating-hand line on compute we do not control, and send **numbers** back (never code) so that they can be compared with ours.
The work arrives as **issued batches** (section 5): we make a batch folder, you run it with `wf.py batch run` (on one machine, or as a SLURM array), you send back one archive, we verify it from the numbers alone
(section 9), and we issue the next batch. A batch is made of up to five kinds of jobs, numbered GPU first (type 4, the census builder, you run yourself):

1. **GPU DSRL-NA training and read-out (job type 1, the main load).** The authors' steering method (DSRL-NA) is trained on our prepared data (many seeds, configurations, priors and training-view budgets) and read out on held-out inputs (section 6).
2. **GPU prior decode (job type 2).** A stored picture of an object goes in, grasps predicted by HUG2, HUG1 or our dumbbell fine-tune come out: censuses of many views (section 7).
3. **CPU rollouts (job type 3).** The evaluation that follows 1 and 2 on the cluster's CPUs: a predicted grasp goes in, a success label and diagnostics come out; no GPU (section 8).
4. **Census builder (job type 4, revision r7a).** You run it on objects you hold, to make a labelled training set in the prep format (section 10.8); batches do not issue it.
5. **Searched constant (job type 5, revision r7a).** The best single noise for a set of training inputs, found by a cross-entropy search with real decodes and rollouts; it runs as the batch group kind `cem_readout` (section 10.7).

**Status of this package (read before anything else)**

| Item | Status |
|---|---|
| Licence of our own code and records | **MIT** ([LICENSE.txt](LICENSE.txt), Copyright (c) 2026 the wuji-floating authors). Files derived from third-party code keep their own licence: [THIRD_PARTY.md](THIRD_PARTY.md) |
| How you receive the code and the assets, and how results come back | **Code:** the public GitHub repository https://github.com/Exiam6/wuji-floating-release (anyone can clone it; it holds no assets). **Assets:** tarballs in a drive folder that the project shares, checked with `SHA256SUMS` (the assets do not go through GitHub). **Results** (when you run a batch the project issued to you): one archive uploaded to `returns/<your name>/<date>/` in the drive folder shared with you for returns, and you tell us its sha256. Sections 3.0 and 8 |
| How work is issued, run and checked | Work arrives as **issued batches** that you run with `wf.py batch run` (one machine or a SLURM array); a new experiment is **data** (jobs files, input sets, DSRL configs): no code ever comes back. Every jobs file holds known-answer jobs (canaries) and `batch pack` returns the evidence with the numbers. Sections 5 and 9 |
| The assets (HUG2, 50k fine-tune, HUG1, HUG-Bench simulation assets, the evaluator, our production batches, ...) | **Each part keeps the licence of its source**, stated per part with the attribution it requires in [ASSETS.md](ASSETS.md) and `ASSET_TERMS.txt`: HUG-Bench data and Google Scanned Objects CC-BY-4.0, the HUG-Bench evaluator MIT, HUG1, HUG2 and the HUG source MIT, DINOv2 Apache-2.0, the Wuji and Sharpa hand models Apache-2.0, wuji-retargeting MIT; our own data (the pictures, the production batches, the DSRL training sets) and the 50k fine-tune of HUG2 MIT. The part `licences` holds the licence texts. **The HUG1 weights are not in the tarballs** (`wf.py assets fetch-hug1` downloads the public file, section 3.4). **The MANO model is not in the tarballs:** you register at the Max Planck Institute and download it yourself under its licence (section 3.0) |

Everything below was run and checked on our own machines before release; section 10 lists what was checked where.

## 1. What the pipeline does

A grasp prior (HUG) looks at one RGB-D picture of an object and predicts how a human hand would grasp it: a wrist pose and 15 finger
rotations of the MANO hand model. The **evaluator** (a MuJoCo 3.10.0 simulation built by the HUG authors, "HUG-Bench") executes
such a grasp with a hand that floats on a six-degree-of-freedom wrist: approach, finger curl, settle, then a 0.5 m lift. The label
`success_native` is 1 when the object no longer touches the table at t = 4.0 s in a valid simulation. We also execute the same
grasps with two robot hands (Sharpa Wave, Wuji Hand 1), whose fingers are fitted to the predicted hand landmarks. **DSRL-NA**
("Steering Your Diffusion Policy with Latent Space Reinforcement Learning", 2025) learns, from labelled
executions, which starting noise to give the prior so that the decoded grasp succeeds more often.

The evaluator has versions (**protocols**); a result is comparable only with results of the same protocol, and every result file
carries the protocol name and digest:

| Protocol | What differs |
|---|---|
| `mano-1111-p1` | the HUG-Bench evaluator as frozen for campaign mano-1111 on 2026-11-11 (MANO capsule hand, wrist under gravity) |
| `mano-1111-p2` | the wrist carries the hand's weight (as an arm's gravity model would) |
| `mano-1111-p2s-dzm1`, `-p2-dzm1`, `-p2-dzm2`, `-p2-dzp1` | p2 with the commanded wrist height offset by k free sags of the hand (EX331) |
| `sharpa-float-p1/p2`, `wuji1hr-float-p1/p2` | Sharpa Wave and Wuji Hand 1 through the multihand retargeter (EX327), p1 / p2 wrist |
| `wuji1-float-p1` | Wuji Hand 1 in the frozen October evaluator (EX292, EX305, EX307, EX308, EX318) |

## 2. What you need

| Item | Requirement |
|---|---|
| Machine | Linux x86-64; tested on Ubuntu 22.04 and 24.04. A workstation, or the nodes of a SLURM cluster with a login node. |
| GPU (job types 1 and 2) | An NVIDIA card with at least 8 GB and a driver for CUDA 12.8 (driver 570 or newer; tested: RTX 3080 10 GB with driver 580, RTX A6000 48 GB with driver 570). The pinned stack is torch 2.9.1 built for CUDA 12.8 (`requirements/gpu.txt`); builds for another CUDA version are untested, ask us first. One job uses one card; a job is never split across cards. |
| Card memory per job (measured, section 10) | prior decode: 1.6 GiB peak for the 192-row check and 1.8 GiB for the 7,040-row census; DSRL-NA training: 0.84 GiB for the 20-call check and 3.4 GiB (3,452 MiB) peak for the 1,000-call training on the largest prep (`prep_ex324_a`). Plan 4 GB per DSRL job and 2 GB per decode job; a 24 GB card takes several (`--gpus 0,0,0`). |
| Host memory | A DSRL-NA job needs about 10 GB (the replay buffer is allocated at 10 million rows, about 9 GB of address space, mostly untouched, and the pool is loaded); a rollout worker 0.4 to 0.6 GB. |
| CPU (job type 3, and the CPU parts of the others) | Any x86-64 CPU; one core per rollout worker. **AMD Zen 2 and Zen 3 CPUs reproduced our stored results bit for bit** (tested: host B Ryzen 9 3900X, host A Threadripper PRO 5975WX, host C EPYC 7542), with one measured exception: of 800 Wuji Hand 1 rollouts we re-rolled on host A against results stored from host C, one differs in six diagnostics by less than 1e-4 mm (all labels equal; status `pass_near`, section 4). One Intel Xeon (Platinum 8468, AVX-512) reproduced every label but only 14 of 200 jobs in every diagnostic field: such a host counts as a separate numerical host (section 4). |
| Python | 3.10 for both environments (tested: 3.10.12 and 3.10.20); CPython 3.12.3 gave bit-equal CPU results (not tested for the GPU environment). |
| Disk | The release folder is about 8 MB. The CPU environment is about 1.6 GB, the GPU environment about 7.5 GB, the asset bundle about 23.3 GB in full (core part 0.14 GB, the batches 0.15 GB; the 8 DSRL training sets are 19.6 GB of it, and the three DexGraspNet parts of r7a 14.6 GB of that). It arrives as tarballs of the same total size, so keep room for both while unpacking (the rollout-only set: 0.30 GB, twice). Put the environments and the assets on node-local or scratch disk, not on a network home directory (SETUP.md, Troubleshooting). |
| Internet | During setup only: PyPI, download.pytorch.org, data.pyg.org, github.com (the HUG source and the code), and huggingface.co only if a model part is missing. **After setup nothing needs the network**: decode runs with `HF_HUB_OFFLINE=1`, training and rollouts read only the asset root. Do the setup on a login node and run the batches on offline compute nodes if you like. |
| Accounts | The drive link (asset tarballs and, for issued batches, the folder for your returns) and a free MANO account at the Max Planck Institute; the code needs no account (public repository). |

**Tolerance policy per GPU model.** On the **same GPU model as ours (RTX A6000, driver 570)** a DSRL-NA training is bit for bit the same as ours: the control run of 2026-11-20 reproduced the 41 log lines and all 1,076 checkpoint tensors of a registered run exactly (section 10).
**On another GPU model** the kernels are not bit-identical: the decode differs from ours by float rounding (L2 measured at most 8.2e-6 for six pictures between an RTX 3080 and an RTX A6000; over the 7,040-row census at most 3.2e-5 for HUG2 and 2.6e-5 for HUG1, and for the dumbbell 50k 7,033 rows within 1e-4 and 2 rows up to 4.5e-2, samples near a branch of the flow), so level L2 allows 1e-3 for its 192 rows; a 20-call training stays within 2 % relative of ours (L3), and later calls diverge slowly,
so our verifier gates only the first 100 logged calls of a training on another GPU model. A rollout of grasps decoded on another GPU can change the label of a few jobs: in the 15 cells of the control run (an RTX A6000 against our RTX 3080 decodes; 5,248 rollouts per cell) the success counts differed from ours by at most 9 (0.17 points) and 0.3 % of the labels flipped (section 10.1). Rollouts on one numerical CPU host are exact.
A change of GPU model, driver or torch version is a change of numerical host: send a new fingerprint (SETUP.md step 8).

**If you hold only public pieces** (the HUG source, the HUG1 weights, DINOv2, the hand models, the retargeter, and your own MANO licence): job type 2 runs with HUG1
(`decode run --prior hug1`, and level L2 for `hug1`; for new objects you supply your own RGB-D pictures, rendering is not shipped, and the six pictures in `reference/decode/` work too); job type 1 trains from a prep folder alone (no model weights; the `prep_ex299_hug1` folder is our data derived from the public
prior) and reads out with HUG1; job type 3 needs the asset parts `core` (the HUG-Bench evaluator, MIT, and the simulation assets, CC-BY-4.0) and, for the robot hands, `hands`. Every part of the tarballs is openly
licensed (ASSETS.md has the licence of each part and the table "Which job types run with openly licensed parts only"). The MANO file is not in them: you download it from the Max Planck Institute under your own licence and put it at `$ASSETS/mano/models/MANO_RIGHT.pkl` (section 3.0); it must be the
right-hand model with the sha256 listed in `asset_manifests/mano.sha256`, or the protocol digests refuse it, and `wf.py assets verify --parts mano` checks it.

## 3. Set up (once per machine)

**[SETUP.md](SETUP.md) walks through these steps in order, with the expected output of every command and the timings; the commands below are the same ones. Follow SETUP.md; this section is the reference.**

**3.0 Get the code and the assets.** The code is the public GitHub repository https://github.com/Exiam6/wuji-floating-release (no account is needed to clone it; the assets are not in it). `RELEASE` below is the folder of your clone that holds this README.
The assets arrive as tarballs in a drive folder that the project shares (they do not go through GitHub). The folder `full` holds all 18 tarballs with `SHA256SUMS`, `ASSET_TERMS.txt` and `README.txt`: that is the **full set** (every part but `mano` and `hug1`, which you get yourself: 23.3 GB, needed for the GPU jobs; the three parts of revision r7a, 14.6 GB, are needed only by the 200-object batches and the census builder, section 10.8). The **rollout-only set** (`core`, `hands`, `batches`, `licences`: about 0.30 GB, enough for job type 3 once the MANO file is in place) is a subset of the same files: download just those four tarballs and the `SHA256SUMS` from `full` and check with `sha256sum -c --ignore-missing SHA256SUMS`; a sibling folder `rollout_only` repeats only that subset's manifest files. Read `ASSET_TERMS.txt` in the folder first: it gives the licence of every
part and the attribution it requires.

```bash
RELEASE=/path/to/wuji_floating_20261120            # your clone of the repository: the folder that holds this README
ASSETS=/path/to/wuji_floating_assets_20261120      # the asset root: an empty folder that the tarballs unpack into (it is never inside git)
ENV=/path/to/venv310                               # outside both folders

# 3.0 the code: clone the repository (public; the tag of this revision is wf-20261123-r7a)
git clone https://github.com/Exiam6/wuji-floating-release.git $RELEASE

# 3.0 the assets: download the folder `full` of the drive link into one folder (a browser, or `rclone copy <your remote>:<the shared folder>/full /path/to/the/downloaded/folder --progress`), verify the tarballs, then unpack all of them into the asset root (about a minute for the full set)
cd /path/to/the/downloaded/folder && sha256sum -c SHA256SUMS                    # every line must end with OK
mkdir -p $ASSETS && for t in *.tar; do tar -xf "$t" -C $ASSETS; done
```

`git clone` needs no account and no credentials. A cluster node without access to github.com can use a copy: clone on a machine that has access and `rsync -a` the folder, `.git` included, to the node; the folder is the same release
(section 4, level L0, checks it against `SHA256SUMS`). To update to a later release, `git pull` in the clone.

**MANO model (you download it yourself).** The MANO right-hand model is not in the tarballs: every user registers at the Max Planck Institute and accepts its MANO licence
(non-commercial research use) themselves.

1. Register (free) at https://mano.is.tue.mpg.de and accept the MANO licence.
2. Download the MANO models, version 1.2 (the archive is named `mano_v1_2.zip`), and unzip it. The file you need is the right-hand model `models/MANO_RIGHT.pkl`; the archive's other files are not used.
3. Copy that one file into the asset root and check it (no Python environment is needed yet):

```bash
mkdir -p $ASSETS/mano/models && cp /path/to/mano_v1_2/models/MANO_RIGHT.pkl $ASSETS/mano/models/
sha256sum $ASSETS/mano/models/MANO_RIGHT.pkl        # must print 45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767
python3 $RELEASE/code/wf.py --assets $ASSETS assets verify --parts mano      # the same check by our tool: PASS, or a message that says what is missing or wrong
```

That sha256 is the one every result of ours was made with (`asset_manifests/mano.sha256`). If your file differs, stop and tell us: the protocol digests refuse any other MANO file, and a different version would change every result.
Every command that needs the model (`verify`, `rollout run`, `decode run`, `dsrl readout`) checks it first and prints the same message when it is missing.

```bash
# 3.1 check what you received (the check reads every file: about a minute)
cd $RELEASE && sha256sum -c --quiet SHA256SUMS && echo "release intact"
cd $ASSETS  && sha256sum -c --quiet $RELEASE/asset_manifests/SHA256SUMS.all && echo "assets intact"     # with only some parts (the rollout-only set): add --ignore-missing

# 3.2 CPU environment (job type 3, the rollouts). `python3.10` must be a CPython 3.10 on your PATH (tested: 3.10.12 and 3.10.20). Many clusters have none: install one (conda, pyenv, or your module system)
# and put its folder first on PATH, or write its full path instead of `python3.10` in the two `-m venv` lines below.
python3.10 -m venv --without-pip $ENV
curl -sSfL https://bootstrap.pypa.io/get-pip.py -o $ENV/get-pip.py
$ENV/bin/python $ENV/get-pip.py --no-cache-dir pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 packaging==26.2
$ENV/bin/python -m pip install --no-cache-dir -r $RELEASE/requirements/cpu.txt
$ENV/bin/python -m pip install --no-cache-dir --no-build-isolation --no-deps \
    "chumpy @ https://github.com/mattloper/chumpy/archive/580566eafc9ac68b2614b64d6f7aaa84eebb70da.zip"
$ENV/bin/python -m pip check                                                    # "No broken requirements found."
# only for the robot-hand protocols (sharpa-float-*, wuji1hr-float-*, wuji1-float-p1):
$ENV/bin/python -m pip install --no-cache-dir -r $RELEASE/requirements/cpu_hands.txt
rm -rf $ENV/lib/python3.10/site-packages/cmeel.prefix/include                   # 160 MB of C++ headers no rollout reads

# 3.3 tell the tools where the assets are (one of the three)
export WF_ASSETS=$ASSETS                       # or write --assets $ASSETS right after wf.py (a top-level option, before the command), or $RELEASE/wf_config.json (wf_config.example.json)
$ENV/bin/python $RELEASE/code/wf.py assets link-hands                           # only with the `hands` part: makes the hand models visible to the code

# 3.4 GPU environment (job types 1 and 2), a second venv
GPUENV=/path/to/gpuenv
python3.10 -m venv --without-pip $GPUENV && curl -sSfL https://bootstrap.pypa.io/get-pip.py -o $GPUENV/get-pip.py
$GPUENV/bin/python $GPUENV/get-pip.py --no-cache-dir pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 packaging==26.2
$GPUENV/bin/python -m pip install --no-cache-dir -r $RELEASE/requirements/gpu.txt
$GPUENV/bin/python -m pip install --no-cache-dir --no-build-isolation --no-deps \
    "chumpy @ https://github.com/mattloper/chumpy/archive/580566eafc9ac68b2614b64d6f7aaa84eebb70da.zip"
$GPUENV/bin/python $RELEASE/code/wf.py assets fetch-hug-source                  # clones the public HUG repository at the pinned commit, links MANO into it
$GPUENV/bin/python $RELEASE/code/wf.py assets fetch-dinov2                      # only if the bundle's `dinov2` part is missing (needs internet)
$GPUENV/bin/python $RELEASE/code/wf.py assets fetch-hug1                        # the public HUG1 weights (not in the tarballs since revision r6; sha256 checked, needs internet)
```

The `fetch-*` commands write into the asset root (`hug_source/`, `hf/`, `ckpt/hug1/`), so a GPU user needs it writable the first time; for a read-only or shared bundle point `WF_HUG_SOURCE=/writable/path`
at the HUG source checkout and fetch the rest into a copy. `fetch-hug-source` links the MANO file by absolute path: run it again after moving the bundle.

**Mirroring the assets onto a cluster.** The tarballs are plain tar files with a `SHA256SUMS` list: download them once (on any machine), check them with `sha256sum -c SHA256SUMS`, copy the folder to the cluster filesystem with `rsync -a --partial --progress`, and unpack once into a shared asset root that every user's `WF_ASSETS` points to. While a batch runs nothing writes into the asset root (results go where you say; the runner refuses to write inside it); only `fetch-hug-source`, `fetch-dinov2` and `fetch-hug1` do, so run them first and then make the asset root read-only if you like (`WF_HUG_SOURCE` can name the HUG source checkout elsewhere). Environments and assets belong on node-local or scratch disk, not on a network home directory (SETUP.md, Troubleshooting).

Nothing is ever written inside `$RELEASE` except the link `code/assets` (hand models) and your `wf_config.json` (the `.git` folder of your clone is ignored). Results go where you say. Do not edit or add files inside the
release folder: `rollout run` refuses to start when any file differs from `SHA256SUMS` or an unlisted file is present, because every result file records the digest of that
`SHA256SUMS` (`run_info.sha256sums_sha256`) and a result is only as good as the code it came from. Keep notes, logs and outputs elsewhere.

## 4. The verification suite: run it first, and again after any change of machine, driver or Python

```bash
PY=$ENV/bin/python
$PY $RELEASE/code/wf.py verify --levels L0,L1 --workers 4 --out /path/to/verify_cpu          # CPU physics, MANO protocols
$PY $RELEASE/code/wf.py verify --levels L1h --workers 4 --out /path/to/verify_hands          # robot-hand protocols (needs the hands part)
$PY $RELEASE/code/wf.py verify --levels L1b --workers 4 --out /path/to/verify_batches        # spot check of our production batches (needs the batches part)
CUDA_VISIBLE_DEVICES=0 $GPUENV/bin/python $RELEASE/code/wf.py verify --levels L0,L2,L3 --out /path/to/verify_gpu   # decode and DSRL
```

Every level writes its records into `--out` (a new or empty folder: a check never overwrites a result file, so a second run needs a new folder); the last line printed is `VERIFY PASS` or `VERIFY FAIL`.
`--levels L0` alone checks the asset parts that are present (`--parts core,mano,...` names them).

| Level | What it checks | Pass means |
|---|---|---|
| L0 | the release folder against `SHA256SUMS`; the asset parts the other levels need (your MANO file included) against `asset_manifests/*.sha256` | every file present and equal |
| L1 | the 200 HUG-Bench reference grasps under `mano-1111-p1` and `-p2`, and the 220 dumbbell census reference grasps under all six MANO protocols, rolled on your CPU and compared with the stored reference records | `pass_bitwise`: every label **and every other record field** equal bit for bit |
| L1h | the robot-hand protocols (200 + 64 reference grasps each for Sharpa and Wuji 1 hr, 200 for `wuji1-float-p1`) | `pass_bitwise` |
| L1b | 100 rows spread over each of our production batches (section 8, needs the `batches` part), rolled and compared with our stored results | every label equal, bit for bit or `pass_near` (at most 2 % of the jobs differ in a diagnostic, by at most 1e-3) |
| L2 | six dumbbell pictures decoded with each prior whose weights you hold (32 codebook noise rows each) against the stored decode | the largest absolute difference of the 99-D output, of the landmarks (m), of the wrist translation (m), of the wrist rotation matrix entries and of the finger angles (rad) is at most 1e-3, and the same rows are finite |
| L3 | the port's exactness tests on the CPU (bit for bit), and 20 calls of DSRL-NA training on a small prep against the stored reference | exact tests equal; every compared training number within 2 % relative |

A machine's results count only after the levels you use have passed there; send us the report folders and `wf.py fingerprint --out host_fingerprint.json` (SETUP.md step 8) before the first batch. Four outcomes are possible for a CPU comparison: `pass_bitwise`
(go), `pass_near` (every label equal and at most 2 % of the jobs, at least one, differ in a diagnostic by at most 1e-3 in the field's unit: floating-point noise of another CPU model in a rare rollout;
accepted only at level L1b, where our results were stored on another host), `labels_only` (every label equal, more or larger differences: this CPU computes slightly different physics; stop and tell us, because our
comparisons flip on single labels and must stay on one numerical host), `fail` (stop and tell us). GPU kernels are not bit-identical across
GPU models (we measured at most 1.1e-5 between an RTX 3080 and an RTX A6000 in L2, and the same GPU twice gives identical decodes and an identical training run), which is why L2 and L3 have stated tolerances; if your L2 maxima are above 1e-4, tell us.

Measured on our reference host A (AMD Threadripper PRO 5975WX, Python 3.10.20, 2 workers, nice 19, a shared machine): L0 + L1 + L1b + L1h took 1,211 s (an earlier run: 789 s for L0 + L1 + L1b and 424 s for L1h) and
L0 + L2 + L3 35 s on an RTX A6000, each ending in `VERIFY PASS` (section 10).

## 5. The unit of work: an issued batch

**The loop: we issue, you run, you return, we verify, we issue the next.**

1. **We issue a batch**: a folder (we send it as one archive) with `BATCH.json` (the release it was issued for, by git commit and by the digest of its `SHA256SUMS`, and the **groups** of jobs: one group is one experiment, with its protocols and their digests),
   the jobs files and configs it needs (`jobs/`, `configs/`), `INSTRUCTIONS.txt`, and its own `SHA256SUMS`. Known-answer jobs (canaries, section 9.2) are already inside every jobs file we send. **Nothing in a batch is code.** A batch is never edited; `wf.py batch run` refuses an edited one,
   and refuses a batch issued for another release than your checkout (update with `git fetch --tags && git checkout <tag>`).
2. **You run it**: `wf.py batch run` on one machine, or a SLURM array (below). **3. You return one archive**: `wf.py batch pack` writes it. **4. We verify it** from the numbers alone (section 9.4) and tell you the verdict. **5. We issue the next batch.**

### 5.1 What a group is

The steps of a group are fixed by this release (`code/wf_lib/batch_cmd.py`), not by the batch. Every step has its own folder `RESULTS/<group>/<step>/`.

| Group kind | Steps | What the batch gives it |
|---|---|---|
| `dsrl_readout` | `train` (GPU: `dsrl train`) -> `readout` (GPU: `dsrl readout`, adds canaries) -> `roll_<protocol>` (CPU: `rollout run`), one per protocol. Without `arms` the group stops after the training | a config (`configs/*.json`: data), a seed, a prep folder of the asset bundle, and one or more read-outs (input set, arms, sets, protocols; one training carries all of them: `readout_1`, `roll_1_<protocol>`, `readout_2`, ...) (section 6) |
| `decode_rollout` | `decode` (GPU: `decode run`) -> `jobs` (`decode make-jobs`, adds canaries) -> `roll_<protocol>`. Without protocols the group stops after the decode | a prior, an input set (and optionally a list of its units), draws and base seed, protocols (section 7) |
| `jobs` | `roll_<protocol>` | a jobs file that came with the batch (canaries already inside) and protocols (section 8) |
| `cem_readout` (r7a) | for every generation: `g<g>_decode` (GPU: `cem decode`) -> `roll_g<g>_<protocol>` (CPU) -> `g<g>_update`; then `constants` -> `readout` (GPU: `dsrl readout --arms const,...`) -> `roll_<protocol>`. Resumes at the generation it stopped in | a search config (`configs/cem.json`), the search units, a seed, and one or more read-outs (section 10.7) |
| `nn_readout` (r7a) | `readout` (GPU: `dsrl readout --arms file`, no training) -> `roll_<protocol>` | a noise file of the batch (one noise per test input), a prior, and the held-out ids with their sha256 |

### 5.2 Running a batch

```bash
PY=$GPUENV/bin/python; W=$RELEASE/code/wf.py
$PY $W batch run    $BATCH --out $RESULTS --cpu-python $ENV/bin/python --dry          # the plan, step by step with the commands; runs nothing
$PY $W batch run    $BATCH --out $RESULTS --gpus 0,1,2,3 --cpu-python $ENV/bin/python # one machine: one group per GPU slot at a time
$PY $W batch slurm  $BATCH --out $RESULTS --partition P --gpus-per-task 1 --cpus 8 --time 24:00:00 --gpu-python $GPUENV/bin/python --cpu-python $ENV/bin/python   # writes $RESULTS/run_batch.sbatch
$PY $W batch shell  $BATCH --out $RESULTS --gpus 0,1,2,3 --gpu-python $GPUENV/bin/python --cpu-python $ENV/bin/python                                          # writes $RESULTS/run_batch.sh, a plain loop
$PY $W batch status $BATCH --out $RESULTS [--array]       # which groups are done; --array prints the group numbers still to run, for `sbatch --array=`
$PY $W batch pack   $BATCH --out $RESULTS --note "which machine"   # the archive to send back
```

- **One GPU per job.** `--gpus 0,1,2,3` makes four slots; each slot runs one group at a time and every GPU step of it gets `CUDA_VISIBLE_DEVICES=<slot id>`. A repeated id (`--gpus 0,0`) is two slots on one card (memory permitting). Without `--gpus` there is one slot and the
  environment's `CUDA_VISIBLE_DEVICES` is left alone (SLURM sets it). `--gpu-prefix "CMD {gpu} --"` puts a command in front of every GPU step (a cluster whose GPUs are guarded or shared); `--nice N` sets the nice level (default 19).
- **SLURM**: `batch slurm` writes an array script with one task per group (`--array=0-N-1`, `--array-parallel` limits how many run at once, `--setup 'module load ...'` adds lines, `--gres gpu:a100:1` replaces `--gpus-per-task`, `--account/--qos/--constraint/--mem` as needed).
  Read the `#SBATCH` lines before you submit. `--dry` prints the script and writes nothing. **Without SLURM**: `batch shell` writes a loop with one background worker per GPU.
- **CPU workers.** The rollout steps use `--workers` processes (default: `$SLURM_CPUS_PER_TASK`, else the cores divided by the GPU slots) of the CPU environment (`--cpu-python`): a rollout needs 0.4 to 0.6 GB and one core.
- **Resume.** Run the same command again after any interruption (a time limit, a node failure, Ctrl-C): a step with a valid `STAMP.json` is skipped, the unfinished step's folder is moved to `RESULTS/_failed/<group>__<step>__<time>/` (kept as evidence) and the step runs again. A group is held by `RESULTS/<group>/RUNNING.lock` while it runs; the SLURM script passes `--force-unlock` because its array task owns its group.

### 5.3 What a run writes

```text
RESULTS/<group>/decode/ | train/ | readout/        the outputs of the wf commands of the step (README sections 6 and 7)
RESULTS/<group>/jobs/jobs.npz (+ .canaries.json)   decode groups: the rollout jobs with their canaries
RESULTS/<group>/roll_<protocol>/results.npz        the rollout results (section 8), results.run.json beside it
RESULTS/<group>/<step>/STAMP.json                  written when the step has finished: batch id, release digest, command, GPU, start and end, host fingerprint, sha256 of every output
RESULTS/<group>/logs/<step>.out|.err|.failed.json  logs; a failed step leaves its error here
RESULTS/_failed/                                   unfinished step folders set aside by a resume
```

**`batch pack`** checks that every group is done (`--partial` packs the finished ones), links the files we need into a new folder `RESULTS_pack/` (everything but the 600 MB `endpoint.pt` of a training and files above `--max-file-mb`), runs `pack-results` on it (section 9.3) and writes `RESULTS_pack.tar` and its sha256:
upload that one archive to `returns/<your name>/<date>/` and message us its name and sha256. **Before the first batch** send us `host_fingerprint.json` and the verify reports (SETUP.md step 8): `wf.py fingerprint --out FILE` writes the fingerprint (CPU model and flags, GPU model and driver, versions, the numerical class of your CPU).

## 6. Job type 1: GPU DSRL-NA training and read-out

**Data:** a *prep folder* from the asset bundle (the dumbbell census with one prior each: `prep_ex299_hug1`, `prep_ex299_dumbbell50k`, `prep_ex299_hug2`, whose `states.npz` holds the states of all 220 census views so that a read-out reaches the 164 held-out views (`reference/census/dumbbell_census220_readout_sets.json`, `--sets 1`); the HUG-Bench objects: `prep_ex315_hug2`, `prep_ex324_a`, `prep_ex324_b`): `manifest.json`, `states.npz`,
`actions.npz`, `pool.npz`. Building a new prep (decoding thousands of pool draws) is not part of this release. A config (`configs/dsrl_*.json`) is the
authors' settings: utd 20, batch 256, three 2048-wide Tanh layers with LayerNorm, two critics, noise-critic steps 10, bound 1.0.

```bash
PY=$ENV/bin/python; G=$GPUENV/bin/python; W=$RELEASE/code/wf.py; export CUDA_VISIBLE_DEVICES=0
# train: 1,000 calls of the authors' train(), then the actor's noise at every state
$G $W dsrl train --config $RELEASE/configs/dsrl_ex324_a.json --seed 1 --out /path/dsrl_ex324_a_s1
# read out: decode the actor's noise at the inputs of an input set -> rollout jobs; the training run's config.json (its seed included) is read from --train-out.
# Our experiments read out all 1,109 inputs for own / stoch but only the 909 unseen ones (readout sets 1 and 2) for normrand, so two read-outs (the default arms are own,stoch,normrand;
# the default --protocols mano-1111-p1 is the protocol the canaries are made for):
$G $W dsrl readout --train-out /path/dsrl_ex324_a_s1 --inputs $ASSETS/inputs/input_sets/ex315_hug2_1109.json --arms own,stoch --out /path/dsrl_ex324_a_s1_readout
$G $W dsrl readout --train-out /path/dsrl_ex324_a_s1 --inputs $ASSETS/inputs/input_sets/ex315_hug2_1109.json --arms normrand --sets 1,2 --out /path/dsrl_ex324_a_s1_readout_normrand
$PY $W rollout run --protocol mano-1111-p1 --jobs /path/dsrl_ex324_a_s1_readout/jobs_readout.npz --out /path/results/ex324_a_s1_readout.npz --workers 8
$PY $W rollout run --protocol mano-1111-p1 --jobs /path/dsrl_ex324_a_s1_readout_normrand/jobs_readout.npz --out /path/results/ex324_a_s1_readout_normrand.npz --workers 8
```

A new experiment of this type is a config (any values of the keys in `configs/dsrl_*.json`), a seed, a prep folder of the bundle and an input set: data (section 9.1).

**Output of `train`** (new folder): `na/endpoint.pt` (networks, optimisers, RNG states), `na/train_log.jsonl`, `na/diagnostics.json`,
`na/readout_noise.npz` (deterministic and stochastic noise per state), `config.json`, `train_run.json`. **Output of `readout`:** `readout_<arm>.npz` (decoded grasps,
Q^W and Q^A values) and `jobs_readout.npz`, keys `<arm>/<input>`, ready for job type 3. The arms: `own` (the actor's deterministic noise), `stoch` (one sample
of its Gaussian), `normrand` (a random direction rescaled to the actor's own noise norm, the project's control).

**Expected time:** 1,000 calls took 629 to 963 s on an RTX A6000 shared by three runs (741 inputs, 23,712 labelled executions) and 1,716 s on an RTX 3080 shared by three
runs (56 views); on a card of your own, expect 5 to 15 minutes (the 20-call check of level L3 takes 6.6 s and peaks at 0.84 GB of card memory). The read-out decodes every input of the input set for every arm (1,109 inputs: a few minutes; the unseen-inputs `normrand` call above: 909).
The training pool (`pool.npz`, up to 2.4 GB) is loaded onto the card.

## 7. Job type 2: GPU prior decode

A **prior** is one of `hug2` (the multimodal HUG model, 10 Euler steps), `hug1` (the public single-mode model, 50 steps) or `dumbbell50k` (our fine-tune
of HUG2, 10 steps). For every input picture the decoder draws `--draws` grasps from fixed noise rows: row i is `torch.randn((1, 99))` from a CPU
generator seeded with `derived_seed(base_seed, "fresh_hug_benchmark_noise", noise_key, i)`, so the same noise serves every input and prior (the
"codebook"). It needs CUDA.

```bash
PY=$ENV/bin/python; G=$GPUENV/bin/python; W=$RELEASE/code/wf.py        # PY: the CPU environment (rollouts), G: the GPU environment (decode, DSRL)
export CUDA_VISIBLE_DEVICES=0
# the dumbbell census views, HUG2, 32 draws: pictures -> grasps
$G $W decode run --prior hug2 --inputs $ASSETS/inputs/input_sets/dumbbell_census220.json --draws 32 --base-seed 20261116 --out /path/dec_hug2
# grasps -> rollout jobs of job type 3, then roll them
$G $W decode make-jobs --decoded /path/dec_hug2 --inputs $ASSETS/inputs/input_sets/dumbbell_census220.json --protocols mano-1111-p2 --out /path/dec_hug2_jobs.npz
$PY $W rollout run --protocol mano-1111-p2 --jobs /path/dec_hug2_jobs.npz --out /path/results/hug2_p2.npz --workers 8
```

`make-jobs` adds the known-answer jobs (canaries, section 9.2) for the protocols you name in `--protocols` (default: the six MANO protocols; add a robot-hand protocol if you will roll one) and writes
`<out>.canaries.json` beside the jobs file. A new experiment of this type is a new input set, store and prior: data (section 9.1).

**Input set** (`wf-input-set-1` json): units with `unit`, `cluster`, `unit_key` (the picture in the input store), `asset`, `noise_key`. A picture is an npz
with `rgb` (480x640x3 uint8), `depth_mm` (uint16), `camera_k`, `condition_uv` (the clicked pixel on the object), and, for rollouts, `T_base_camera` and
`settled_object_qpos`. Pictures are stored once and read back, never re-rendered: the anti-aliased colour render is not bit-stable across GPUs.
Use `--store DIR` for a store other than `$ASSETS/inputs/input_store` (the reference pictures in `reference/decode/` are an example).

**Output** (`--out`, new folder): `preds_<prior>.npz` (per unit and draw: `T_camera_wrist`, `mano_pose_aa`, `landmarks_camera`, `mano_shape`,
`mano_params99`, `noise`, `finite`, `noise_seed`, `candidate_index`) and `decode.json` (checkpoint sha256, steps, torch, device, counts, seconds).

**Expected speed** (our records, RTX 3080, the 112-view census with 64 draws = 7,168 rows): HUG2 156 s, dumbbell 50k 134 s, HUG1 (50 steps) 223 s; model load 2 to 5 s. On an RTX A6000 (this release's level L2: 6 pictures x 32 draws = 192 rows)
HUG2 takes 2.2 s, dumbbell 50k 1.8 s, HUG1 3.1 s after a model load of about 4 s, and the card's peak memory is 1.6 GB.

## 8. Job type 3: CPU rollouts

**Input:** a jobs file (npz, one row per grasp): `key` (unique), `sim_assets_dir` (object folder relative to `$ASSETS/assets`), `T_base_camera`
(4x4), `T_camera_wrist` (4x4), `finger_aa` (15x3), `settled_qpos7` (object pose at rest), `bench_table_z` (optional), `landmarks_camera`
(21x3, robot-hand protocols only). The full description is at the top of `code/wf_lib/roll.py`; `reference/cpu/jobs_*.npz` are examples. A new experiment of this type is a new jobs file: data, no code change (section 9.1).
A jobs file we send you, or that `wf.py decode make-jobs` / `dsrl readout` makes, holds known-answer jobs (canaries, section 9.2) next to the real ones: roll the file as it is, under every protocol you will report.

```bash
PY=$ENV/bin/python; W=$RELEASE/code/wf.py
$PY $W rollout check  --protocol mano-1111-p2 --jobs my_jobs.npz                       # loads the jobs against the assets, rolls nothing
$PY $W rollout verify --protocol mano-1111-p2                                          # level W1: digests of the evaluator, MANO, code, assets
nice -n 19 $PY $W rollout run --protocol mano-1111-p2 --jobs my_jobs.npz --out /path/results/my_p2.npz --workers 8 > /path/results/my_p2.run.json
$PY $W rollout compare --results /path/results/my_p2.npz --expected expected.npz        # only when you have a stored reference
```

`run` refuses to start unless W1 passes, never overwrites its output, keeps MuJoCo's own log out of the release folder (the count of its
warnings is in `run_info`: none for MANO, one per job for the Wuji models), and runs one single-threaded process per worker. Use as many
workers as you have free cores; a worker needs 0.4 to 0.6 GB.

**Output:** `results.npz`, one row per job in job order: `key`, `success_native` (the label), `success_strict_v1`, `sim_valid`, `stage_code`
(0 invalid_input, 1 sim_invalid, 2 no_prelift_contact, 3 not_lifted, 4 dropped, 5 success), contact counts and masks, penetration depths, object
rise, `n_steps`, `t_rollout_s` (wall time, never compared), the protocol stamp, and `run_info` (host, CPU, versions, workers, timing, memory, load). The
robot-hand protocols add the fit diagnostics (`tipfit_rms_mm`, ...). A job with a non-finite grasp is not run and is recorded as a failure with zero
diagnostics (stage `invalid_input`).

**Expected speed** (rollouts per second; one rollout is one core; more workers scale it up until memory or cores run out):

| Where | Protocol | Workers | Rollouts per second | Source |
|---|---|---|---|---|
| host A (Threadripper PRO 5975WX, shared) | MANO p1 / p2, 200 HUG-Bench grasps | 2 | 5.9 / 6.2 | this release, L1 (200 jobs in 34 s) |
| host A | MANO p1..p2-dzp1, 220 dumbbell grasps | 2 | 4.1 to 6.5 | this release, L1 |
| host B (Ryzen 9 3900X) | MANO p1, 2,000 HUG-Bench grasps | 6 | 11.1 | our gate record (181 s) |
| host A | MANO p1, dumbbell census, 21,120 grasps | 4 | about 7 (0.57 worker-seconds per rollout: 21,120 rollouts in about 50 minutes) | EX330 measurement of these very jobs |
| host C (EPYC 7542) | Wuji 1, HUG-Bench objects / dumbbell | 4 | 12.9 to 14.1 / 8.1 to 9.6 | stored EX318 / EX305, EX307, EX308 runs |
| host A | Sharpa / Wuji 1 hr, 200 grasps (fit included) | 4 | 5.6 to 6.5 | EX327 gate record |

### Our production batches: reproducing our numbers

The asset part `batches` holds the jobs files of our experiments and our stored results for them (`$ASSETS/batches/<batch>/`, with `INDEX.json`). Run a batch with the
protocol in the table and compare with the stored file; on AMD Zen 2 and Zen 3 CPUs the labels are equal and almost every field is bit-equal (the Wuji 1 batches were rolled on host C, the others on host A;
re-rolled on host A, 799 of 800 spot-checked Wuji jobs match in every field and one differs by less than 1e-4 mm in six diagnostics: section 10).

| Batch | Grasps | Protocol(s) | Our stored time |
|---|---:|---|---|
| `ex330_ex331_census220`: dumbbell census, MANO hand, 3 priors x 220 views x 32 samples | 21,120 per protocol | the six MANO protocols | 450 to 820 s with 10 to 16 workers on host A (about 50 min at 4 workers) |
| `ex305_wuji_dumbbell`: Wuji 1, 3 priors x 112 views x 32 | 10,752 | `wuji1-float-p1` | 1,122 s at 4 workers on host C |
| `ex307_wuji_dumbbell_new_views`: Wuji 1, the other 108 views | 10,368 (2 parts) | `wuji1-float-p1` | about 550 s per part at 4 workers |
| `ex308_wuji_steering_readout`: Wuji 1, DSRL-NA read-out grasps | 2,016 | `wuji1-float-p1` | 248 s at 4 workers |
| `ex318_wuji_multiobject`: Wuji 1, multi-object steering and HUG2 census | 31,815 (4 parts) | `wuji1-float-p1` | 565 to 619 s per part at 4 workers (13.7 rollouts per second) |
| `ex327_hands_pop9216`: Sharpa Wave and Wuji 1 hr, p1 and p2; MANO p2 | 9,216 per protocol | `sharpa-float-p1/p2`, `wuji1hr-float-p1/p2`, `mano-1111-p2` | 1,286 to 1,974 s per hand protocol at 4 workers |

```bash
B=$ASSETS/batches/ex318_wuji_multiobject; S=/path/slice
# spot check first: 200 rows spread over the batch, rolled and compared with our stored results (minutes)
$PY $W rollout slice --jobs $B/ex318_jobs_part1.npz --expected $B/expected_part1_wuji1-float-p1.npz --n 200 --out-dir $S
$PY $W rollout run --protocol wuji1-float-p1 --jobs $S/jobs_slice.npz --out /path/slice_result.npz --workers 8
$PY $W rollout compare --results /path/slice_result.npz --expected $S/expected_slice.npz
# then the whole part, and the comparison with our stored file
nice -n 19 $PY $W rollout run --protocol wuji1-float-p1 --jobs $B/ex318_jobs_part1.npz --out /path/results/ex318_part1.npz --workers 8 > /path/results/ex318_part1.run.json
$PY $W rollout compare --results /path/results/ex318_part1.npz --expected $B/expected_part1_wuji1-float-p1.npz
```

## 9. Sending results back, and how we check them

**You can only send us numbers, so the numbers carry their own evidence.** Nothing you send is code and nothing you send is run on our side except by our own tools.

### 9.1 What a new experiment can be, and what needs a new release

| Without any code change (the release you have stays as it is; we send you data) | Needs a new release (we push it to the repository and you pull it; nothing comes back as code) |
|---|---|
| Job type 3: any jobs file (section 8) under any protocol of the table in section 1 | A new protocol: a changed evaluator, wrist rule, hand model or scene (protocols are digest-pinned) |
| Job type 2: any input set (json) with its pictures in an input store, any prior of the three, any number of draws and base seed (section 7) | A new method or prior: another learner, decoder, retargeter, reward or score; code of any kind |
| Job type 1: any DSRL config (the keys of `configs/dsrl_*.json`), seed and input set, on any prep folder of the bundle (section 6) | New objects, pictures or prep folders: they are asset parts, a new tarball whose manifests are part of the release digest |

**Added in revision r4 (data on top of r3, no code change: the evaluator, the protocols and their digests are untouched):** the asset part `prep_ex338_a_p2` (prep `ex324_a` with the labels of the same stored grasps under `mano-1111-p2`: `prep/ex338_a_p2/`, its `states.npz` and `pool.npz` are byte for byte those of `prep_ex324_a`), the input set `inputs/input_sets/ex315_hug2_1109_codebook.json` (the 1,109 units of `ex315_hug2_1109.json` with `noise_key` "codebook" for every unit, so that a decode draws the same noises for every input) in the `inputs` part, and three DSRL configs: `configs/dsrl_ex338_b_p2.json`, `dsrl_ex338_c_p2.json` (the 200 training inputs of EX315 as `gate_train_views`) and `dsrl_ex339_b_p2.json` (341 training inputs). A DSRL config key that the loader does not read (`label_protocol`, `training_inputs`, `note`) is descriptive only.

**Already on the list for the next release (planned, not built in this one):**
- two more DSRL read-out arms for the experiments that need the trained scorers, since checkpoints do not come back: `qapick32` (the trained Q^A scores the 32 decoded census candidates of each input; the pack returns the picked key and its stored label, plus the 32 scores)
  and `qwpick32` (the trained Q^W scores the 32 census noises; the same return);
- DSRL config keys that the loader does not take yet: the learning rate and `tau` (r7a takes `noise_source` `fresh` and `state_mode` `full`, section 10.7);
- a protocol part for the Wuji Hand 2.2 (`wuji22-float-p1` and `wuji22-float-p2`): its evaluator code, two hand-model asset folders and two protocol files, all digest-pinned (a new protocol: a new release).

### 9.2 Known-answer jobs (canaries)
`wf.py decode make-jobs`, `wf.py dsrl readout` and our own batch tool add **canaries** to every jobs file: rows copied from our own batches (`batches/INDEX.json`, the dumbbell census) and from the HUG-Bench reference files of this release (`reference/cpu`) whose results under a protocol we stored. The rule:
at least 20 per file and at least 1 % of the real jobs, at least one for every (protocol, prior, object set) cell: the protocols you named with `--protocols`, the priors of the jobs' keys, and the object set of the jobs themselves (the dumbbell, or HUG-Bench objects: read from each job's `sim_assets_dir`),
so jobs on HUG-Bench objects get HUG-Bench canaries and dumbbell jobs dumbbell canaries; tagged `__canary__/<protocol>/<batch>/<the row's own key>` in the key column,
at random places among the real jobs; `<jobs file>.canaries.json` lists them (no answers in it). Roll the file as it is: the canaries cost about 1 % of the rollouts.
`wf.py pack-results` refuses to pack results whose jobs file is missing, whose keys are not exactly the jobs file's keys, or in which a canary is missing, and it refuses a jobs file without canaries unless it is one of our own reference or batch jobs files
(`--no-canaries` packs anyway and records the waiver: our verifier rejects that pack).

### 9.3 The pack
**`wf.py batch pack` (section 5.3) does all of this for a batch you ran with `wf.py batch run`.** The commands below are for results you made by hand with the single commands of sections 6 to 8.
```bash
NAME=yourname; DAY=$(date +%Y%m%d)                 # the name we know you by, and the upload date
mkdir /path/results_to_send && cp -r <your results npz, .run.json, decode / read-out / training folders> /path/results_to_send/
$PY $RELEASE/code/wf.py pack-results --folder /path/results_to_send --note "what this is, which machine"
cd /path && tar -cf results_${NAME}_${DAY}.tar results_to_send && sha256sum results_${NAME}_${DAY}.tar
```
`pack-results` checks itself (9.2) and then writes into the folder: `PACK.json` (the release: name, git commit, SHA256SUMS digest, `assets.json` and every asset manifest's sha256; the machine that packs: CPU model, vendor, family, model, stepping, microcode and flags,
operating system, kernel, Python and package versions, GPU models and driver; and for every file its sha256 and, for a results file, its protocol and protocol digest, the sha256 of its jobs file, the job count, the canaries declared and present, start and end times,
workers, the release digest and asset manifests it ran with and the CPU that ran it), `jobs/` (a copy of every jobs file with its canary record, found by its sha256), `RETURN.json` and `SHA256SUMS` over everything. Every results file also holds all of this in its `run_info`,
and a digest of its own arrays (`results_digest`), so an edit after the run shows. `tar` makes one file (no compression: the results are already compressed).
**Upload that one archive to `returns/<your name>/<date>/` (date as `YYYYMMDD`) in the drive folder shared with you for returns** (create the two sub-folders if they are missing; one date folder per upload)
**and message us the archive's name and the sha256 it printed.** Keep every comparison on one host: never split a batch between machines and do not pool results of different CPU families or GPU models (a pack that does is rejected, section 9.4; on a cluster with mixed node types pin one type with `--constraint` or `--gres gpu:<type>:1`).

### 9.4 How we check a pack
We unpack it and, with the batch we issued (private) and our stored answers, check: **integrity** (every file against `SHA256SUMS` and `PACK.json`); **identity** (the release digest, `assets.json`, every asset manifest and the git commit are the ones we issued; each results file ran with them
under the protocol we issued, whose digest is this release's); **completeness** (every issued job file and protocol returned once, the keys exactly the issued keys, no result edited after its run); **canaries** (each canary's labels and every record field equal our stored answer: bit for bit on the AMD Zen 2 and Zen 3 CPUs
this release was tested on, within the `pass_near` tolerance of section 4 on any other CPU, labels always equal); **sanity** (no non-finite value, no mass of rows that never simulated, no constant success, no mass of identical records for different grasps);
**step stamps** (a pack from `batch pack` carries BATCH.json and a `STAMP.json` for every step of every group: each must say ok, this batch and this release digest, and the files must be the files stamped); **one machine type** (all rollouts of a pack on one CPU class, where every AMD Zen 2 and Zen 3 part counts as the same class and any other CPU is its own by vendor, family and model, and all decode, training and read-out steps on one GPU model, read from the step stamps and from the run record of every results file: a pack that mixes machines is rejected, not only noted); **reference checks** for what canaries cannot cover (the decoded grasps of a decode group: at least 99.5 % of the rows within 1e-3 of our stored decode of the same key (or of our own re-decode of picked units) and every row within 0.1 (a few rows near a branch of the flow differ more on another GPU: 2 of 7,040 rows of the 50k census decode, up to 4.5e-2); a DSRL training log equal, every field but `secs`, to our stored or re-run log of the same config and seed: bit for bit on the same GPU model, within 2 % on the first 100 calls on another);
and a **spot check**: we re-roll k random real jobs per file and protocol ourselves (k = the larger of 10 and the square root of the job count, doubled if a canary or the success rate raises doubt, chosen by a seed you cannot know) and compare.
Any failure of the checks before the spot check rejects the whole pack, with the reason. Protocols whose labels are almost equal (mano-1111-p1 and mano-1111-p2s-dzm1 agree on 99.99 %) are told apart by the protocol digest of every results file and by the record fields of the canaries, not by labels. If they pass, the verdict is *needs spot check* until our re-run agrees, then *accepted*. **Reproduced is a separate verdict from accepted.** For every cell of a batch that repeats one of our experiments we also report the rate difference in points and the number of flipped labels against our stored labels of the same keys: a cell is reproduced if the rate differs by at most 0.31 points and at most 2 % of the labels flip, close up to 1.0 points and 5 %, and a replay of our own jobs is reproduced only if no label flips (DSRL cells: equal counts, close within 6 of 268 inputs); the pack is reproduced only when every cell is, whatever the decode tolerance above allows.
A host whose canaries differ in more than the tolerance is not a numerical host we can use:
we will ask you to run on an AMD Zen 2 / Zen 3 machine or to send us the host for qualification.

## 10. What was checked where (2026-11-20)

All on our reference host A (AMD Ryzen Threadripper PRO 5975WX, Ubuntu 24.04, NVIDIA driver 570, GPU 0 an RTX A6000), at `nice -n 19` with two CPU workers, in virtual environments built from
`requirements/*.txt`, from the two asset sets exactly as they are delivered (tarballs, unpacked with the commands of section 3.0, the MANO file placed by hand). The full record (every check, the commands, the findings) is [docs/SMOKE_TEST.md](docs/SMOKE_TEST.md).

| Check | Result |
|---|---|
| Release built from our repository commit recorded in `RELEASE_SOURCES.json`; every shipped physics file byte-identical to the code that produced our records | yes (the protocol digests are unchanged, so result stamps match ours) |
| The tarballs of both sets against their `SHA256SUMS`, unpacked into empty asset roots, then `sha256sum -c` of the unpacked files (rollout-only set: 4 tarballs, 1,846 files; full set: 13 tarballs, 4,580 files) | pass |
| MANO file missing, wrong, right (`assets verify`, `verify`, `rollout run`, `decode run`, `dsrl readout`) | missing or wrong: a message that says where to get the file and which sha256 it must have, exit 1 before any work starts; right: pass |
| L0 + L1 + L1b + L1h on the rollout-only asset root, CPython 3.10.20 | `VERIFY PASS`, 41 checks in 1,211 s: the 200 HUG-Bench grasps under p1 and p2 and the 220 dumbbell grasps under six MANO protocols bit for bit; the robot-hand protocols bit for bit (Sharpa Wave and Wuji 1 hr under p1 and p2, the Wuji 1 October evaluator); 18 of 19 production-batch spot checks bit for bit and one `pass_near` (one Wuji 1 job of 800 differs from host C's stored result by at most 8.6e-5 mm in six diagnostics) |
| L0 + L1, CPython 3.12.3 | `VERIFY PASS`, 11 checks in 341 s, bit for bit: the same numbers as under 3.10 |
| L0 over all 14 parts of the full asset root | `VERIFY PASS`, 15 checks in 3.5 s |
| L0 + L2 + L3, RTX A6000 | `VERIFY PASS`, 12 checks in 35 s: decode within 8.2e-6 of the stored RTX 3080 decode (HUG2, HUG1, dumbbell 50k); the port's exactness tests bit for bit; the 20-call training reproduces its record |
| GPU time and memory (A6000) | decode of 192 rows 1.7 to 3.1 s warm (first run) and 8.4 s from a cold start (4.1 s of it loading the model), peak 1.6 GiB; 20 DSRL calls 6.6 s, peak 0.84 GB (first run); a three-call load test of each of the four released prep folders 5.9 to 26.9 s, peak card memory 1.0 to 3.4 GiB |
| The three job types end to end | decode -> jobs -> rollout (192 grasps); DSRL train (seed 3) -> read-out (seed 3 recorded) -> rollout (18 grasps); `rollout slice`; `pack-results`, `tar`, unpack and verify (section 9) |
| Public pieces only, plus your own MANO file (section 2) | `fetch-hug-source`, `fetch-dinov2`, `fetch-hug1` into an empty asset root (sha256 matched), then `decode run --prior hug1` on the six reference pictures: 48 grasps, all finite, 1.6 s |
| Not tested | the real download of the MANO archive from the Max Planck Institute (it needs a registered account; the stand-in was the same file, sha256 checked), the drive link (the tarballs were copied; the GitHub clone was tested with the GitHub CLI at revision r1 and, for this revision, from a local copy), a SLURM cluster (the array script is generated and checked with `bash -n`, a SLURM array task is simulated with `--dry`), a CPU outside the AMD Zen family (an Intel Xeon reproduced the labels but not every diagnostic: expect `labels_only`), GPUs other than an RTX 3080 and an RTX A6000 (in particular a 1,000-call DSRL-NA training on any GPU model but the RTX A6000), and the full Wuji production batches |

### 10.1 The control run on a cluster and the SETUP test (2026-11-21)

A second session played a user of the release on host A (AMD Ryzen Threadripper PRO 5975WX, 64 logical cores, three RTX A6000; the other cards were in use by other work, so GPU 0 was used behind our GPU guard and at most 12 CPU workers). It started from the GitHub repository (cloned with the GitHub CLI at the published commit, while the repository was still private) and the 13 asset tarballs, built fresh environments from `requirements/*.txt`, and then reproduced a registered experiment, with the README text as the only instruction. Every number below is read from the files of that run (the criterion was written before it ran: `runs/release_repro_20261120/CRITERION.md` in our project repository).

**A. The dumbbell census table (EX331), own decode.** Decode with HUG2, HUG1 and the dumbbell 50k on the RTX A6000, rollouts under five wrist conditions; the table is the success count of the 164 held-out views x 32 draws = 5,248 per cell. Each cell: registered count (ours) / the control's count (difference; labels that differ from our stored label of the same key). Rule: reproduced if the count differs by at most 16 and at most 105 labels flip; close up to 52 and 262.

| condition | 50k | HUG1 | HUG2 |
|---|---|---|---|
| mano-1111-p1 | 3,930 / 3,938 (+8; 20 flips; reproduced) | 3,906 / 3,900 (-6; 14 flips; reproduced) | 3,669 / 3,678 (+9; 11 flips; reproduced) |
| -2 sags (-p2-dzm2) | 3,045 / 3,038 (-7; 29 flips; reproduced) | 4,206 / 4,202 (-4; 14 flips; reproduced) | 4,003 / 4,003 (+0; 24 flips; reproduced) |
| -1 sag (-p2-dzm1) | 3,961 / 3,964 (+3; 21 flips; reproduced) | 3,888 / 3,894 (+6; 16 flips; reproduced) | 3,680 / 3,684 (+4; 16 flips; reproduced) |
| commanded (mano-1111-p2) | 4,522 / 4,529 (+7; 23 flips; reproduced) | 3,436 / 3,433 (-3; 5 flips; reproduced) | 3,136 / 3,128 (-8; 14 flips; reproduced) |
| +1 sag (-p2-dzp1) | 4,316 / 4,308 (-8; 26 flips; reproduced) | 2,722 / 2,722 (+0; 6 flips; reproduced) | 2,317 / 2,313 (-4; 16 flips; reproduced) |

15 of 15 cells finished: 15 reproduced, 0 close, 0 failed; 255 of 78,720 held-out labels differ from our stored labels in total (0.32 %). The registered signs hold: True (HUG1 and HUG2 do better one sag lower than at the commanded offset, and still better two sags lower) and the 50k is best at the commanded offset: True.

Decode against our stored census decode (made on host B's RTX 3080; the control's on an RTX A6000; the same checkpoints, sha256 prefixes 50k e40c61a1, HUG1 515b5c3b, HUG2 ca33bff2), over all 7,040 rows per prior:

| prior | largest difference (wrist pose / finger angles / landmarks) | decode time | peak card memory |
|---|---|---|---|
| 50k | 0.045 / 0.016 / 0.0071 | 330 s | 1.8 GiB |
| HUG1 | 2.6e-05 / 1.5e-05 / 1.5e-06 | 260 s | 1.8 GiB |
| HUG2 | 3.2e-05 / 2.2e-05 / 3e-06 | 581 s | 1.8 GiB |

The differences are float rounding (the same GPU twice gives identical decodes), above the 8.2e-6 of the six-picture check because 7,040 rows have a longer tail; the dumbbell 50k has 7,033 rows within 1e-4, 5 more within 1e-3 and two beyond 1e-3 (`l_a300#17`, `y090_a075#30`; one of them in the held-out views), samples near a branch of the flow, while HUG1 and HUG2 have every row within 1e-4 (`runs/release_repro_20261120/stage1/decode_rows.txt`). The decode times were measured on a card shared with a DSRL training and ten rollout workers.

**A2. Replay of our stored jobs.** The 21,120 stored census jobs rolled under `mano-1111-p2` with the release's own tools: 21,120 of 21,120 labels equal, 0 rows differ in any field (status `pass_bitwise`), 776 s with 10 workers (27 rollouts per second).

**B. One DSRL-NA row end to end** (EX324 row a, seed 1: `dsrl train` for 1,000 calls on the RTX A6000, `dsrl readout`, rollouts under `mano-1111-p1`).

| arm | inputs | registered (ours) | the control | difference |
|---|---|---|---|---|
| own | 268 unseen-object inputs | 186 / 268 | 186 / 268 | +0 |
| stoch | 268 unseen-object inputs | 187 / 268 | 187 / 268 | +0 |
| normrand | 268 unseen-object inputs | 152 / 268 | 152 / 268 | +0 |
| own | 100 unseen-object inputs | 70 / 100 | 70 / 100 | +0 |
| stoch | 100 unseen-object inputs | 70 / 100 | 70 / 100 | +0 |
| normrand | 100 unseen-object inputs | 62 / 100 | 62 / 100 | +0 |

Training: 41 of 41 log lines equal to the stored run's in every field but the time (690 s against 628 s), 1,076 of 1,076 tensors of the final checkpoint equal, the read-out noise equal in every array. Read-out and rollouts: 3,127 of 3,127 per-input labels equal to the stored records (the numeric fields agree to float32 rounding in 2,656 records: the decoded grasps differ from the stored ones by at most 6.6e-7, which moves the contact diagnostics of the others but no label).

**SETUP.md, line by line.** The guide was followed in a fresh folder with the commands as written (the clone from a local copy of the repository instead of github.com, the drive and MANO downloads replaced by copies of the same files, the GPU commands behind our GPU guard, `--gpus 0` instead of four GPUs, `--workers 2`, the SLURM script checked with `bash -n` because this machine has no SLURM): measured: step 1 clone and check under a second, step 2 CPU environment 50 s and GPU environment 329 s, step 3 tarball check 4 s, step 4 unpack and check 9 s, step 5 MANO under a second, step 6 links and HUG source 3 s, step 7 CPU verify 200 s and GPU verify 32 s, step 9 the three-group test batch 46 s; all 16 bash blocks of SETUP.md ran as written.

### 10.2 Revision r3: the dumbbell preps (2026-11-21)

Revision r3 (this one) adds to the batch runner, the known-answer jobs and the SETUP guide described in section 10.1 two asset parts, `prep_ex299_dumbbell50k` and `prep_ex299_hug2` (DSRL training data of the dumbbell census for the 50k prior and for HUG2: EX299 rows 0c and 5), their configs `configs/dsrl_ex299_dumbbell50k.json` and `dsrl_ex299_hug2.json`,
and a 220-view `states.npz` in all three dumbbell preps (the released HUG1 prep had the states of 112 of the 220 census views, so a read-out stopped at the 108 new views). Checked on host A with the new parts laid over the asset root of the control run (`runs/release_repro_20261120/runner_tests/m2_validate.out`):
the manifests of the three parts against their files pass (`wf.py verify --levels L0`); a 20-call training through each config on the RTX A6000 has a call-0 log line equal in every field (relative difference 0) to the registered EX299 log of the same row and seed (rows 4, 0c, 5, seed 1, trained on host B's RTX 3080);
a read-out at the 164 held-out views (`reference/census/dumbbell_census220_readout_sets.json`, `--sets 1`) runs for each of the three. The 112 states of the old file are unchanged in the new one (maximum difference 0.0, EX309's check, repeated), and the loader reads only `ids` and `states`:
a 20-call training on the released HUG1 prep and on a copy with the 220-id file give identical log lines (`runner_tests/states220_compare.txt`). The 50k prep's own manifest lacked the `pass: true` flag the loader requires; the staged manifest adds it and says so in `released_from.changes`.

### 10.3 The work order for an agent, followed literally (2026-11-21)

A fresh session followed the agent work order of revision r3 (since revision r5 it is `QUICKSTART_AGENT.md`; revision r6 adds one command, `assets fetch-hug1`, and changes the release tag) on host A in an empty folder: a script extracted its 21 bash blocks and ran them in order (the block that starts the batch twice: the second run is the resume after an interruption), and checked every `expect` line of the document against the output of the block before it. 22 block runs finished and 30 of 30 expected lines matched. Only these things differed from the text: the clone came from a local copy of the repository instead of github.com; the drive download, the MANO download and the batch archive were stand-ins (hard links of the 15 delivered tarballs, the MANO zip rebuilt from the registered file, and a three-group stand-in batch issued with the same tool and named like batch 1a: two 20-call DSRL trainings and one decode with its rollouts); `python3.10` is a link to a CPython 3.10.20 (host A has no `python3.10` on its PATH); the GPU steps ran behind our local GPU-sharing wrapper with `--gpus 0 --workers 2`; the first run of the batch was stopped with SIGTERM while it worked and the same block was run again (the resume the document describes); `sbatch` and `squeue` were replaced by `bash -n` and an echo (no SLURM on this machine). 6 blocks carried a substitution; each is in `runs/release_repro_20261120/agent_test/final/*.substitutions.txt`.

| Section (blocks) | Measured on host A |
|---|---|
| 1 machine check; 2 variables | 4 s + 0 s |
| 3 clone and check; 4 assets check and unpack; 5 MANO | 0 s + 4 s + 10 s + 0 s |
| 6 CPU environment, GPU environment, hand models and HUG source | 59 s + 314 s + 2 s |
| 7 verify CPU (L0, L1, 4 workers) and GPU (L0, L2, L3) | 212 s: `VERIFY PASS (11 checks, 212.1 s)`; 34 s: `VERIFY PASS (12 checks, 33.5 s)` |
| 8 fingerprint; 9 first report | 1 s + 0 s |
| 10 unpack the batch and dry run; 11 run (way A: the first run stopped by SIGTERM during the decode, then the same block again) | 0 s + 28 s + resume 20 s |
| 11 watch; completeness; 12 pack; 13 second report | 30 s; 0 s; 1 s; 0 s |

A separate test killed the run hard (SIGKILL of the whole process group of the runner and its steps) in the middle of the decode of group D1 and ran the same command again: the stale lock was removed, the unfinished step folder was moved to `_failed/` and the group finished; the rollout results of that run and of the SIGTERM run above are bit-identical to an uninterrupted run of the same group (24 of 24 real rows, 29 fields; `runs/release_repro_20261120/runner_tests/kill9_and_resume.out`).

Not tested: way B on a SLURM cluster (the array script was generated and checked with `bash -n`), the real downloads from the drive and GitHub, the MANO registration, and the upload.

### 10.4 Revision r4: the p2 prep, the codebook input set and three configs (2026-11-21)

Revision r4 is data on top of r3, for the three batches of 2026-11-27 (EX338: 14 groups, EX339: 3, EX352: 5). Nothing under `code/` changed (the evaluator files, the protocol files and their digests are byte for byte r3's), and 15 of the 16 tarballs are byte for byte r3's:
`inputs.tar` gained one member and `prep_ex338_a_p2.tar` is new. Every added file's sha256 equals the pin registered for it:

| File (below the release or the asset root) | sha256 |
|---|---|
| `configs/dsrl_ex338_b_p2.json` | `590a9a54b59cab4821a0a6c1420dec25e57eb39d72855df5822244fc9cbab32c` |
| `configs/dsrl_ex338_c_p2.json` | `791e9467c6cd4d52b4fb098b2a016111ea66089f4eb92c2bfd550bff831395e0` |
| `configs/dsrl_ex339_b_p2.json` | `86584fc8f7251aafd3b80519e83069e7b6c627a6e1b807a2b943c7241d79d945` |
| `inputs/input_sets/ex315_hug2_1109_codebook.json` (part `inputs`) | `e2d9f0a0331740ed729f92a08b88cfd1c78b44b85d799f6061b9e75890fb6ceb` |
| `prep/ex338_a_p2/manifest.json` (part `prep_ex338_a_p2`; the `prep_manifest_sha256` of all three configs) | `eaa611e1db33133cb50f5d2c9ec1b97e4f86055b2fdd676923179e96b0152bec` |
| `prep/ex338_a_p2/actions.npz` (the only new array data: the labels under `mano-1111-p2`) | `2476722efab19ac01cf8c026588cba480f83caa19d68e24e42aabb848e6dfb66` |
| `prep/ex338_a_p2/states.npz` (equal to `prep/ex324_a/states.npz`) | `dd5548a5adcc7d22ee752afc64c4b564795c92ca4a993e943217db34ab18b06a` |
| `prep/ex338_a_p2/pool.npz` (equal to `prep/ex324_a/pool.npz`) | `edcffdfdaad8bc8fdde4b12cd2080ef8d2c2257d8a2bcb83f943a08f5413f170` |
| `prep/ex338_a_p2/pins.json` (the build's provenance record) | `3444f515bb6ad7b7f7d6de9f301b6ec7c007d677274573a871985820b9b8306e` |

Checked on our side before the release was cut: the codebook input set is `ex315_hug2_1109.json` with `noise_key` "codebook" for every unit and nothing else changed (same 1,109 units in the same order); the prep manifest derives from the released `prep_ex324_a` manifest and names no machine or path;
the new part's `states.npz` and `pool.npz` have the same sha256 as the released `prep_ex324_a` files; the three configs carry no pending pin; the tarballs read back member by member against the manifests. The three configs and the new part reach the batch tools unchanged (the control batch of this revision is kept in our project repository, not here).
Unpacking `prep_ex338_a_p2` after `prep_ex324_a` stores 2.4 GB of pool data twice; that is the price of a self-contained part.

### 10.5 Revision r5: licences and generic guides (2026-11-21)

Revision r5 changes licences and documents only. Nothing under `code/`, `configs/` or `reference/` changed (the evaluator files, the protocol files and their digests are byte for byte r4's), and 15 of the 16 tarballs are byte for byte r4's.

- **Licences.** This project's own code, records and data are MIT ([LICENSE.txt](LICENSE.txt)); HUG2, the 50k fine-tune and the HUG2-derived training sets are MIT with the HUG authors' permission. Every third-party part keeps its original licence, stated per part with the attribution it requires ([THIRD_PARTY.md](THIRD_PARTY.md), [ASSETS.md](ASSETS.md), `ASSET_TERMS.txt`). [LICENSES/HUG-Bench-evaluator-MIT.txt](LICENSES/HUG-Bench-evaluator-MIT.txt) is the MIT licence of the HUG-Bench evaluator, which covers the code copied from it.
- **`licences.tar`** holds two more texts (`Apache-2.0.txt` for DINOv2, `wuji-floating-LICENSE.txt` for this project's parts) and a new index `licences/README.txt` with the attribution; the manifest `asset_manifests/licences.sha256`, `sizes.json` and `SHA256SUMS.all` change with it. The other 15 tarballs and their manifests are unchanged.
- **Guides.** The one-page checklist and the agent work order are [QUICKSTART.md](QUICKSTART.md), [QUICKSTART_zh.md](QUICKSTART_zh.md) and [QUICKSTART_AGENT.md](QUICKSTART_AGENT.md); their commands are those of revision r4 apart from the release tag and the expected commit subject.
- **Batches** issued for revision r4 do not run on r5 (the release digest, `assets.json` and the `licences` manifest changed): they are issued again for r5.

Checked before the release was cut: the code copies and protocol files rebuilt from their source commit equal the shipped ones (36 files); L0 of a fresh clone of the r5 tag against our r5 asset bundle passes; every tarball was read back member by member against its manifest.

### 10.6 Revision r6: no personal or internal names in any shipped file (2026-11-22)

Revision r6 changes wording, metadata and digests only; nothing that a run computes. Every shipped file (code, protocol files, reference records, the tarballs and their members, the side files) was scanned for personal names, internal host or account names and per-person wording; the only names left are the copyright lines the MIT licence texts require and the account names inside the URLs of the public upstream sources the tools download from.

- **Code and protocol files.** Docstrings, comments, the unused default paths of `collab_native.py` and `wuji_eval.py` (every caller passes its own) and the descriptive strings of the protocol files are reworded. Four of the changed files are digest-pinned, so every protocol has a new digest; for each protocol, undoing the re-pinned sha256 values gives the old digest again, and with docstrings removed the programs are identical except those unused default strings. The stored results in `reference/` and in the `batches` part carry the new digests (only their `protocol_digest` field and the digests in `run_info` changed; every record is byte for byte the same).

| Protocol | digest up to r5 | digest from r6 |
|---|---|---|
| `mano-1111-p1` | `6758220ecbd09519` | `854767bca05682bd` |
| `mano-1111-p2` | `a675bc3787479b9e` | `387d90ffe1bdd287` |
| `mano-1111-p2s-dzm1` | `157be7e00e4273d7` | `70534d6fd56da8ce` |
| `mano-1111-p2-dzm1` | `3c6b8fbf99345714` | `ba6a1b8e87ef49f7` |
| `mano-1111-p2-dzm2` | `63887c8b298c2c06` | `17c6b793e51c2524` |
| `mano-1111-p2-dzp1` | `3bc85fcba076a37c` | `598994eb7fae2bd6` |
| `sharpa-float-p1` | `7f44a4737a5420bf` | `9103bf1f41dac163` |
| `sharpa-float-p2` | `a7f6ba0e726e26e9` | `86d80ad632b72732` |
| `wuji1hr-float-p1` | `0813189ea19405e4` | `bbb66c23e105de12` |
| `wuji1hr-float-p2` | `ca5325083e6d82b5` | `6c3f70e3d7c28293` |
| `wuji1-float-p1` | `4ef77035171024b2` | `7dd5fdbe4dd7369a` |

- **Assets.** The HUG1 weights are no longer in the tarballs (`wf.py assets fetch-hug1` downloads the public file and checks its sha256): the public file carries its authors' training paths in its metadata. The HUG2 and dumbbell 50k checkpoints have two training-config strings (`save_dir`, `entity`) replaced by placeholders, so their sha256 is new; every tensor is unchanged (the model the HUG source builds from each file has the same 483 weight tensors). The evaluator copy in `core` leaves out four files no run reads (`README.md`, `.gitignore`, `.gitmodules`, `assets/img/aria2mesh.gif`), and two object files name the MuJoCo conversion of Google Scanned Objects without its account. `licences.tar` and `batches.tar` change with them; the other 10 tarballs are byte for byte r5's. The full set is 15 tarballs, 8.7 GB.
- **Records made with the earlier files** (section 10, `reference/decode/PROVENANCE.json`, the prep manifests) name the checkpoint sha256 and protocol digests they were made with; the weights and the protocols are the same.
- **Batches** issued for an earlier revision do not run on r6; they are issued again for r6.

### 10.7 Revision r7a: the training-object-count sweep path (2026-11-23)

Revision r7a adds code and one reference vector for experiments that train the steering method on many training objects and compare it with the best single noise. The evaluator files, the protocol files and their digests, the
asset parts of r6, `configs/` and every reference vector of r6 are byte for byte r6's (r7a adds three asset parts, section 10.8). The release digest (the sha256 of `SHA256SUMS`) changes because files were added and others changed, so **batches issued for r6 do not run on r7a;
they are issued again for r7a.**

- **`dsrl train`** accepts a training list file (`train_ids_file`, its sha256 in the config; one id per line) instead of a list inside the config, and keeps the noise pool on disk as a read-only memory map that is tiled by index, so a
  training set of many thousand views needs no memory for its pool. A config with a list file must carry the registered recipe (the update-to-data ratio, the noise-critic steps, batch size, network size, learning rate, soft-update rate,
  noise bound and pool size are asserted in the config and in the learner that was built). The number of offline calls is the one in the config (it follows the calls rule 1,000 x labelled executions / 23,712, rounded to 250, when the config says
  so); the diagnostics report the epochs, and a run stops when the labelled executions found differ from the list's by more than 0.5 %.
- **`--state pca64|full`** (`dsrl train`; config key `state_mode`, default `pca64`). `pca64` is the behaviour of r6 and of every earlier batch: the actor sees the stored 64-number state. `full` makes the actor see the 1,024-number pooled scene encoding of the prior's
  encoder, read from `states_full.npz` next to the prep's `states.npz` (`ids`: exactly the ids of `states.npz`, in the same order; `pooled`, shape (n, 1024), float64, not standardised) and standardised per dimension with the mean and standard deviation of the TRAINING
  views only (the statistics are stored in the run and used again by the read-out). A config may carry `states_full_sha256` (then the file must match it); without it the run records the sha256 of the file it read, in `diagnostics.json` and in the statistics file,
  together with the largest absolute standardised value over the non-training units and the number of dimensions without spread. `wf.py census encode` writes the file for a prep you built yourself. No `states_full.npz` of the shipped preps is part of this release.
- **`--noise-source pool|fresh`** (config key `noise_source`, default `pool`). `fresh` is the authors' own noise-critic update: fresh N(0, I) draws decoded by the prior at every noise-critic step instead of the stored pool. It needs the config key `fresh_input_set` (the input
  set that names the training views) and the prior's weights, keeps every training view's scene encoding on the GPU (about 0.8 MB per view), opens only the pictures of the training units (a unit of the test set in the list is refused), and is 14 to 19 times slower per call
  (about 2.1 to 2.5 hours per 1,000 calls at 40 objects on an RTX A6000). The diagnostics report `fresh_decoder_calls`.
- **`dsrl readout --normrand-key index|unit`.** The norm-matched random control draws its direction from the position of the unit in the prep (`index`, the default and the behaviour of r6) or from a number derived from the unit's id (`unit`): then the same unit gets the same direction
  at every training-set size.
- **Test-object exclusion.** A config with a training list or a search list must name a file of held-out objects (with its sha256); the training list, the search list and every read-out of the run are refused when they name one.
- **`dsrl readout`** has four more arms: `zero` (the zero noise), `prior` (one seeded draw per input), `const` (one stored 99-number vector for every input) and `file` (one noise per input from an npz file). The last three and `file` need no trained actor.
- **`wf.py cem`** (job type 5) searches the single best noise for a set of training inputs by an iterated cross-entropy search on the shell of the prior (8 generations of 32 draws, 8 elites), decoding and rolling out every draw. It runs as
  the batch group kind `cem_readout`, one generation per step, and resumes where it stopped. The batch group kind `nn_readout` reads out the `file` arm without any training.
- **Sweep pages.** `sweep_page.py` (it ships next to `run_all.sh` in the batch's delivery folder, not in this release) prints one page of at most 50 lines of 8 numbers per sweep batch: the successes of each of 20 unseen test objects for the trained
  method, the searched constant, the zero noise and one prior draw, the training set that was really used, and two self-test rows. `run_all.sh` handles a batch folder with a `SWEEP.json`.
- **`wf.py verify` level L3** gains three checks on `reference/dsrl/mini_sweep` (3 shipped objects, 55 views, 1,760 labelled executions, 25 calls): the memory-mapped pool trains bit for bit like the stored tiled pool (T5, T6b) and the 25-call run
  equals its stored record within the tolerance of the other mini training (T6).

Checked on our side before the release was cut: the code copies and protocol files rebuilt from the source commit equal r6's (36 files); the memory-mapped pool draws the same batches as the stored pool bit for bit, on synthetic data and on the shipped
40-object prep, and a training on either gives identical parameters; the constant search, replayed on the stored labels of three earlier searches (3 seeds x 8 generations), reproduces every draw, elite set, distribution and final constant bit for bit; the
page printer and its decoder reject every single-digit change, a transposition, a field swap and a wrong object order.

**Gates run before this release (what was run, on what, and what was not).** They ran on 2026-11-22 on an RTX A6000, on a staged copy of this release's code files (byte for byte the files of this folder; the tag did not exist yet), not on a clean checkout of the tag:

| Gate | Result |
|---|---|
| Replay of the earlier trainer: 26 calls of the sweep trainer path on the 40-object prep, compared with the stored log of the earlier trainer | calls 0 and 25 equal field by field (only the run time differs); the first generation of an earlier constant search replayed through the new search code: draws, 32 x 200 labels and the next distribution equal; the zero-noise read-out on the 368 test inputs equals the stored result |
| `wf.py dsrl check` (T1, T3, T4, T5, T6b) | pass; T4 (20 calls on the mini prep) worst relative difference to the stored record 0, T5 and T6b bit for bit |
| `wf.py dsrl check` with the shipped record `reference/dsrl/mini_sweep/expected_mini_sweep.json` (T6) | **pass**: the 25-call run on the 3-object mini prep equals the shipped record (worst relative difference 0 on the same card model that wrote it; tolerance 0.02) |
| Tiny 200-object batch: training (250 calls, 3,704 views, mapped pool), read-out, rollouts | finished |
| One generation of the constant search with real decodes and rollouts, from the unpacked object part | finished |
| `--noise-source fresh`, 25 calls on the 3-object mini prep with the real prior | finished (250 decoder calls = 25 x 10) |
| Census self-test (`wf.py census selftest`, 57 views of three shipped objects against the shipped census) | passes: 57 of 57 views reproduced, 9 of 1,824 labels differ (the bar on another card model is 18), largest state difference 4.6e-5, largest action difference 5.1e-5 (bar 1e-4), 0 invalid rolls |
| Census with the label-noise mode `fresh64` (a real census of two objects, noise rows checked distinct at `finish`) | run on 2026-11-22, 22:30 to 22:37, on an RTX A6000, with a census builder of this release (commit 0efb711bd, with the arena fix): two tiny census runs on the same 2 benchmark objects (37 views, 1,184 draws, pool rows 16): `fresh64` rows distinct yes (1,184 of 1,184, 0 duplicate pairs; the pool rows 592, 0 pairs), noise rule tagged `fresh64`, 0 invalid rolls, 0 MuJoCo warnings, canaries 0 of 20; `fresh` noise rows bit-equal to the 40-object `fresh` prep for the same views yes (1,184 of 1,184); `fresh64` rows equal `fresh` rows: 0 of 1,184, as they must differ |
| `--state full` on a real prep | run on 2026-11-22 on a 40-object prep built by us with the census builder (741 views, label noise fresh) that carries `states_full.npz`: a 25-call training with `--state full` (state width 1,024, no pin in the config, the sha256 of the file that was read is recorded in the diagnostics and the statistics file; 9.7 s, 1.5 GB peak) and the read-out of the 368 test inputs (all finite). The largest standardised value over the non-training units was 15.8, and no dimension was without spread. That prep and its `states_full.npz` are not part of this release |
| Bit-equality of the prior's decode against its own sampler on this card model | **not run** (it was run on another card model in November; it has not been repeated on an RTX A6000) |

The census builder was changed after the first gate runs (the `fresh64` label-noise mode and the uniqueness check at `finish`); the census builder's own gates (the self-test, the `fresh64` census) and the replay of the earlier trainer were repeated with the final census builder: the self-test gave the same numbers as in the table (57 of 57 views, 9 of 1,824 labels, largest differences 4.6e-5 and 5.1e-5), and the replay passed again (calls 0 and 25 equal, the constant-search generation equal, the zero-noise read-out equal). The training, constant-search and read-out gates were not repeated: they do not use the census builder, and the files they run are byte for byte the files that were tested.

Two things this table does not say: the gates ran the code of this release on our machines, not on yours; and the neutral-vocabulary renaming of files, keys and texts is not part of r7a (names of the generator and the benchmark appear as in r6).

### 10.8 The 200-object training set and the DexGraspNet objects (release r7a)

Release r7a adds three asset parts. `prep_ex365_union200_p2` (about 12 GB) is the training set of the 200-object sweep: 200 objects (the 40 HUG-Bench objects of `prep_ex338_a_p2` and 160 objects of the DexGraspNet dataset), 3,704 views, 32 decoded grasps per view with their success labels under `mano-1111-p2`, the 4,096-row noise pool of every view, and `train_ids.txt`, the list of the 200 object ids. A batch that trains on 200 objects names this folder in its configuration (`prep`, a folder name below `<assets>/prep`); the trainer checks the sha256 of its manifest against the pin in the configuration. `dgn200_objects` (0.4 GB) and `dgn1126_objects` (about 2 GB) hold the simulation assets, input pictures and input sets of 200 and of 1,009 qualified DexGraspNet objects (61 objects are in both parts, with identical files: unpack both into one asset root); the searched-constant jobs and the census builder read them, and `dgn200_objects` also holds the union input set `ex365_union200_1122.json` that every 200-object group reads.

**The DexGraspNet objects are licensed CC BY-NC 4.0: non-commercial use only.** Each part carries an `ATTRIBUTION.txt` with the source and the notice of every collection its objects come from (and `LICENSE_DexGraspNet.txt`; the `ATTRIBUTION.txt` inside the prep part calls that file `LICENSE.txt`: it is the same file under its real name); read them before you use the parts. The numbers in the parts (decodes and labels) are this project's work and follow the objects' licence.

Download these parts only when a batch or the census builder asks for them; the batches of 40 objects need none.

## Building your own training set: the census (`wf.py census`)

**What it builds.** For a list of qualified objects the census builds the labelled training set that the steering actor trains on, in the prep format that `wf.py dsrl train` reads: per *view* (one object at one of the 19 camera levels) a 64-D state, 32 grasps decoded by the HUG2 generator from fixed random noise (the noise and the normalised 99-D action are stored), a pool of 1,024 further noise rows with the generator's outputs, and a simulated success label for each of the 32 grasps under protocol `mano-1111-p2`. Nothing is refitted: the state is the generator's scene encoding standardised and projected by the fit that ships with the benchmark prep.

**Inputs (all from this release's assets, or from the qualified object part you received).**

| Input | Where |
|---|---|
| the object part: object folders (`assets/<source>/<object>/sim_assets`) and the pictures | the part's `assets/` (link its source folders below `<assets>/assets`), its `inputs/input_store/` and the input set `inputs/input_sets/<set>.json` |
| the fit (mean, spread, projection) | `<assets>/prep/ex338_a_p2/states.npz` (default; never refitted) |
| the generator, the encoder, the evaluator, the canary bank | the `hug2`, `dinov2`, `core` and `batches` parts of this release, and the MANO model you downloaded |
| the object list | a text file with one object id per line, or a json `{"ids": [...], "shards": {"L1": [0, 160]}}`; the rank of an object in the list fixes the noise pool of its views, so keep the list order across shards |

**Commands, in order** (`--assets` goes right after `wf.py`; one work folder per shard, outside the release and the asset root):

    python code/wf.py --assets $ASSETS census plan   --objects LIST --shard L1 --inputs PART/inputs/input_sets/SET.json --store PART/inputs/input_store --out WORK/L1
    python code/wf.py --assets $ASSETS census decode --out WORK/L1 --gpu 0                  # GPU: states, 32 draws, noise pool, per 64-view chunk
    python code/wf.py --assets $ASSETS census roll   --out WORK/L1 --workers 8 --cpu-python $CPU_PY      # CPU: the labels (the release's own `rollout run`, with its known-answer jobs)
    python code/wf.py --assets $ASSETS census finish --out WORK/L1                          # joins the chunks into WORK/L1/prep and checks it
    python code/wf.py --assets $ASSETS census status --out WORK/L1                          # where every chunk stands

`census run` does plan, decode, roll and finish in one call. The plan is written once; a different shard, list, input set or fit needs a new `--out`. A chunk is 64 views; every stage of a chunk ends in a stamp, so after any interruption you start the same command again and finished chunks are skipped (an unfinished one is set aside in `_failed` and redone). Run decode and roll at the same time (decode on the GPU, roll on the CPU: they work on different chunks and the roll is hidden behind the decode). One decode process per GPU; for several GPUs start `decode --worker I/N --gpu I` for I = 0 .. N-1 on the same work folder. The roll stage checks about 20 known-answer jobs per chunk against stored answers and refuses to stamp a chunk whose answers differ: that points to a wrong install, hand model or CPU class, not to your objects. `finish` writes `WORK/L1/prep/{states.npz,actions.npz,pool.npz,tokens.npz,manifest.json}`; `manifest.json` has `pass: true` only when every check held. The shipped evaluation states are appended to `states.npz`, so a training on this prep can read out the shipped test inputs.

**Two workers on one shard (a second GPU, or a second machine on the same disk).** Start the first worker as above. On the second: `census decode --out WORK/L1 --gpu 1 --reverse` and `census roll --out WORK/L1 --workers 8 --reverse`; it takes the chunks from the last to the first and never writes the prep. Per-chunk lock files keep the two apart, so no chunk is decoded or rolled twice. Only the first worker runs `census finish`. If `finish` says that some chunks are not finished, the second worker had not ended: wait until it has, check that `ls WORK/L1/chunks/*/*.lock` lists nothing, and run `census finish` again (a refused `finish` writes nothing). If a worker was killed, its chunk keeps a lock file: after you have checked that no process of yours still works on it, remove that one lock file (or run the stage once with `--force-unlock --chunks K`) and start the worker again.

**Cost** (measured on RTX A6000 cards that other users were using at 96 to 100 %; an unloaded card is probably faster). Decode 4.65 to 5.0 seconds per view all in (about 1.0 to 2.1 s for the encoder and the 32 draws, 3 ms per pool row); a 64-view chunk takes 4 to 5.6 minutes. Rollouts cost 0.22 core-seconds each, 32 per view, so about 7 core-seconds per view (older, slower CPUs: up to 18). For the 18,695 views of a 1,009-object part that is 24 to 26 GPU-hours and about 36 core-hours (up to 97 on slow CPUs); our own build of its 948 objects that the 200-object part does not have is 17,568 views, 23 to 24 GPU-hours. With N GPUs the decode time divides by N when the disk keeps up, and about 12 busy cores keep the rolls behind eight decodes. Disk: 846 MB per 1,000 views, so about 15.8 GB for 18,695 views. We used cards with 48 GB; the decode needs a few GB.

**Order for a first build.** (1) `census selftest` (below). (2) One shard: `decode --max-chunks 1`, then `roll` on that chunk, and read the roll line (`canary mismatches 0/21` and a label rate near 30 to 40 %): a CPU class or install that does not reproduce our known-answer jobs shows here, not after the GPU hours. (3) Everything else. Each chunk's roll must show at most one known-answer mismatch; the build as a whole is checked again at `finish`.

**A chunk the evaluator cannot roll.** Objects with about 100 or more collision pieces can produce grasps with so many contacts that the evaluator's constraint arena overflows (`mujoco.FatalError: mj_stackAlloc: out of memory ... mj_makeAR`; seen on an object with 122 pieces, never on one with 92; the largest of our 948 new objects has 170, 16 have 100 or more). It is deterministic (the same chunk fails again on another machine), the scene and the evaluator are pinned, and it stops the chunk: by default `census roll` stops with the chunk named. `census roll --keep-going` records that chunk in `<work>/unrollable.json` and goes on; `--retry-unrollable` forgets the list. `census finish` refuses while a chunk is unrolled and names it; `census finish --exclude-unrollable` builds the prep WITHOUT the listed chunks and writes into the manifest (`excluded`) the chunks, the number of views, the objects that lost all or some levels and the error text: no label is invented for a grasp the evaluator cannot run, and the training set is simply smaller. On our parts this was 1 chunk of 79 (64 views). Keep the number of excluded views under 0.5 % of the shard, or say so in what you send us. **In this release the stop does not happen on our objects any more:** the release's roll job gives every scene a larger MuJoCo arena (`ARENA_MEMORY` in `wf_lib/roll.py`, 128 MB instead of 16 MB). The size changes no physics: the same jobs give bit-identical results with it (shown on a finished chunk, and on the object that stopped the roll). `--keep-going` and `--exclude-unrollable` stay for any other failure of the evaluator.

**Two switches for experiments (defaults unchanged).** `plan --label-noise` picks how the 32 label noises of a view are drawn. `shared` (default, the shipped census's rule): the 19 levels of one object decode the same 32 noise rows. `fresh`: every (object, level) gets its own 32 rows, keyed `<object>_<level>`. `fresh64`: every VIEW gets its own 32 rows, keyed by the object plus the unit id exactly as your input set names it (`a1/ab_roller`; `rv005/ab_roller` in a random-view set), so the noise follows the view and not its level slot; two builds share a view's noise only when the unit ids match. Why `fresh64` exists, in plain words: the release derives a 63-bit number per noise row, but the CPU random generator it feeds uses only the low 32 bits, so two different keys can end up with the same row. Among n rows the expected number of such equal pairs is n x n / 2 / 2^32: about 0.1 for 32,000 rows (1,000 objects under `shared`), 1.6 for 118,000 rows and about 40 for 590,000 rows (`fresh` at 19 views per object). `fresh64` seeds numpy's generator with all 256 bits of the same hash, so such pairs do not occur. `shared` and `fresh` are unchanged, bit for bit, so a census built with them stays valid. `census finish` now checks every label-noise row (rows meant to be identical, the same key and draw, are exempt) and every pool noise row, writes the counts into the manifest (`checks.noise_uniqueness`) and refuses a prep with an equal pair before writing anything; `--allow-duplicate-pairs N` accepts up to N pairs of a legacy (`shared` or `fresh`) prep and records N. The pool rows (numpy generator, seed 315000 + pool index, unique per view) never had the 32-bit limit. The builder still takes at most one view per (object, level) and so at most 19 views per object; more than 19 views per object are not supported by this builder.  `census finish --full-states --eval-encodings FILE` (and `census join --full-states`) also write `states_full.npz` next to `states.npz`: the pooled 1,024-D encoding per row, `ids` exactly those of `states.npz`, `pooled` float64 unstandardised; `census encode --inputs SET.json --out FILE` makes it for the shipped benchmark views and test units (GPU, encode only; it checks the projection against the stored states). Several things are NOT promised: a rebuilt pool equals ours only with our list file and shard (the rank of an object in the list fixes its pool rows; `--objects ALL` ranks differently), and a rebuilt shard from your own renderings differs from ours at the level of the renderer's own differences.

**Check your install against ours before the first big shard.** One line to paste into the chat (it needs the same assets and a GPU):

    python code/wf.py --assets $ASSETS census selftest --out WORK/selftest --pool-rows 1024 --workers 4 --cpu-python $CPU_PY

It builds 57 views (3 of the 40 benchmark objects, 57 of the 741 benchmark training views) with 1,024 of the 4,096 stored pool rows, and compares them with our shipped census of the same views. This is a tolerance check, not a bit-for-bit one: noise rows and pool noise exactly equal, states, actions and pool outputs within 1e-4, and the labels within 0.2 % on the GPU model that made the shipped census (an RTX 3080) or 1.0 % (18 of 1,824) on any other model. It prints a summary and the line `page_row`: `{"B": 0, "I": 997, "c": [views reproduced, views, label mismatches, largest state difference in 1e-6 units, 0]}`; on an RTX A6000 we measured `[57, 57, 9, 47, 0]` and `pass: true`. `pass` is true only when the labels, states, actions and pool are within the bars and the 57-view prep's own checks held. A failure shows as `"pass": false` (the manifest's `problems` list says which condition), a roll that stops with "known-answer jobs differ", or `finish` with `pass: false`; send us `WORK/selftest/selftest.json`, the `manifest.json` of the failed prep and the output of `python code/wf.py fingerprint --out FILE`. For orientation: our reference census of 200 objects (3,705 views) has manifest sha256 `0fd8b1a13468bdab964173192377a63738518597df3bd670c395a7deb18294da`, the shipped benchmark prep `eaa611e1db33133cb50f5d2c9ec1b97e4f86055b2fdd676923179e96b0152bec`; a census you rebuild will not be bit-equal to ours (another GPU decodes the last digits differently, and a few labels flip), only within the bars above.

**Several parts of one training set.** The rank of an object in its list fixes the pool rows of its views (`pool_k = 2000 + rank x 19 + level index`). With `--objects ALL` the rank restarts at 0 in every part, so give each later part `plan --pool-k-base N`, N = 19 x the number of objects of the parts before it (default 0 = the shipped rule); two parts must not share pool rows. `plan` also refuses a shard that holds an object named like one of the shipped test objects, or listed in `--forbid-ids FILE` (exact names only: check near-duplicates of the test objects yourself), and refuses an input set with two units for one (object, level). A picture rendered on your machine differs from ours in the last digits of the anti-aliased colour, so a view you render yourself has states and labels close to, not equal to, ours. The input set's `dgn_split` and `readout_set` fields are ignored by the builder: the objects come from your list, and the shipped evaluation rows come from the fit file.

**Putting a shard into a training run.** The trainer reads one prep folder per run, so the benchmark prep and your shards are joined first: `census join --out DIR --train-ids IDS --source bench=$ASSETS/prep/ex338_a_p2 --source L1=WORK/L1/prep [--source L2=...] [--rows-stored 4096]` (CPU only, no torch needed). `IDS` lists the training objects, benchmark and new alike; at the default 4,096 stored rows the benchmark pools stay bit-identical to the shipped ones (60 GB for 1,000 objects), `--rows-stored 1024` keeps 15 GB and cuts the benchmark views that carry 4,096 distinct rows (recorded in the manifest). Measured for 200 objects (3,704 views): 4 minutes 22 seconds, 12 GB written, peak resident memory 11.1 GB, mostly memory-mapped file pages. At 1,000 objects it has not been measured; expect 60 GB of disk and a run of about 20 minutes. If the sources were built with different `--label-noise` rules, `join` refuses (`--allow-mixed-label-noise` records `mixed` in the manifest). `census tokens` and `census tokens-join` only matter for a projection-coverage report.

**The full encoding per view (optional).** `census finish --full-states` (and `census join --full-states`) also write `states_full.npz` (`ids` = exactly those of `states.npz`, `pooled` (n, 1024) float64, unstandardised mean over the scene encoder's tokens). `census encode --inputs SET.json --out FILE [--save-token-grid]` makes it for the shipped benchmark views and test units (GPU, encode only, about 1 s per view); `--save-token-grid` also writes `tokens_grid.npz` next to it (the token grid per unit, float32, about 1 MB per unit). Both are off by default.

**What the uniqueness check at `census finish` means for you.** Under the default `shared` rule a build over about 1,000 objects stops at `finish` with about an 11 % chance (about 0.1 expected equal pairs among the label-noise rows): that is the 32-bit limit of the old rule and not a damaged install; either rebuild with `--label-noise fresh64`, or accept the pair with `--allow-duplicate-pairs 1` (the count is then recorded in the manifest). Every new manifest, also one made with `shared` or `fresh`, carries the extra record `checks.noise_uniqueness`; the arrays of such a build are byte for byte those of the earlier builder.

## 11. What is not in this release, and what to know

- **Not included:** the experiments' drivers and registry code, rendering of new pictures (a new prep from objects you hold is built with the census builder, section 10.8), the analysis tables (bootstraps), the
  wall / keep-out / cabinet-door scenes, real-robot code. The simplified "floating DSRL" of our earlier work is deliberately absent: it is not the authors' algorithm.
- **Files that must not be edited:** `code/tools/float/collab_native.py`, `wristload.py`, `wristoffset.py`, `code/tools/float_hands/{fit4r,hand_eval,rollout_hand}.py`,
  `code/tools/float_wuji/{wuji_eval,rollout_wuji}.py` and `code/rl/sim/multihand/{retarget,specs,handspec_build}.py`: their sha256 is part of the protocol digests, so an edit
  changes every result stamp. Revision r6 replaced the unused default paths, a branch name and role wording in four of them by placeholders and descriptions (only docstrings, comments and unused defaults: section 10.6), so their sha256 and the protocol digests are new; `L0` flags any edit.
- **Release provenance:** `RELEASE_SOURCES.json` lists every copied file with its source path and sha256 and, for the files edited on the way, what was edited (none of them is digest-pinned): the
  protocol json copies lost their `hosts` tables and machine names (no digest covers them), `protocol_hand.py` and `protocol_wuji.py` lost the functions that write protocol files from our machines, and
  four docstrings lost paths. Provenance fields in the protocol files, reference files and some docstrings name files of our internal repository (`runs/...`, `docs/...`) that are not shipped; they are pointers for us.
- **The DSRL prep folders** (`prep_*`) are our run outputs with the names of our machines and their absolute paths replaced (`host A`, `host B`, asset-root paths); every array is unchanged except the `host` array of the pools.
  Each manifest's `released_from` holds the sha256 of our original manifest, and `configs/dsrl_*.json` pin the released manifests.
- **Licences:** our own code, records and data are MIT ([LICENSE.txt](LICENSE.txt)), except the three parts made from DexGraspNet objects (`prep_ex365_union200_p2`, `dgn200_objects`, `dgn1126_objects`), which are CC BY-NC 4.0: non-commercial research use only, with attribution (`LICENSE_DexGraspNet.txt` and `ATTRIBUTION.txt` in each part). Some files wrap or derive from third-party code (the adapter and the robot-hand evaluators around the MIT-licensed HUG-Bench evaluator; the DSRL port of the authors' MIT-licensed code): [THIRD_PARTY.md](THIRD_PARTY.md). Every asset part keeps the licence of its source: [ASSETS.md](ASSETS.md).
