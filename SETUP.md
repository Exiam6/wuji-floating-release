# Setup from zero: run our simulation experiments on your cluster

Chinese version with the same steps: [SETUP_zh.md](SETUP_zh.md). The reference behind this guide is [README.md](README.md).
Shorter documents for the first verification run: the one-page checklist [QUICKSTART.md](QUICKSTART.md) and the work order for an AI agent, [QUICKSTART_AGENT.md](QUICKSTART_AGENT.md).

**Who this is for.** Anyone with a cluster account (or a GPU workstation) who runs the simulation experiments of this release; when you run a batch issued by the project, you send back **numbers, never code**. This guide takes a fresh account to a verified machine and to the first batch.
Every command below was run as written, in a fresh folder, before release; the expected output is shown under each step. Times were measured on a 64-core AMD machine with an RTX A6000 and will differ on yours.

## What you receive from us, and what you need

| From us | What it is |
|---|---|
| The code | the public GitHub repository https://github.com/Exiam6/wuji-floating-release (about 8 MB; no account is needed). Release tag: `wf-20261122-r6` |
| A drive link | the asset tarballs (they do not go through GitHub): the folder `full` holds 15 tarballs, 8.69 GB, every part but MANO and HUG1 (you download those two yourself), with `SHA256SUMS`, `ASSET_TERMS.txt`, `README.txt`; the CPU-only subset, `rollout_only`, is 4 of the same tarballs (0.32 GB), a sibling folder repeats its manifest files; and, if you run batches issued by the project, the drive folder `returns/<your name>/<date>/`, where your results go |
| Issued batch folders | the work: the first one is **verification batch 1** (in parts, 1a first), sent as an archive after step 8 |

| You need | |
|---|---|
| Machine | Linux x86-64 (Ubuntu 22.04 and 24.04 tested): a GPU workstation, or a cluster with a login node and GPU nodes |
| NVIDIA driver | one that supports CUDA 12.8 (driver 570 or newer): `nvidia-smi` shows it. The pinned stack is torch 2.9.1 built for CUDA 12.8; other CUDA builds are untested |
| Python | 3.10 (tested for both environments). 3.12 was tested for the CPU environment only |
| Disk | about 27 GB free while you unpack (tarballs 8.7 GB, unpacked assets 8.7 GB, environments 9 GB), about 18 GB after you delete the tarballs, plus your results. Use a fast local or scratch disk (see "Troubleshooting") |
| Internet | during setup only (PyPI, download.pytorch.org, data.pyg.org, github.com, huggingface.co): a login node is enough. Batches run without it. |
| A MANO account | free, at https://mano.is.tue.mpg.de (step 5) |

The tarballs, with the sizes and sha256 values you must see:

```text
file                                  bytes   sha256
core.tar                        140,113,920   d618c33f0f8364d31a3a219fb79fb5a33278dd584a8de10d8e8e4da949dc2dd3
hands.tar                        16,670,720   0fb724f749efc40116f7e8d6b2aa7867ead6f36aeb3547dd945a8c2a2bf09601
dinov2.tar                      346,368,000   7cb905da8bcff10a96c0605df43a8eae805fe33d988c08df7a065c535d53367e
hug2.tar                        241,418,240   df52f406f1418545db86fc9500ef408cc1578d4b534a8e6120749160e818fe6b
dumbbell50k.tar                 241,418,240   b4713a3cd331c432b88bd3df4c0e6ea4b324befca79760693f6969bbe746574c
inputs.tar                       83,773,440   3bcc397104e074160b954b549cd2a292ac06fcff6bf3bd3222055b0b697888e7
licences.tar                         40,960   ed94c56584415462853879d8f398b3c2420c4b12c7bbd4f28415c9b8e932e09c
batches.tar                     147,138,560   f3da3a72e05b31680825e55dea99e5c32addb548f25063d7ae28a79375dc692e
prep_ex299_hug1.tar             183,377,920   50d132dda6be8f9178da790b58a702ec83611266375b9e3c8d3d465db887e3ff
prep_ex299_dumbbell50k.tar      195,747,840   a05476242494a6ac6855c5dc79a01f36ac39d60e76dcad1397d8b81041ce1c6f
prep_ex299_hug2.tar             183,377,920   241cc0b34b513ba701294d0e31097b8a75be97c2d26efaae81198815c4938dc6
prep_ex315_hug2.tar           1,394,216,960   4476a37008d666a0c97696196e42ef6bc34c06643af08e669ae0e39f405975e8
prep_ex324_a.tar              2,430,515,200   954227086d067cf421fc6ab46f399bf784e8a0789c2b53f5a3efe975ab252943
prep_ex324_b.tar                656,885,760   56c8114ab7ece578e670b5f057c7a14ab1542bd0fba262a877d83f320613b688
prep_ex338_a_p2.tar           2,430,515,200   93f4af5af53ca6e99eea95137c102ae5a10be467450a54079a34ae71cb2a9f82
Total 8,691,578,880 bytes (15 tarballs). The folder rollout_only holds core, hands, batches and licences (303,964,160 bytes).
```

## Step 0. Choose a folder and check the machine

Everything goes below one folder, `BASE`. Change the first line if you want another disk. The variables are kept in `env.sh`: after you open a new terminal, run `export BASE=...` (same value) and `source "$BASE/env.sh"`.

```bash
export BASE=$HOME/wuji-floating
mkdir -p "$BASE" && cd "$BASE"
cat > "$BASE/env.sh" <<'EOF'
: "${BASE:?export BASE=/path/to/your/folder first}"
export RELEASE=$BASE/release
export ASSETS=$BASE/assets
export WF_ASSETS=$ASSETS
export DOWNLOADS=$BASE/downloads
export ENV=$BASE/venv310
export GPUENV=$BASE/gpuenv
export PYTHONDONTWRITEBYTECODE=1
EOF
source "$BASE/env.sh"
uname -m
python3.10 --version
df -h "$BASE" | tail -n 1
```

Expected: `x86_64`, `Python 3.10.x`, and a line for your disk with at least 30 GB available. If `python3.10: command not found`: see "Troubleshooting". On a GPU node, also run `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader`; it prints for example `NVIDIA RTX A6000, 570.172.08, 49140 MiB`.

## Step 1. Get the code

The repository is public: `git clone` needs no account and no credentials.

```bash
source "$BASE/env.sh"
git clone -c advice.detachedHead=false --branch wf-20261122-r6 https://github.com/Exiam6/wuji-floating-release.git "$RELEASE"
cd "$RELEASE" && git log -1 --format='%H %s' && sha256sum -c --quiet SHA256SUMS && echo "release intact"
```

Expected: the commit hash and subject of the release, then `release intact` (no other output). Never edit a file in this folder (see the rules box below).

## Step 2. The two Python environments

A CPU environment runs the rollouts, a GPU environment runs the decode and the DSRL training. Both are built from pinned requirements. If your cluster's Python 3.10 is not called `python3.10`, put its full path in the first line of each block.

```bash
source "$BASE/env.sh"
python3.10 -m venv --without-pip "$ENV"
curl -sSfL https://bootstrap.pypa.io/get-pip.py -o "$ENV/get-pip.py"
"$ENV/bin/python" "$ENV/get-pip.py" --no-cache-dir pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 packaging==26.2
"$ENV/bin/python" -m pip install --no-cache-dir -r "$RELEASE/requirements/cpu.txt"
"$ENV/bin/python" -m pip install --no-cache-dir --no-build-isolation --no-deps "chumpy @ https://github.com/mattloper/chumpy/archive/580566eafc9ac68b2614b64d6f7aaa84eebb70da.zip"
"$ENV/bin/python" -m pip check
rm -rf "$ENV/lib/python3.10/site-packages/cmeel.prefix/include"
```

Expected last line of `pip check`: `No broken requirements found.` (about 2 minutes, 1.6 GB). The robot-hand protocols (`sharpa-float-*`, `wuji1hr-float-*`, `wuji1-float-p1`) need one more file, only if a batch names them: `"$ENV/bin/python" -m pip install --no-cache-dir -r "$RELEASE/requirements/cpu_hands.txt"`.

```bash
source "$BASE/env.sh"
python3.10 -m venv --without-pip "$GPUENV"
curl -sSfL https://bootstrap.pypa.io/get-pip.py -o "$GPUENV/get-pip.py"
"$GPUENV/bin/python" "$GPUENV/get-pip.py" --no-cache-dir pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 packaging==26.2
"$GPUENV/bin/python" -m pip install --no-cache-dir -r "$RELEASE/requirements/gpu.txt"
"$GPUENV/bin/python" -m pip install --no-cache-dir --no-build-isolation --no-deps "chumpy @ https://github.com/mattloper/chumpy/archive/580566eafc9ac68b2614b64d6f7aaa84eebb70da.zip"
"$GPUENV/bin/python" -m pip check
```

Expected: `No broken requirements found.` (about 7 minutes, 7.4 GB: the CUDA libraries are most of it). On a node with a GPU, `"$GPUENV/bin/python" -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints `2.9.1+cu128 True`; on a login node without a GPU it prints `False`, which is fine for now.

**Conda instead of venv.** Replace the first three lines of each block (`python3.10 -m venv`, `curl`, `get-pip.py`) by
`conda create -y -p "$ENV" python=3.10 pip` (and `-p "$GPUENV"` for the GPU environment), then the line `"$ENV/bin/python" -m pip install --no-cache-dir pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 packaging==26.2`; every later line stays the same.

## Step 3. Download the assets and check them

Open the drive link and download the folder `full` into `$DOWNLOADS` (a browser downloads a folder as several zip files: unzip them so that the 15 `.tar` files and `SHA256SUMS` lie directly in `$DOWNLOADS`, or download the files one by one; or, with your own rclone remote for that drive,
`rclone copy "<your remote>:<the shared folder>/full" "$DOWNLOADS" --progress`). If you only run CPU rollouts, download just `core.tar`, `hands.tar`, `batches.tar`, `licences.tar` and `SHA256SUMS` from `full` (0.32 GB; the sibling folder `rollout_only` repeats the manifest files of that subset) and check with `sha256sum -c --ignore-missing SHA256SUMS`.
On a cluster whose login node cannot reach the drive, download on your own computer and copy with `rsync -a --partial --progress "$DOWNLOADS/" you@cluster:/path/to/wuji-floating/downloads/`.

```bash
source "$BASE/env.sh"
cd "$DOWNLOADS" && ls
sha256sum -c SHA256SUMS
```

Expected: the 15 tarballs plus `ASSET_TERMS.txt`, `README.txt`, `SHA256SUMS`, and 15 lines ending in `OK` (`batches.tar: OK`, `core.tar: OK`, ...). A line ending in `FAILED` means that tarball is damaged: download that file again. Read `ASSET_TERMS.txt`: it states the licence of every part and the attribution it requires.

## Step 4. Unpack into the asset root

```bash
source "$BASE/env.sh"
mkdir -p "$ASSETS" && for t in "$DOWNLOADS"/*.tar; do tar -xf "$t" -C "$ASSETS"; done
cd "$ASSETS" && sha256sum -c --quiet "$RELEASE/asset_manifests/SHA256SUMS.all" && echo "assets intact"
```

Expected: `assets intact` (about 10 seconds to 2 minutes, depending on the disk). With only the rollout-only set, add `--ignore-missing` to the second command. You may delete the tarballs afterwards.

## Step 5. The MANO hand model (you download it yourself)

The MANO model is not in the tarballs: every user registers and accepts its licence themselves.

1. Register (free) at https://mano.is.tue.mpg.de and accept the MANO licence (non-commercial research use).
2. Download the MANO models, version 1.2: the archive `mano_v1_2.zip`. Put it into `$DOWNLOADS`.
3. Run:

```bash
source "$BASE/env.sh"
unzip -q "$DOWNLOADS/mano_v1_2.zip" -d "$DOWNLOADS"
mkdir -p "$ASSETS/mano/models" && cp "$DOWNLOADS/mano_v1_2/models/MANO_RIGHT.pkl" "$ASSETS/mano/models/"
sha256sum "$ASSETS/mano/models/MANO_RIGHT.pkl"
"$ENV/bin/python" "$RELEASE/code/wf.py" assets verify --parts mano
```

Expected: the sha256 line `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767  .../MANO_RIGHT.pkl`, then a JSON report that ends with `"pass": true`. If the hash differs, stop and tell us: a different MANO file changes every result.

## Step 6. Link the hand models, fetch the HUG source and the HUG1 weights

```bash
source "$BASE/env.sh"
"$ENV/bin/python" "$RELEASE/code/wf.py" assets link-hands
"$GPUENV/bin/python" "$RELEASE/code/wf.py" assets fetch-hug-source
"$GPUENV/bin/python" "$RELEASE/code/wf.py" assets fetch-hug1
```

Expected: the first prints a JSON with `"action": "linked"`; the second clones the public HUG repository at the pinned commit (needs github.com, once) and prints a JSON with `"pass": true` and `"problems": []`; the third downloads the public HUG1 weights (needs huggingface.co, once; they are not in the tarballs) and prints a JSON with `"pass": true`. The `full` tarballs already hold the DINOv2 weights; if that part is missing, `wf.py assets fetch-dinov2` downloads it.

## Step 7. Verify the machine

Do this on the kind of node you will compute on. On SLURM, run the commands through an interactive allocation, e.g. `srun -p <partition> --gres=gpu:1 --cpus-per-task=4 --time=00:30:00 --pty bash`, then `export BASE=...` and `source "$BASE/env.sh"` again. The first command needs no GPU; the second does.

```bash
source "$BASE/env.sh"
nice -n 19 "$ENV/bin/python" "$RELEASE/code/wf.py" verify --levels L0,L1 --workers 4 --out "$BASE/verify_cpu"
```

Expected last line: `VERIFY PASS (11 checks, 201.6 s)` (the seconds vary with your CPU; about 3.5 minutes with 4 workers). Above it, every `l1_...` line must say `pass_bitwise ... all fields equal: True` on an AMD Zen 2 or Zen 3 CPU. Another CPU may print `labels_only`: that is not a pass, see the rules box.

```bash
source "$BASE/env.sh"
CUDA_VISIBLE_DEVICES=0 "$GPUENV/bin/python" "$RELEASE/code/wf.py" verify --levels L0,L2,L3 --out "$BASE/verify_gpu"
```

Expected last line: `VERIFY PASS (12 checks, 31.8 s)`. The `L2 decode` lines say `PASS_TOLERANCE max |d params99| ...`: the decode of the same six pictures differs from ours by GPU rounding; anything below 1e-3 passes (we measured 8.2e-6 on an RTX A6000 against an RTX 3080 reference). `L3 T4_mini_training ... PASS_TOLERANCE` compares 20 training calls with ours within 2 %.

## Step 8. Send us your host fingerprint and the two reports

```bash
source "$BASE/env.sh"
CUDA_VISIBLE_DEVICES=0 "$GPUENV/bin/python" "$RELEASE/code/wf.py" fingerprint --out "$BASE/host_fingerprint.json"
ls "$BASE/verify_cpu" "$BASE/verify_gpu"
```

Expected: a JSON with your CPU model, `numerical_host_class` (`amd_zen2_zen3_bitwise` on AMD Zen 2 and Zen 3 CPUs, otherwise `other_tolerance`) and your GPU model and driver (it lists every GPU of the node, whichever you use); then the two report folders. **Send us `host_fingerprint.json` and the file `verify_report.json` of each report folder** (upload to the drive folder `returns/<your name>/setup/`). We confirm your host, then send verification batch 1. Do not run any batch before that: the first thing we need from you is the GPU verify report under the pinned stack.

## Step 9. Run an issued batch

A batch is a folder we send as one archive (we tell you where to download it; put it into `$DOWNLOADS`). Unpack it; it never changes afterwards. Below, the batch is called `verification_batch_1`: use the name of the archive you received (`verification_batch_1a`, `verification_batch_1b`, ...).

```bash
source "$BASE/env.sh"
mkdir -p "$BASE/batches" "$BASE/results"
tar -xf "$DOWNLOADS/verification_batch_1.tar" -C "$BASE/batches"
export BATCH=$BASE/batches/verification_batch_1
export RESULTS=$BASE/results/verification_batch_1
cat "$BATCH/INSTRUCTIONS.txt"
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch run "$BATCH" --out "$RESULTS" --cpu-python "$ENV/bin/python" --dry
```

`INSTRUCTIONS.txt` lists the groups (one group is one experiment). `--dry` prints the plan, step by step with the commands, and runs nothing; it refuses a batch that was issued for another release than the one you have checked out. Then run it in one of three ways.

**A. One machine, several GPUs (the plain way).** One group runs per GPU at a time; every GPU step gets its own `CUDA_VISIBLE_DEVICES`.

```bash
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch run "$BATCH" --out "$RESULTS" --gpus 0,1,2,3 --cpu-python "$ENV/bin/python"
```

Use `--gpus 0,0` for two jobs at once on one card with enough memory (a DSRL job needs about 4 GB, a decode about 2 GB). Add `--workers N` to limit the CPU workers of the rollout steps (default: the cores divided by the GPU slots, which is all of them on a dedicated node). Run it inside `tmux` or `nohup`.

**B. SLURM, one array task per group.**

```bash
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch slurm "$BATCH" --out "$RESULTS" --partition <partition> --gpus-per-task 1 --cpus 8 --time 24:00:00 --mem 64G --gpu-python "$GPUENV/bin/python" --cpu-python "$ENV/bin/python"
sbatch "$RESULTS/run_batch.sbatch"
```

Add `--account`, `--qos`, `--constraint`, `--gres gpu:a100:1` (instead of `--gpus-per-task`), `--array-parallel N` and `--setup 'module load cuda/12.8'` as your cluster needs; read the `#SBATCH` lines in the script before you submit. `squeue -u $USER` shows the array.

**C. No SLURM (a plain shell loop).**

```bash
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch shell "$BATCH" --out "$RESULTS" --gpus 0,1,2,3 --gpu-python "$GPUENV/bin/python" --cpu-python "$ENV/bin/python"
nohup bash "$RESULTS/run_batch.sh" > "$RESULTS/run_batch.log" 2>&1 &
```

**Watch, resume, return.** `batch status` shows which groups are done. If a job dies (time limit, node failure, Ctrl-C), run the same command again: every finished step is skipped, an unfinished step folder is moved to `$RESULTS/_failed/` and the step runs again. With SLURM, `batch status ... --array` prints the group numbers still to run: `sbatch --array=<that list> "$RESULTS/run_batch.sbatch"`.

```bash
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch status "$BATCH" --out "$RESULTS"
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch pack "$BATCH" --out "$RESULTS" --note "cluster name, partition, anything unusual"
```

`pack` checks that every group is done, collects what we need (the results, their jobs files with the known-answer jobs, the step stamps, the training logs; not the 600 MB checkpoints), and writes one archive, `${RESULTS}_pack.tar`, and prints its sha256. **Upload that one archive to `returns/<your name>/<YYYYMMDD>/` in the drive folder and message us the archive's name and the sha256.** We check the numbers (README section 9) and tell you the result before we send the next batch.

Where things are: `$RESULTS/<group>/<step>/` holds each step's outputs and its `STAMP.json`; `$RESULTS/<group>/logs/` the logs; a group that keeps failing shows its error in `logs/<step>.err` and `<step>.failed.json`: send us that folder.

## Step 10. The first real job: verification batch 1

Verification batch 1 reproduces our most important experiments, so that we can check your cluster against ours before you take new work. It comes in parts, each an archive you run with step 9 (A, B or C), pack, upload, and tell us:

- **1a** reproduces our dumbbell census table (EX331): three grasp priors decoded on your GPU for 164 held-out views and rolled on your CPUs under five wrist conditions, 15 numbers to compare with ours. Our projection: about 0.1 GPU-hours and 13 CPU core-hours. We issue it after step 8 and after our own control run agrees with our registered numbers.
- **1b** trains seven DSRL-NA seeds (about 1.3 to 2 GPU-hours on a card like an RTX A6000) after a first group that repeats one training we already stored, so that we can check your GPU against it.

`INSTRUCTIONS.txt` inside each archive lists its groups. We tell you the verdict of each return before we send the next part.

## The rules, in one box

> 1. **Do not modify any file** of the release folder, the assets or an issued batch. Any edit makes `rollout run` refuse to start and changes the digest stamped in every result.
> 2. If `verify` fails, or prints `labels_only` or `FAIL`, **stop** and send us the report folder. Do not run batches on that machine.
> 3. Tell us your **CPU model and GPU model**: `host_fingerprint.json` (step 8) does it; send a new one after any change of machine, driver or Python.
> 4. **Never pool results from different machines into one pack.** One pack, one machine type. A second cluster gets its own fingerprint and its own pack.
> 5. Send numbers only: the archive `batch pack` writes. No code, no checkpoints, no MANO file, no credentials.

## Troubleshooting

| What you see | What to do |
|---|---|
| `python3.10: command not found` | install a CPython 3.10 (conda, pyenv, or your module system) and write its full path instead of `python3.10` in step 2 |
| `git clone` cannot connect (a node without access to github.com) | clone on another machine and `rsync -a` the folder (`.git` included) to `$RELEASE`; the folder is the same release (L0 of step 7 checks it) |
| `wf: the MANO model is missing` (or `MANO not linked`) | step 5: the file must be at `$ASSETS/mano/models/MANO_RIGHT.pkl`; then run `assets link-hands` / `fetch-hug-source` again |
| `assets verify --parts mano` says the sha256 differs | you have another version of the file: download `mano_v1_2.zip` again (the right-hand model, version 1.2) and stop if it still differs |
| `L0 release folder: FAIL ... not in SHA256SUMS` | a stray file is in `$RELEASE` (a `__pycache__`, an editor backup, your own notes): remove it, or `git status` and `git checkout .`; the `.git` folder itself is allowed |
| `import torch` or any `python` start takes many minutes | the environment or the assets are on a network file system (NFS, Lustre home): put `$ENV`, `$GPUENV` and `$ASSETS` on node-local or scratch disk; we measured 10 to 40 minutes of imports on NFS against 3 seconds locally |
| `CUDA out of memory` | one job per card: fewer slots in `--gpus`; a card below 8 GB cannot run the decode; a DSRL job needs about 4 GB, a decode about 2 GB, plus other users' jobs |
| `no CUDA device is visible` | the node has no GPU or `CUDA_VISIBLE_DEVICES` is empty; on SLURM request `--gres=gpu:1` |
| a decode or training differs from our reference in the last digits | normal across GPU models and driver versions (README section 2). Our verifier accepts a decode when at least 99.5 % of its rows are within 1e-3 of our stored decode and all within 0.1: on an RTX A6000 against our RTX 3080 decode we measured 0 rows beyond 1e-3 for HUG1 and HUG2 and 2 of 7,040 rows for the dumbbell 50k (samples near a branch of the flow, up to 4.5e-2). A DSRL training log is compared line by line (bit for bit on the same GPU model, within 2 % on the first 100 calls on another). Whether an experiment is *reproduced* is decided per cell by the rate difference in points (0.31 for reproduced, 1.0 for close) and the number of flipped labels, not by these tolerances. A difference above them: send us the report |
| `this batch cannot run on this copy of the release` | your checkout is not the release the batch was issued for: `git fetch --tags && git checkout <tag in INSTRUCTIONS.txt>`, then repeat step 7 once |
| `the batch folder differs from its SHA256SUMS` | the archive was damaged or edited: unpack it again from the original |
| `sha256sum -c` says `FAILED` for a tarball | download that tarball again |
| `group ... is being run by another process` | another `batch run` works on that group, or an earlier one died on another node: remove `$RESULTS/<group>/RUNNING.lock`, or pass `--force-unlock` |
| SLURM: `Invalid generic resource (gres) specification` | your cluster names GPUs differently: use `--gres gpu:<type>:1` in `batch slurm` instead of `--gpus-per-task` |
