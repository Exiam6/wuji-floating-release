# Work order for the agent: verify this machine, then run `verification_batch_1a` and return the numbers

You are an AI agent operating a machine or a cluster account for a person who runs this release. This file is your complete work order. Do the sections **in order**. Run every command exactly as written, except where a comment says CHANGE. After each command compare the output with **Expect** (a line in angle brackets such as `<N>` stands for any value). Where it says **STOP**, stop, do not try to repair, and send the report of section 15. The checklist for the human is [QUICKSTART.md](QUICKSTART.md) (Chinese: [QUICKSTART_zh.md](QUICKSTART_zh.md)). The long explanations are in [SETUP.md](SETUP.md) and [README.md](README.md): read the section named in a step only when its output differs from what is expected there. The commands of this file were run as written, in a fresh folder, for revision r4 (README section 10.3); revision r6 changes the release tag, the commit subject that section 3 expects and the tarball count (15), and adds `assets fetch-hug1` to section 6 (the HUG1 weights are not in the tarballs).

**The job.** Install our pinned software stack on this machine, verify it, send us a host fingerprint and two verification reports, wait for our go, run the issued batch `verification_batch_1a` (grasp decoding on the GPU, physics rollouts on the CPUs) and return one archive of numbers. We compare your numbers with ours. You send numbers only: no code, no checkpoints, no credentials. To install and verify the release only, sections 1 to 8 are all you need; sections 9 to 15 are for a batch the project issues to you.

**What only the human can do** (ask for it, do not try to do it yourself): download the Drive folder we shared; register at https://mano.is.tue.mpg.de and download the MANO archive; upload your files to the Drive folder (or give you an rclone remote that does it); message us.

## 0. The rules (read twice)

1. **Do not modify any file** of the release folder, of the assets or of an issued batch folder: no patches, no "fixes", no re-formatting. An edit makes `rollout run` refuse to start and changes the digest stamped in every result. If something does not work, report it; we fix the release.
2. **No batch before our go.** Run sections 1 to 9, send the report of section 9, and wait for our message before section 10.
3. **One pack, one machine type.** All steps of one results folder run on the same CPU family and the same GPU model; never copy a results folder to another machine and continue there; never merge two results folders. On a cluster with mixed node types pin one type (section 11, SLURM). A pack that mixes machines is rejected.
4. **Do not re-run a finished job**, delete a results or `_failed` folder, or pass `--partial` or `--force-unlock`, unless we tell you to.
5. **Known-answer jobs are interleaved in every batch** (README section 9.2). `wf.py batch pack` refuses a results folder in which one is missing, and our verifier rejects a pack in which one is missing or altered. Run every row of every jobs file exactly as it is: never filter, reorder, subsample, edit, treat differently or re-run selected rows, and never edit a `.npz`.
6. **Stop and report** on any verify failure, any `labels_only`, any `FAIL`, a hash that differs, a MANO file whose hash differs, `release_ok` false, a driver older than 570, or a change of machine or GPU model in the middle of a batch. Do not start a batch on a machine whose verification did not pass.
7. **Send numbers only.** Upload the files named in section 15 and nothing else: no checkpoint, no MANO file, no token, no password, no environment dump. The asset parts carry their own licences (`ASSET_TERMS.txt`); the MANO file is under the user's own licence from the Max Planck Institute: never pass it on or put it on a shared or public place.
8. **Tell us your CPU model and GPU model** (the fingerprint of section 9 does it) and send a new fingerprint after any change of machine, driver or Python.

## 1. Check the machine

```bash
export BASE=$HOME/wuji-floating      # CHANGE if $HOME is a network file system: use node-local or scratch disk with 30 GB free
mkdir -p "$BASE" && cd "$BASE"
uname -m
grep PRETTY_NAME /etc/os-release
nproc
free -g | sed -n 2p
df -h "$BASE" | tail -n 1
df -T "$BASE" | tail -n 1 | awk '{print "file system type:", $2}'
"${PY310:-python3.10}" --version || echo "MISSING: python3.10"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "no GPU on this node (fine on a login node: the GPU steps run on a GPU node)"
nvidia-smi | grep -o 'CUDA Version: [0-9.]*' || true
command -v sbatch sinfo || echo "no SLURM commands on this node"
for t in git curl tar unzip sha256sum rsync; do command -v "$t" > /dev/null || echo "MISSING tool: $t"; done
for u in https://pypi.org/simple/ https://download.pytorch.org/whl/cpu/ https://data.pyg.org/whl/ https://github.com https://huggingface.co; do printf '%s %s\n' "$(curl -s -o /dev/null -m 20 -w '%{http_code}' "$u")" "$u"; done
```

Expect: `x86_64`, and a line `Python 3.10.<N>`.

| What you see | Required | If not |
|---|---|---|
| architecture | `x86_64` | **STOP**: no other architecture is supported |
| operating system | Linux (Ubuntu 22.04 and 24.04 were tested) | other glibc distributions usually work: say so in the report |
| `df`: available | at least 59 GB while you unpack (about 34 GB afterwards, plus your results) | choose another `BASE` on a bigger disk |
| file system type | a local disk (`ext4`, `xfs`, `tmpfs`, `btrfs`) | on `nfs`, `nfs4`, `lustre`, `gpfs`, `beegfs`, `cephfs` the environments and assets must go on node-local or scratch disk instead (imports take 10 to 40 minutes on NFS): change `BASE` |
| Python | CPython 3.10.x | find one (`module avail python`, conda, pyenv), then `export PY310=/full/path/to/python3.10` before section 2 |
| driver and CUDA | driver 570 or newer, and `CUDA Version: 12.8` or higher in the `nvidia-smi` header (the pinned stack is torch 2.9.1 built for CUDA 12.8 and brings its own CUDA libraries: only the driver matters, no CUDA toolkit or module is needed) | **STOP**: ask the human for a node with a newer driver |
| GPU memory | 8 GB or more per card (a decode needs about 2 GB, a DSRL training about 4 GB) | use another card |
| network | a status other than `000` for all five addresses (`200`, `301`, `302` are fine) | build the environments (section 6) on a node that has internet; batches run without it |
| tools | `git`, `curl`, `tar`, `unzip`, `sha256sum` (and `rsync` if you have to copy folders) | no line `MISSING tool: ...` may appear: install it (or ask the human) |
| CPU | any x86-64 (AMD Zen 2 and Zen 3 reproduce our results bit for bit; other CPUs are held to a tolerance) | nothing to do, the fingerprint tells us |

## 2. Variables

Every later block starts with `source "$BASE/env.sh"`: after you open a new terminal run `export BASE=...` (the same value) first.

```bash
export BASE=${BASE:?run the block of section 1 first}
cat > "$BASE/env.sh" <<'EOF'
: "${BASE:?export BASE=/path/to/your/folder first}"
export RELEASE=$BASE/release
export ASSETS=$BASE/assets
export WF_ASSETS=$ASSETS
export DOWNLOADS=$BASE/downloads
export ENV=$BASE/venv310
export GPUENV=$BASE/gpuenv
export PYTHONDONTWRITEBYTECODE=1
export PY310=${PY310:-python3.10}
export ON_GPU=${ON_GPU:-"env CUDA_VISIBLE_DEVICES=0"}
EOF
source "$BASE/env.sh"
echo "BASE=$BASE PY310=$PY310 ON_GPU=$ON_GPU"
```

Expect:

```expect
BASE=<path> PY310=<path> ON_GPU=<command>
```

`ON_GPU` is the prefix that puts a command on one GPU. On a workstation or GPU node the default is right. On a SLURM login node set it before sourcing, for example `export ON_GPU="srun -p <gpu partition> --gres=gpu:1 --cpus-per-task=4 --time=00:30:00"`, and use that same value whenever a block starts with `$ON_GPU`.

## 3. Get the code

The repository is public: `git clone` needs no account and no credentials.

```bash
source "$BASE/env.sh"
git clone -c advice.detachedHead=false --branch wf-20261123-r7a https://github.com/Exiam6/wuji-floating-release.git "$RELEASE"
cd "$RELEASE" && git log -1 --format='%H %s' && sha256sum -c --quiet SHA256SUMS && echo "release intact"
```

Expect:

```expect
<commit> wuji-floating release r6
release intact
```

The commit hash must equal the one in our message. If the node cannot reach github.com: clone on another machine and `rsync -a` the folder (`.git` included) to `$RELEASE`. If `release intact` is missing or `sha256sum` reports a mismatch: **STOP** (you changed or damaged a file: delete `$RELEASE` and clone again; if it repeats, report).

## 4. Download the assets and check them

Ask the human to download the folder `full` of the Drive link into `$DOWNLOADS` (a browser downloads a folder as several zip files: unzip them so that the 15 `.tar` files and `SHA256SUMS` lie directly in `$DOWNLOADS`), or to give you an rclone remote for that drive:
`rclone copy "<remote>:<shared folder>/full" "$DOWNLOADS" --progress`. A cluster whose login node cannot reach the drive: download on another computer and `rsync -a --partial --progress` the folder to `$DOWNLOADS`. Download all 15 files.

```bash
source "$BASE/env.sh"
cd "$DOWNLOADS" && ls
sha256sum -c SHA256SUMS
```

Expect:

```expect
batches.tar: OK
core.tar: OK
prep_ex324_b.tar: OK
```

All 18 tarballs must end in `OK`. A `FAILED` line: that file is damaged, download it again (never unpack it). Then read `ASSET_TERMS.txt` in `$DOWNLOADS` (rule 7) and unpack:

```bash
source "$BASE/env.sh"
mkdir -p "$ASSETS" && for t in "$DOWNLOADS"/*.tar; do tar -xf "$t" -C "$ASSETS"; done
cd "$ASSETS" && sha256sum -c --quiet "$RELEASE/asset_manifests/SHA256SUMS.all" && echo "assets intact"
```

Expect:

```expect
assets intact
```

About 10 seconds to 2 minutes. Any other output: **STOP** (a missing or damaged file; report which). Afterwards you may delete the tarballs.

## 5. The MANO hand model (the human downloads it)

The MANO model is not in the tarballs: every user registers and accepts the licence (non-commercial research use) at https://mano.is.tue.mpg.de and downloads the archive `mano_v1_2.zip` (MANO models, version 1.2). Ask the human to put it into `$DOWNLOADS`. Never try to get it any other way.

```bash
source "$BASE/env.sh"
unzip -q "$DOWNLOADS/mano_v1_2.zip" -d "$DOWNLOADS"
mkdir -p "$ASSETS/mano/models" && cp "$DOWNLOADS/mano_v1_2/models/MANO_RIGHT.pkl" "$ASSETS/mano/models/"
sha256sum "$ASSETS/mano/models/MANO_RIGHT.pkl"
```

Expect:

```expect
45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767  <path>/MANO_RIGHT.pkl
```

A different hash: **STOP** and tell us (a different MANO file changes every result; the right-hand model of version 1.2 is required).

## 6. The two Python environments, the hand models, the HUG source

A CPU environment runs the rollouts, a GPU environment runs the decode. Both are built from pinned requirements (needs the network, once). If `python3.10` is not on your PATH, `PY310` (section 2) must hold its full path.

```bash
source "$BASE/env.sh"
"$PY310" -m venv --without-pip "$ENV"
curl -sSfL https://bootstrap.pypa.io/get-pip.py -o "$ENV/get-pip.py"
"$ENV/bin/python" "$ENV/get-pip.py" --no-cache-dir pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 packaging==26.2
"$ENV/bin/python" -m pip install --no-cache-dir -r "$RELEASE/requirements/cpu.txt"
"$ENV/bin/python" -m pip install --no-cache-dir --no-build-isolation --no-deps "chumpy @ https://github.com/mattloper/chumpy/archive/580566eafc9ac68b2614b64d6f7aaa84eebb70da.zip"
"$ENV/bin/python" -m pip check
rm -rf "$ENV/lib/python3.10/site-packages/cmeel.prefix/include"
```

Expect (last line; about 2 minutes, 1.6 GB):

```expect
No broken requirements found.
```

```bash
source "$BASE/env.sh"
"$PY310" -m venv --without-pip "$GPUENV"
curl -sSfL https://bootstrap.pypa.io/get-pip.py -o "$GPUENV/get-pip.py"
"$GPUENV/bin/python" "$GPUENV/get-pip.py" --no-cache-dir pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 packaging==26.2
"$GPUENV/bin/python" -m pip install --no-cache-dir -r "$RELEASE/requirements/gpu.txt"
"$GPUENV/bin/python" -m pip install --no-cache-dir --no-build-isolation --no-deps "chumpy @ https://github.com/mattloper/chumpy/archive/580566eafc9ac68b2614b64d6f7aaa84eebb70da.zip"
"$GPUENV/bin/python" -m pip check
$ON_GPU "$GPUENV/bin/python" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect (about 7 minutes, 7.4 GB; the CUDA libraries are most of it):

```expect
No broken requirements found.
2.9.1+cu128 True
```

`False` instead of `True`: the command ran on a node without a usable GPU or with an empty `CUDA_VISIBLE_DEVICES`: run it through your GPU prefix (`ON_GPU`, section 2); if it stays `False` on a GPU node with a driver of 570 or newer: **STOP**. An error while installing: the usual causes are a network that blocks `download.pytorch.org` or `data.pyg.org` (build on a node with internet), a full disk, or a Python that is not 3.10 (SETUP.md "Troubleshooting"); report the last 30 lines of the output if you cannot fix it without changing a pinned file.

```bash
source "$BASE/env.sh"
"$ENV/bin/python" "$RELEASE/code/wf.py" assets verify --parts mano
"$ENV/bin/python" "$RELEASE/code/wf.py" assets link-hands
"$GPUENV/bin/python" "$RELEASE/code/wf.py" assets fetch-hug-source
"$GPUENV/bin/python" "$RELEASE/code/wf.py" assets fetch-hug1
```

Expect: the first prints a JSON whose last line is `"pass": true` (a note that `bundle` is `null` is normal); the second a JSON with `"action": "linked"`; the third (needs github.com, once; it clones the public HUG source at the pinned commit) a JSON with `"pass": true` and `"problems": []`; the fourth (needs huggingface.co, once; it downloads the public HUG1 weights and checks their sha256) a JSON with `"pass": true`:

```expect
"pass": true
"action": "linked"
"problems": []
```

## 7. Verify the machine

Run both on the kind of node you will compute on. The first needs no GPU; the second does. They write a report folder each; keep the printed text (`tee`) too. On a SLURM login node run them inside an allocation (the first with `srun -p <partition> --cpus-per-task=4 --time=01:00:00 --pty bash`, then the block; the second through `ON_GPU`, section 2).

```bash
source "$BASE/env.sh"
nice -n 19 "$ENV/bin/python" "$RELEASE/code/wf.py" verify --levels L0,L1 --workers 4 --out "$BASE/verify_cpu" 2>&1 | tee "$BASE/verify_cpu.log"
```

Expect the last line (about 3.5 minutes with 4 workers; the seconds depend on your CPU):

```expect
VERIFY PASS (<N> checks, <S> s)
```

Above it every `L1` line must say `pass_bitwise` (AMD Zen 2 and Zen 3 CPUs). On another CPU a line may say `pass_near` (acceptable: every label equal, tiny differences in a diagnostic: tell us your CPU) but `labels_only` or `FAIL` is not a pass: **STOP**, send the folder `$BASE/verify_cpu` (section 15), do not run batches on this machine.

```bash
source "$BASE/env.sh"
$ON_GPU "$GPUENV/bin/python" "$RELEASE/code/wf.py" verify --levels L0,L2,L3 --out "$BASE/verify_gpu" 2>&1 | tee "$BASE/verify_gpu.log"
```

Expect the last line (30 to 60 seconds on an RTX A6000, longer if the card is shared):

```expect
VERIFY PASS (<N> checks, <S> s)
```

The `L2 decode` lines say `PASS_TOLERANCE max |d params99| <x>`: the decode of six pictures differs from ours by GPU rounding; anything below 1e-3 passes (we measured 8.2e-6 on an RTX A6000 against our RTX 3080). The `L3 T4_mini_training` line says `PASS_TOLERANCE` (20 training calls within 2 % of ours). A `FAIL` anywhere: **STOP**.

| Level | What it checks | Pass line |
|---|---|---|
| L0 | every file of the release and of the unpacked assets against the sha256 lists | `PASS (<N> files)` |
| L1 | our reference grasps (200 from HUG-Bench, 220 of the dumbbell census) rolled on your CPUs and compared with our stored results | `pass_bitwise` (or `pass_near`) |
| L2 | the decode of six pictures on your GPU against ours | `PASS_TOLERANCE` |
| L3 | two exactness tests of the training code and a 20-call training against our reference | `PASS_BITWISE`, `PASS_TOLERANCE` |

## 8. The host fingerprint

```bash
source "$BASE/env.sh"
$ON_GPU "$GPUENV/bin/python" "$RELEASE/code/wf.py" fingerprint --out "$BASE/host_fingerprint.json"
ls "$BASE/verify_cpu/verify_report.json" "$BASE/verify_gpu/verify_report.json"
```

Expect: a JSON with your CPU model, `numerical_host_class` (`amd_zen2_zen3_bitwise` on AMD Zen 2 and Zen 3, otherwise `other_tolerance`) and the GPU models and driver of the node (it lists every GPU of the node, whichever you use), then the two report files:

```expect
"numerical_host_class": "<class>"
verify_cpu/verify_report.json
verify_gpu/verify_report.json
```

## 9. Send us the first report, then wait for our go

This block writes the report text for you. Set `WHO` to the name we know you by (the Drive folder `returns/<name>/`).

```bash
source "$BASE/env.sh"
export WHO=yourname      # CHANGE
"$ENV/bin/python" - "$BASE" "$RELEASE" "$WHO" <<'EOF' | tee "$BASE/report_setup.txt"
import json, subprocess, sys
base, rel, who = sys.argv[1:4]
fp = json.load(open(f"{base}/host_fingerprint.json"))
h, r = fp["host"], fp["release"]
def last(path):
    try:
        return open(path).read().strip().splitlines()[-1]
    except OSError:
        return "MISSING"
git = lambda *a: subprocess.run(["git", "-C", rel, *a], capture_output=True, text=True).stdout.strip()
print(f"REPORT setup from {who}")
print("release commit:", git("rev-parse", "HEAD"), "| SHA256SUMS digest:", r["sha256sums_sha256"][:16], "| dirty:", r["git"].get("dirty"))
print("host:", h.get("hostname"), "|", h.get("os"), "| kernel", h.get("kernel"), "| python", h.get("python"))
print("CPU:", h["cpu"].get("model_name"), "| logical cpus", h["cpu"].get("logical_cpus"), "| family", h["cpu"].get("family"), "model", h["cpu"].get("model"), "| class:", fp["numerical_host_class"])
for g in h.get("gpus") or []:
    print("GPU:", g.get("name"), "| driver", g.get("driver"), "|", g.get("memory"))
print("verify cpu:", last(f"{base}/verify_cpu.log"))
print("verify gpu:", last(f"{base}/verify_gpu.log"))
print("files to send: host_fingerprint.json, verify_cpu/verify_report.json, verify_gpu/verify_report.json")
print("anomalies: (write here anything unusual: a retry, a warning, a workaround, a step that took much longer than stated)")
EOF
```

Expect:

```expect
REPORT setup from <name>
verify cpu: VERIFY PASS (<N> checks, <S> s)
verify gpu: VERIFY PASS (<N> checks, <S> s)
```

Upload `host_fingerprint.json`, `verify_cpu/verify_report.json`, `verify_gpu/verify_report.json` and `report_setup.txt` to the Drive folder `returns/<name>/setup/` (section 14), and message us the text of the report. **Then wait for our go.** We confirm your host and send the batch `verification_batch_1a` with its sha256; do not run any batch before that (rule 2).

## 10. Receive the batch and check that it is for this release

The human puts the archive `verification_batch_1a.tar` into `$DOWNLOADS`. Its sha256 is in our message.

```bash
source "$BASE/env.sh"
mkdir -p "$BASE/batches" "$BASE/results"
sha256sum "$DOWNLOADS/verification_batch_1a.tar"
tar -xf "$DOWNLOADS/verification_batch_1a.tar" -C "$BASE/batches"
export BATCH=$BASE/batches/verification_batch_1a
export RESULTS=$BASE/results/verification_batch_1a
cat "$BATCH/INSTRUCTIONS.txt"
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch run "$BATCH" --out "$RESULTS" --cpu-python "$ENV/bin/python" --dry | grep -E '"(batch|release_ok)"'
```

Expect:

```expect
Batch verification_batch_1a: <N> groups, issued for release wuji_floating_20261120
"release_ok": true
```

The printed `sha256` must equal the one in our message (a transfer error otherwise: download the archive again). `"release_ok": false`, or an error saying `this batch cannot run on this copy of the release` or `the batch folder differs from its SHA256SUMS`: **STOP** (your checkout is not the release the batch was issued for, or the archive is damaged: `git fetch --tags && git checkout wf-20261123-r7a`, unpack the archive again, repeat; if it persists, report). `INSTRUCTIONS.txt` lists the groups: one group is one experiment (for this batch: three grasp priors decoded on your GPU for 164 held-out views and rolled under five wrist conditions, plus a replay of 3,936 of our stored jobs). Our projection for all groups: about 0.1 GPU-hours and 13 CPU core-hours. `--dry` runs nothing.

## 11. Run the batch

Choose **one** way and use it until the batch is finished (rule 3). On a machine you log in to and run on directly (workstation, or a GPU node you hold): way A. On a SLURM login node: way B. A cluster without SLURM: way C. A new terminal needs `export BATCH=$BASE/batches/verification_batch_1a RESULTS=$BASE/results/verification_batch_1a` again (section 10).

**Way A: one machine, one or several GPUs.** One group runs per GPU slot at a time; each GPU step gets its own `CUDA_VISIBLE_DEVICES`. Set the two CHANGE lines: `GPUS` are the ids of the cards this job may use (`nvidia-smi -L`; a repeated id such as `0,0` runs two jobs on one card, enough memory is needed), `WORKERS` the CPU cores it may use for rollouts (the default is all cores divided by the GPU slots: set it on a shared node). The command runs in the background and appends to a log, so that you can watch it and run the same block again after an interruption.

```bash
source "$BASE/env.sh"
export GPUS=0,1,2,3      # CHANGE
export WORKERS=16        # CHANGE
nohup "$GPUENV/bin/python" "$RELEASE/code/wf.py" batch run "$BATCH" --out "$RESULTS" --gpus "$GPUS" --workers "$WORKERS" --cpu-python "$ENV/bin/python" >> "$BASE/run_batch.log" 2>&1 &
echo $! > "$BASE/run_batch.pid"
sleep 20
tail -n 4 "$BASE/run_batch.log"
```

Expect (the first group's first step starts within seconds):

```expect
<group>: <step>: start
```

**Way B: SLURM, one array task per group.** Choose the values from your cluster: `sinfo -h -o "%P|%G|%l|%c|%m"` lists partitions, their GPUs (`%G`), time limits, cores and memory; `sacctmgr -nP show assoc user=$USER format=account,partition,qos` lists your accounts and QOS. Rules of thumb for this batch: one GPU per task (`--gpus-per-task 1`, or `--gres gpu:<type>:1` if your cluster names GPUs by type), `--cpus` 8 to 16 (these are the rollout workers), `--time` at least 6:00:00 (one group takes about 30 to 60 minutes with 8 cores after its decode), `--mem` 32G or more. All tasks of one batch must run on **the same CPU family and GPU model** (rule 3): on a mixed partition add `--constraint <feature of one node type>` (or `--nodelist`, or the GPU type in `--gres`); `--account`, `--qos`, `--array-parallel N` and `--setup 'module load cuda/12.8'` as your cluster needs. Read the `#SBATCH` lines before you submit.

```bash
source "$BASE/env.sh"
export PARTITION=gpu     # CHANGE
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch slurm "$BATCH" --out "$RESULTS" --partition "$PARTITION" --gpus-per-task 1 --cpus 8 --time 06:00:00 --mem 64G --gpu-python "$GPUENV/bin/python" --cpu-python "$ENV/bin/python"
grep '^#SBATCH' "$RESULTS/run_batch.sbatch"
sbatch "$RESULTS/run_batch.sbatch"
squeue -u "$USER"
```

Expect: a JSON with `"array_tasks": <N>` and a `submit` line, the `#SBATCH` lines you chose, then a job id from `sbatch` and a `squeue` listing. `Invalid generic resource (gres) specification`: your cluster names GPUs differently: use `--gres gpu:<type>:1` instead of `--gpus-per-task 1`.

**Way C: no SLURM, a plain shell loop.** It writes `run_batch.sh`; use `GPUS` as in way A.

```bash
source "$BASE/env.sh"
export GPUS=0,1,2,3      # CHANGE
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch shell "$BATCH" --out "$RESULTS" --gpus "$GPUS" --gpu-python "$GPUENV/bin/python" --cpu-python "$ENV/bin/python"
nohup bash "$RESULTS/run_batch.sh" > "$RESULTS/run_batch.log" 2>&1 &
echo $! > "$BASE/run_batch.pid"
```

### Watch it

```bash
source "$BASE/env.sh"
until ! kill -0 "$(cat "$BASE/run_batch.pid")" 2>/dev/null; do sleep 30; done
tail -n 25 "$BASE/run_batch.log"
```

With way B watch `squeue -u "$USER"` instead (the array is finished when it is gone) and read `$RESULTS/*/logs/`. Each finished step prints `<group>: <step>: ok (<seconds> s)`. A step that fails prints its error and its folder moves to `$RESULTS/_failed/`: look at `$RESULTS/<group>/logs/<step>.err` and `<step>.failed.json`.

### If the run stops (time limit, node failure, Ctrl-C, reboot)

Run the same block again, unchanged (way C: `bash "$RESULTS/run_batch.sh"` again): every finished step is skipped and the unfinished step runs again; the results are identical to an uninterrupted run (checked on a stand-in batch: a SIGTERM and a SIGKILL in the middle of a decode, then the same command again, gave rollout results bit-identical to an uninterrupted run).
After a clean stop (SIGTERM, a SLURM time limit, Ctrl-C) the interrupted step leaves nothing behind. After a hard stop (node failure, `kill -9`, reboot) the run finds the stale `RUNNING.lock` of the dead process on the same host and removes it, moves the unfinished step folder to `$RESULTS/_failed/<group>__<step>__<time>` (kept as evidence) and runs the step again.
With SLURM: `"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch status "$BATCH" --out "$RESULTS" --array` prints the group numbers still to run; submit `sbatch --array=<that list> "$RESULTS/run_batch.sbatch"`.
A message `group <g> is being run by another process` (a lock left by another node, or a run that is still alive): first make sure that nothing of this batch is running anywhere; ask us before you pass `--force-unlock`.
The same machine type applies after a restart (rule 3): if the node came back as another type, stop and report.

### Is it complete?

```bash
source "$BASE/env.sh"
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch status "$BATCH" --out "$RESULTS" | "$ENV/bin/python" -c 'import json,sys; d=json.load(sys.stdin); print("done", d["done"], "of", d["of"]); sys.exit(0 if d["done"] == d["of"] else 1)'
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch status "$BATCH" --out "$RESULTS" --array
ls "$RESULTS/_failed" 2>/dev/null || echo "no _failed folder"
```

Expect: `done <N> of <N>` (the same number; the exit code is 0 only then), an empty line for `--array` (no group left), and either `no _failed folder` or the folders of steps that were interrupted and re-run: list them in the report.

## 12. Pack the results

Only when every group is done. `--note` is free text for us.

```bash
source "$BASE/env.sh"
export NOTE="cluster: CHANGE; partition: CHANGE; run mode: A; anything unusual: none"      # CHANGE
"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch pack "$BATCH" --out "$RESULTS" --note "$NOTE"
ls -l "${RESULTS}_pack.tar"
sha256sum "${RESULTS}_pack.tar"
```

Expect:

```expect
"canary_check": "passed"
"archive_sha256": "<H>"
"not_packed": []
```

`batch pack` collects what we need (the results, their jobs files with the known-answer jobs, the step stamps, the training logs; not the 600 MB checkpoints), runs `pack-results` on it (README section 9.3), checks itself and writes one archive `${RESULTS}_pack.tar`. An error saying a group is not done: finish the batch (section 11). An error about a missing known-answer job or a jobs file: **STOP** and report (rule 5: do not repair). Do not use `--partial` unless we ask.

## 13. The second report

```bash
source "$BASE/env.sh"
export WHO=${WHO:-yourname}      # CHANGE: the name we know you by
"$ENV/bin/python" - "$BASE" "$RESULTS" "$WHO" "$RELEASE" <<'EOF' | tee "$BASE/report_batch.txt"
import collections, glob, hashlib, json, os, sys
base, res, who, rel = sys.argv[1:5]
sys.path.insert(0, rel + "/code")
from wf_lib import util
pack = res + "_pack.tar"
sha = hashlib.sha256(open(pack, "rb").read()).hexdigest()
tot, kinds, hosts, first, lastt = collections.defaultdict(float), collections.Counter(), set(), None, None
cpus, gpus = collections.Counter(), collections.Counter()
for p in glob.glob(res + "/*/*/STAMP.json"):
    d = json.load(open(p))
    tot[d["group"]] += d["wall_s"]
    kinds[d["status"]] += 1
    host, c = d["host"], d["host"]["cpu"]
    names = [g.get("name") for g in host.get("gpus") or []]
    hosts.add((host.get("hostname"), c.get("model_name"), tuple(sorted(set(names)))))
    first = min(filter(None, [first, d["started"]]))
    lastt = max(filter(None, [lastt, d["finished"]]))
    if d["kind"] == "roll":
        cpus["AMD Zen 2/3" if util.numerical_host_class(c) == "amd_zen2_zen3_bitwise" else f"{c.get('vendor')} family {c.get('family')} model {c.get('model')}"] += 1
    if d["kind"] in ("decode", "train", "readout"):
        slot = str(d.get("gpu"))
        gpus[names[int(slot)] if slot.isdigit() and int(slot) < len(names) else (" + ".join(sorted(set(names))) or "unknown")] += 1
print(f"REPORT batch from {who}")
print("batch:", os.path.basename(res), "| pack:", os.path.basename(pack), "| bytes:", os.path.getsize(pack), "| sha256:", sha)
print("steps by status:", dict(kinds), "| first start:", first, "| last finish:", lastt)
for g, s in sorted(tot.items()):
    print(f"step time {g}: {s:.0f} s")
for hname, cpu, gp in sorted(hosts, key=str):
    print("ran on:", hname, "|", cpu, "|", ", ".join(gp) or "no GPU")
print("rollout steps by CPU class:", dict(cpus), "| decode, training and read-out steps by GPU model:", dict(gpus))
print("OK: one machine type" if len(cpus) <= 1 and len(gpus) <= 1 else "STOP: this pack mixes machine types")
print("interrupted and re-run steps (_failed):", sorted(os.listdir(res + "/_failed")) if os.path.isdir(res + "/_failed") else "none")
print("anomalies: (write here anything unusual)")
EOF
```

Expect:

```expect
REPORT batch from <name>
OK: one machine type
```

`STOP: this pack mixes machine types` (more than one CPU class among the rollout steps or more than one GPU model among the decode, training and read-out steps; every AMD Zen 2 and Zen 3 part counts as one CPU class): **STOP** (rule 3), tell us, do not upload the pack.

## 14. Return the pack

Ask the human to upload (or do it with the rclone remote they gave you), then message us the report text.

```text
returns/<your name>/setup/                  host_fingerprint.json, verify_cpu/verify_report.json, verify_gpu/verify_report.json, report_setup.txt        (section 9)
returns/<your name>/<YYYYMMDD>/             verification_batch_1a_pack.tar, report_batch.txt                                                            (sections 12 and 13)

rclone copy "$BASE/results/verification_batch_1a_pack.tar" "<remote>:<shared folder>/returns/<your name>/<YYYYMMDD>/" --progress
```

The upload date folder is the day of the upload as `YYYYMMDD` (create it if it is missing; one folder per upload). Tell us the archive name and its sha256 (the report has it). We check the numbers (README section 9.4) and tell you the result before we send anything else.

## 15. What to send when something fails

Send, to `returns/<your name>/failed/<YYYYMMDD>/` and in a message: the full output of the command that failed; for a verify failure the whole report folder (`$BASE/verify_cpu` or `$BASE/verify_gpu`) and the `.log` next to it; for a batch failure the folder `$RESULTS/<group>/logs/` and the `*.failed.json` files; `host_fingerprint.json`; the output of `git -C "$RELEASE" status --short` (must be empty) and of `df -h "$BASE"`. Do not send credentials, tokens or the MANO file. Then wait.

## 16. The do-not list, in one place

- Do not edit, patch, rebuild or "fix" anything in `$RELEASE`, `$ASSETS` or the batch folder, and do not install a package that the requirements files do not pin.
- Do not run a batch before our go (section 9), and do not start another batch until we have told you the result of this one.
- Do not mix machines in one pack: one CPU family, one GPU model, one results folder; never move a results folder to another machine and continue.
- Do not re-run a finished job, delete or rename a results or `_failed` folder, or use `--partial` or `--force-unlock` unless we ask.
- Do not filter, reorder, subsample, repair or edit rows of a jobs or results file; the batch has known-answer jobs among them and a pack fails if one is missing or altered.
- Do not upload anything but the files of sections 9, 14 and 15. Do not share the MANO file with anyone or put it on a shared or public disk (it is under the user's own licence); the other asset parts are passed on only under their licences, with their attribution (`ASSET_TERMS.txt`).
- Do not guess when an output differs from what is expected: stop at the step, send the report, wait.
