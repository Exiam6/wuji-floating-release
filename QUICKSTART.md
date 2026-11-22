# Verification run: your checklist

One page, no explanations. Every command with its expected output and the failure handling: [QUICKSTART_AGENT.md](QUICKSTART_AGENT.md) (section numbers below; hand it to your agent). Long guide: [SETUP.md](SETUP.md). Chinese: [QUICKSTART_zh.md](QUICKSTART_zh.md).

**You receive:** (a) the public GitHub repository `Exiam6/wuji-floating-release` (tag `wf-20261122-r6`; no account needed); (b) a drive link: the folder `full/` (15 tarballs, 8.7 GB, with `SHA256SUMS`) and, if you run batches for the project, your upload folder `returns/<your name>/`; (c) for a batch issued to you: its archive (the first is `verification_batch_1a.tar`) with its sha256, sent after step 5. Steps 1 to 5 install and verify the release; step 6 is for an issued batch.
**Needs:** Linux x86-64, NVIDIA driver 570 or newer, CPython 3.10, 30 GB free on a local or scratch disk, internet during setup. Set `export BASE=<your folder>` and make `env.sh` (AGENT section 2).

1. **Clone at the tag.**
   `git clone --branch wf-20261122-r6 https://github.com/Exiam6/wuji-floating-release.git "$RELEASE"`
   `cd "$RELEASE" && sha256sum -c --quiet SHA256SUMS && echo "release intact"`
2. **Download `full/` into `$DOWNLOADS`, check, unpack** (15 lines `OK`, then `assets intact`).
   `cd "$DOWNLOADS" && sha256sum -c SHA256SUMS`
   `mkdir -p "$ASSETS" && for t in "$DOWNLOADS"/*.tar; do tar -xf "$t" -C "$ASSETS"; done`
   `cd "$ASSETS" && sha256sum -c --quiet "$RELEASE/asset_manifests/SHA256SUMS.all" && echo "assets intact"`
3. **MANO.** Register at https://mano.is.tue.mpg.de, download `mano_v1_2.zip` into `$DOWNLOADS`.
   `unzip -q "$DOWNLOADS/mano_v1_2.zip" -d "$DOWNLOADS"`
   `mkdir -p "$ASSETS/mano/models" && cp "$DOWNLOADS/mano_v1_2/models/MANO_RIGHT.pkl" "$ASSETS/mano/models/"`
   `sha256sum "$ASSETS/mano/models/MANO_RIGHT.pkl"` must print `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767`.
4. **Build the two environments** from `requirements/cpu.txt` (CPU, `$ENV`) and `requirements/gpu.txt` (GPU, `$GPUENV`): AGENT section 6; both end `No broken requirements found.` Then `"$ENV/bin/python" "$RELEASE/code/wf.py" assets link-hands` and `"$GPUENV/bin/python" "$RELEASE/code/wf.py" assets fetch-hug-source` and `"$GPUENV/bin/python" "$RELEASE/code/wf.py" assets fetch-hug1` (same section; the HUG1 weights are not in the tarballs).
5. **Verify, fingerprint, send, wait for our go** (AGENT sections 7 to 9).
   `nice -n 19 "$ENV/bin/python" "$RELEASE/code/wf.py" verify --levels L0,L1 --workers 4 --out "$BASE/verify_cpu"`
   `CUDA_VISIBLE_DEVICES=0 "$GPUENV/bin/python" "$RELEASE/code/wf.py" verify --levels L0,L2,L3 --out "$BASE/verify_gpu"`
   `CUDA_VISIBLE_DEVICES=0 "$GPUENV/bin/python" "$RELEASE/code/wf.py" fingerprint --out "$BASE/host_fingerprint.json"`
   Both verifies must end `VERIFY PASS`. Upload the files below to `returns/<your name>/setup/`.
6. **Run batch 1a, pack, return** (AGENT sections 10 to 14).
   `tar -xf "$DOWNLOADS/verification_batch_1a.tar" -C "$BASE/batches"`, then `export BATCH=$BASE/batches/verification_batch_1a RESULTS=$BASE/results/verification_batch_1a`
   `"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch run "$BATCH" --out "$RESULTS" --gpus 0,1,2,3 --cpu-python "$ENV/bin/python"` (SLURM: AGENT section 11, way B; after an interruption run the same command again)
   `"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch status "$BATCH" --out "$RESULTS"` must say `"done"` equal to `"of"`.
   `"$GPUENV/bin/python" "$RELEASE/code/wf.py" batch pack "$BATCH" --out "$RESULTS" --note "cluster, partition, anything unusual"` prints the archive's sha256.

| Send back | File | Where (Drive) |
|---|---|---|
| host fingerprint, both verify reports | `host_fingerprint.json`, `verify_cpu/verify_report.json`, `verify_gpu/verify_report.json` | `returns/<your name>/setup/` |
| batch results (numbers only) | `${RESULTS}_pack.tar`, plus a message with its sha256, your CPU model and GPU model | `returns/<your name>/<YYYYMMDD>/` |
| anything that failed | the whole report folder, `$RESULTS/<group>/logs/`, the output of the failed command | `returns/<your name>/failed/<YYYYMMDD>/` |

> **Rules.** 1. Change no file of the release, the assets or the batch: no code, no asset changes. 2. A verify failure, `labels_only`, `FAIL` or a hash that differs: **stop and send us the report**; no batch on that machine. 3. One pack, one machine: one CPU family and one GPU model; never mix machines or continue a results folder elsewhere. 4. Tell us your CPU model and GPU model (the fingerprint does). 5. Send numbers only: the pack, never code, checkpoints, the MANO file or credentials. The asset parts carry their own licences (`ASSET_TERMS.txt`); the MANO file is under your own licence and is never passed on. 6. No batch before our go; do not re-run a finished job unless we ask.
