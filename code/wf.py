#!/usr/bin/env python
"""wf: the command line of the wuji-floating release. Read ../README.md first.

  python code/wf.py assets verify|parts|link-hands|fetch-hug-source|fetch-dinov2|fetch-hug1   asset checks and setup
  python code/wf.py verify [--levels L0,L1,L2,L3]                                    the verification suite (run it first)
  python code/wf.py fingerprint --out FILE                                           the host fingerprint file we ask for before a batch (CPU, GPU, driver, versions)
  python code/wf.py batch run|status|slurm|shell|pack BATCH --out RESULTS            run an issued batch on one machine or a cluster and send the results back (README sections 5 and 9)
  python code/wf.py dsrl train|readout|check ...                                     job type 1: GPU DSRL-NA training and read-out
  python code/wf.py decode run|make-jobs|check ...                                   job type 2: GPU prior decode (HUG2 / HUG1 / 50k)
  python code/wf.py rollout verify|run|compare|check|slice ...                       job type 3: CPU rollouts, the evaluation that follows them (no GPU)
  python code/wf.py pack-results --folder DIR                                        SHA256SUMS + PACK.json + RETURN.json for sending results back (`batch pack` calls it)

The asset root (the folder that holds evaluator/, mano/, assets/, ckpt/ ...) is --assets PATH (a top-level option: write it right after
wf.py, before the command), or $WF_ASSETS, or wf_config.json.
Nothing is written inside this release folder: results go where you say with --out / --folder.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"          # the release folder stays exactly as shipped
sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wf.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default=None, help="asset root (default: $WF_ASSETS, then wf_config.json)")
    sub = ap.add_subparsers(dest="group", required=True)

    s = sub.add_parser("assets", help="asset checks and setup")
    ss = s.add_subparsers(dest="cmd", required=True)
    v = ss.add_parser("verify", help="check asset parts against asset_manifests/<part>.sha256 (the MANO model too: it is downloaded by the user, not shipped)")
    v.add_argument("--parts", default=None, help="comma list; `parts` lists the names (default: the parts present in the asset root, and mano)")
    ss.add_parser("parts", help="list the asset parts")
    ss.add_parser("link-hands", help="link <assets>/hand_models/assets into code/assets (part `hands`)")
    ss.add_parser("fetch-hug-source", help="git clone the public HUG repository at the pinned commit into <assets>/hug_source")
    ss.add_parser("fetch-dinov2", help="download the DINOv2 weights HUG loads, at the pinned revision (needs internet)")
    ss.add_parser("fetch-hug1", help="download the public HUG1 weights (Hugging Face kevinywu/hug) and check their sha256 (needs internet)")

    s = sub.add_parser("verify", help="the verification suite")
    s.add_argument("--levels", default="L0,L1", help="L0 files, L1 CPU rollouts, L1h hand protocols, L1b production batches, L2 decode (GPU), L3 DSRL (GPU); default L0,L1")
    s.add_argument("--workers", type=int, default=2, help="CPU worker processes for the L1 rollouts")
    s.add_argument("--out", required=True, help="folder for the verification records (outside the release folder)")
    s.add_argument("--protocols", default=None, help="L1 only: comma list of protocols to check (default: the MANO protocols)")
    s.add_argument("--parts", default=None, help="L0: comma list of asset parts to check (default: the parts the chosen levels need, or the parts present)")
    s.add_argument("--set", default="ref200,w3", help="L1 reference sets: ref200 (HUG-Bench grasps), w3 (dumbbell census), bench2000 (throughput set)")
    s.add_argument("--slice-n", type=int, default=100, help="L1b: rows per batch pair to spot-check")
    s.add_argument("--gpu", default=None, help="L2 / L3: CUDA device index to use (default: the first visible)")
    s.add_argument("--priors", default=None, help="L2: comma list of priors to check (default: those whose checkpoint is present)")

    s = sub.add_parser("rollout", help="CPU rollouts (job type 1)")
    ss = s.add_subparsers(dest="cmd", required=True)
    v = ss.add_parser("verify")
    v.add_argument("--protocol", required=True)
    r = ss.add_parser("run")
    r.add_argument("--protocol", required=True)
    r.add_argument("--jobs", required=True)
    r.add_argument("--out", required=True, help="results npz, outside the release and the asset root; never overwritten")
    r.add_argument("--workers", type=int, required=True)
    r.add_argument("--nice", type=int, default=19)
    r.add_argument("--chunk", type=int, default=64, help="jobs per task (one object per task)")
    c = ss.add_parser("compare")
    c.add_argument("--results", required=True)
    c.add_argument("--expected", required=True)
    k = ss.add_parser("check")
    k.add_argument("--protocol", required=True)
    k.add_argument("--jobs", required=True)
    sl = ss.add_parser("slice", help="pick rows spread evenly over a batch and over its stored results (spot check)")
    sl.add_argument("--jobs", required=True)
    sl.add_argument("--expected", default=None)
    sl.add_argument("--n", type=int, default=200)
    sl.add_argument("--out-dir", required=True)

    s = sub.add_parser("decode", help="GPU prior decode (job type 2)")
    ss = s.add_subparsers(dest="cmd", required=True)
    d = ss.add_parser("run", help="decode an input set with one prior: stored pictures x noise rows -> grasps")
    d.add_argument("--prior", required=True, choices=["hug2", "hug1", "dumbbell50k"], help="the grasp prior to decode with")
    d.add_argument("--inputs", required=True, help="input set json (inputs/input_sets/*.json or reference/decode/*.json)")
    d.add_argument("--out", required=True, help="output folder (new)")
    d.add_argument("--draws", type=int, default=32)
    d.add_argument("--base-seed", type=int, default=20261116)
    d.add_argument("--units", default=None, help="comma list of unit ids to decode (default: all of the input set)")
    d.add_argument("--batch-inputs", type=int, default=8)
    d.add_argument("--request-seed", type=int, default=0)
    d.add_argument("--mem-fraction", type=float, default=None, help="cap this process's share of the GPU memory (0-1)")
    d.add_argument("--store", default=None, help="input store folder (default: <assets>/inputs/input_store)")
    m = ss.add_parser("make-jobs", help="turn a decode output into a rollout jobs npz (CPU stage input)")
    m.add_argument("--decoded", required=True, help="the decode output folder")
    m.add_argument("--inputs", required=True)
    m.add_argument("--out", required=True, help="jobs npz (new)")
    m.add_argument("--store", default=None, help="input store folder (default: <assets>/inputs/input_store)")
    m.add_argument("--protocols", default=None, help="comma list of the protocols you will roll the jobs under: known-answer jobs (canaries) are added for each (default: the six MANO protocols)")
    m.add_argument("--no-canaries", action="store_true", help="debugging only: no known-answer jobs; `pack-results` refuses the results of such a file and our verifier rejects the pack")
    k = ss.add_parser("check", help="decode the reference inputs and compare with the stored reference decode")
    k.add_argument("--priors", default=None)
    k.add_argument("--out", required=True)

    s = sub.add_parser("dsrl", help="DSRL-NA training and read-out (job type 3)")
    ss = s.add_subparsers(dest="cmd", required=True)
    t = ss.add_parser("train", help="offline DSRL-NA on a prep folder")
    t.add_argument("--config", required=True, help="training config json (see configs/dsrl_ex299_hug1.json)")
    t.add_argument("--out", required=True, help="output folder (new)")
    t.add_argument("--seed", type=int, default=None, help="override the config's seed")
    t.add_argument("--calls", type=int, default=None, help="override offline_calls (tests only)")
    t.add_argument("--device", default="cuda")
    q = ss.add_parser("readout", help="decode the trained actor's noise at each input and write the rollout jobs")
    q.add_argument("--train-out", required=True, help="the folder `dsrl train` wrote (its config.json, seed included, is what the read-out uses)")
    q.add_argument("--config", default=None, help="optional: must describe the same experiment as the training run's config.json")
    q.add_argument("--protocols", default="mano-1111-p1", help="comma list of the protocols you will roll the read-out jobs under: known-answer jobs (canaries) are added for each")
    q.add_argument("--no-canaries", action="store_true", help="debugging only: no known-answer jobs; `pack-results` refuses the results of such a file and our verifier rejects the pack")
    q.add_argument("--inputs", required=True, help="input set json")
    q.add_argument("--out", required=True, help="output folder (new)")
    q.add_argument("--arms", default="own,stoch,normrand")
    q.add_argument("--sets", default=None, help="only units of these readout sets (comma list of 0 / 1 / 2: training inputs, unseen configurations, unseen objects)")
    q.add_argument("--store", default=None, help="input store folder (default: <assets>/inputs/input_store)")
    q.add_argument("--device", default="cuda")
    k = ss.add_parser("check", help="the port's exactness tests (CPU) and a short training run (GPU) against the reference")
    k.add_argument("--out", required=True)
    k.add_argument("--device", default="cuda")
    k.add_argument("--cpu-only", action="store_true", help="run only the CPU tests")
    k.add_argument("--write-expected", default=None, help="maintainers: write this run's mini-training record as the reference file")
    k.add_argument("--tf32", action="store_true", help="stress test: allow TF32 matrix products (a much larger numerical perturbation than another GPU model)")

    s = sub.add_parser("fingerprint", help="write the host fingerprint file (CPU, GPU, driver, versions, numerical class of the CPU) that we ask for before the first batch")
    s.add_argument("--out", required=True, help="the json file to write")
    s.add_argument("--no-gpu", action="store_true", help="a machine without a GPU")

    s = sub.add_parser("batch", help="run an issued batch (a folder we send) on one machine or a cluster, and send the results back")
    ss = s.add_subparsers(dest="cmd", required=True)
    b = ss.add_parser("run", help="run the groups of the batch in order; resumes after an interruption")
    b.add_argument("batch", help="the issued batch folder (BATCH.json inside)")
    b.add_argument("--out", required=True, help="results folder (outside the release and the asset root); one sub-folder per group")
    b.add_argument("--gpus", default=None, help="comma list of GPU ids, one group at a time per entry (0,1,2,3; a repeated id is two slots on one card). Default: one slot that keeps the environment's CUDA_VISIBLE_DEVICES (SLURM sets it)")
    b.add_argument("--workers", type=int, default=None, help="CPU workers of a rollout step (default: $SLURM_CPUS_PER_TASK, else the cores divided by the GPU slots)")
    b.add_argument("--cpu-python", default=None, help="the CPU environment's python for the rollout steps (README 3.2); default: this interpreter")
    b.add_argument("--only", default=None, help="comma list of group ids to run")
    b.add_argument("--task", type=int, default=None, help="run only the group with this index (SLURM_ARRAY_TASK_ID)")
    b.add_argument("--gpu-prefix", default=None, help='a command put in front of every GPU step, {gpu} replaced by the slot id, e.g. "/path/gpu_guard.sh {gpu} --" (a cluster whose GPUs are guarded or shared)')
    b.add_argument("--nice", type=int, default=19, help="nice level of the steps (default 19; pass 0 for none)")
    b.add_argument("--dry", action="store_true", help="print the plan (groups, steps, state, commands) and run nothing")
    b.add_argument("--force-unlock", action="store_true", help="take over a group whose RUNNING.lock was left by a run that died on another node")
    b.add_argument("--recheck", action="store_true", help="re-hash every output of a finished step before skipping it")
    b = ss.add_parser("status", help="state of every group of the batch in a results folder")
    b.add_argument("batch")
    b.add_argument("--out", required=True)
    b.add_argument("--array", action="store_true", help="print only the indices of the groups that are not done (for sbatch --array=)")
    b = ss.add_parser("slurm", help="write a SLURM array script: one array task per group")
    b.add_argument("batch")
    b.add_argument("--out", required=True)
    b.add_argument("--partition", required=True)
    b.add_argument("--gpus-per-task", type=int, default=1)
    b.add_argument("--gres", default=None, help="use --gres=<this> (e.g. gpu:a100:1) instead of --gpus-per-task")
    b.add_argument("--cpus", type=int, default=8, help="CPUs per task: the workers of the rollout steps")
    b.add_argument("--time", default="24:00:00")
    b.add_argument("--mem", default=None)
    b.add_argument("--account", default=None)
    b.add_argument("--qos", default=None)
    b.add_argument("--constraint", default=None)
    b.add_argument("--array-parallel", type=int, default=None, help="at most this many array tasks at once (%%N)")
    b.add_argument("--setup", action="append", default=[], help="a line to put before the run (module load ..., source ...); repeatable")
    b.add_argument("--gpu-python", default=None, help="the GPU environment's python (default: this interpreter)")
    b.add_argument("--cpu-python", default=None, help="the CPU environment's python (default: this interpreter)")
    b.add_argument("--script", default=None, help="where to write the script (default: <out>/run_batch.sbatch)")
    b.add_argument("--dry", action="store_true", help="print the script and write nothing")
    b = ss.add_parser("shell", help="write a plain shell loop for a machine without SLURM")
    b.add_argument("batch")
    b.add_argument("--out", required=True)
    b.add_argument("--gpus", required=True, help="comma list of GPU ids, e.g. 0,1,2,3")
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--gpu-python", default=None)
    b.add_argument("--cpu-python", default=None)
    b.add_argument("--gpu-prefix", default=None)
    b.add_argument("--script", default=None, help="default: <out>/run_batch.sh")
    b.add_argument("--dry", action="store_true", help="print the script and write nothing")
    b = ss.add_parser("pack", help="collect what we need from the finished groups, run pack-results on it, write the archive to send back")
    b.add_argument("batch")
    b.add_argument("--out", required=True)
    b.add_argument("--pack-dir", default=None, help="the pack folder to create (default: <out>_pack); the archive is <pack folder>.tar")
    b.add_argument("--note", default="", help="free text: which machine, anything we should know")
    b.add_argument("--max-file-mb", type=int, default=200, help="files above this size stay out of the pack (the 600 MB endpoint.pt of a training always does)")
    b.add_argument("--partial", action="store_true", help="pack the finished groups although others are not done")

    s = sub.add_parser("pack-results", help="write SHA256SUMS and RETURN.json into a results folder")
    s.add_argument("--folder", required=True)
    s.add_argument("--note", default="")
    s.add_argument("--no-canaries", action="store_true", help="pack results whose jobs file has no known-answer jobs (the pack records the waiver and our verifier rejects it)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from wf_lib import paths
    if a.group == "assets":
        from wf_lib import manifest
        from wf_lib.util import emit
        if a.cmd == "parts":
            emit({"parts": manifest.parts_available(), "manifests": str(paths.MANIFESTS)})
            return 0
        root = paths.assets_root(a.assets)
        if a.cmd == "verify":
            wanted = [p for p in a.parts.split(",") if p] if a.parts else sorted(set(manifest.present_parts(root)) | set(manifest.EXTERNAL))
            rep = manifest.check_parts(root, wanted)
            bundle = manifest.check_bundle(root)
            ok = all(r["pass"] for r in rep.values()) and bundle["pass"] is not False
            emit({"asset_root": str(root), "parts": rep, "bundle": bundle, "pass": ok})
            for part, r in rep.items():
                for prob in r["problems"]:
                    print(f"asset part {part}: {prob}", file=sys.stderr)
            return 0 if ok else 1
        rep = {"link-hands": manifest.link_hands, "fetch-hug-source": manifest.fetch_hug_source, "fetch-dinov2": manifest.fetch_dinov2,
               "fetch-hug1": manifest.fetch_hug1}[a.cmd](root)
        emit(rep)
        return 0 if rep.get("pass") else 1
    if a.group == "rollout":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only, one maths thread per process (set before numpy / torch load)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[k] = "1"
        from wf_lib import roll
        if a.cmd in ("compare", "slice"):
            return roll.main(a, None)
        return roll.main(a, paths.Layout(paths.assets_root(a.assets)))
    if a.group == "decode":
        from wf_lib import decode_cmd
        return decode_cmd.main(a)
    if a.group == "dsrl":
        from wf_lib import dsrl_cmd
        return dsrl_cmd.main(a)
    if a.group == "verify":
        from wf_lib import verify_cmd
        return verify_cmd.main(a)
    if a.group == "pack-results":
        from wf_lib import pack_cmd
        return pack_cmd.main(a)
    if a.group == "batch":
        from wf_lib import batch_cmd
        return batch_cmd.main(a)
    if a.group == "fingerprint":
        from wf_lib import batch_cmd
        return batch_cmd.fingerprint(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())
