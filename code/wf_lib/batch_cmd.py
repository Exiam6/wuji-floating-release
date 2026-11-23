"""wf batch: run an issued batch of jobs on one machine or on a cluster, and send the results back (README sections 5 and 9).

An *issued batch* is a folder that we make (maintainers' tool issue_batch.py) and send you: `BATCH.json` (the release it was issued for and the groups of jobs), the jobs files and the
configs it needs, `SHA256SUMS`, `INSTRUCTIONS.txt`. Nothing in it is code. A **group** is one experiment, run as a chain of steps; the steps are fixed by the release (this file), not by the batch:

  decode_rollout   decode (GPU: `decode run`) -> jobs (`decode make-jobs`, adds the known-answer jobs) -> roll_<protocol> (CPU: `rollout run`) for every protocol of the group
  dsrl_readout     train (GPU: `dsrl train`) -> readout (GPU: `dsrl readout`, adds the known-answer jobs) -> roll_<protocol> for every protocol; one training can carry several read-outs
                   (`readouts` in BATCH.json: then readout_1, roll_1_<protocol>, readout_2, ...); without a read-out the group stops after the training
  jobs             roll_<protocol> of a jobs file that came with the batch (known-answer jobs already in it)
  cem_readout      (r7) the searched-constant arm: for every generation g of the cross-entropy search g<g>_decode (GPU: `cem decode`) -> roll_g<g>_<protocol> (CPU) -> g<g>_update (`cem update`); then
                   constants (`cem constants`) -> readout (GPU: `dsrl readout --train-out <constants> --arms const,...`) -> roll_<protocol> for every protocol of the read-out. Resumes at the generation
                   it stopped in (every step has its own STAMP.json)
  nn_readout       (r7) a read-out WITHOUT any training: readout (GPU: `dsrl readout --prior P --arms file --noise-file <noise_file of the group>`: one noise per test input, e.g. a retrieved one) -> roll_<protocol>.
                   The group names `noise_file` (a file of the batch folder), `prior` (default hug2) and, compulsory, `forbidden_ids_file` + `forbidden_ids_sha256` (the retrieval pool in the noise file is checked
                   against them). Resumable like every kind.

  wf.py batch run    BATCH --out RESULTS [--gpus 0,1,2,3] [--workers N] [--cpu-python PY] [--only G,..] [--task K] [--gpu-prefix "CMD {gpu} --"] [--nice 19] [--dry] [--force-unlock]
                     runs the groups in the order of BATCH.json, one group per GPU slot at a time (`--gpus 0,1` = two slots; a repeated id is two slots on one card); every GPU step gets
                     CUDA_VISIBLE_DEVICES=<its slot's id>; without --gpus there is one slot and the environment's CUDA_VISIBLE_DEVICES is left alone (SLURM sets it). Every step writes its own
                     folder RESULTS/<group>/<step>/ and, when it has finished, STAMP.json in it; run it again after an interruption and the steps that have a valid STAMP.json are skipped,
                     an unfinished step folder is moved to RESULTS/_failed/ (kept as evidence) and the step is run again. --dry prints the plan and runs nothing.
  wf.py batch status BATCH --out RESULTS [--array]       state of every group (--array: the group indices still to run, for `sbatch --array=`)
  wf.py batch slurm  BATCH --out RESULTS --partition P [--gpus-per-task 1] [--cpus 8] [--time HH:MM:SS] ...   writes a SLURM array script: one array task per group
  wf.py batch shell  BATCH --out RESULTS --gpus 0,1,2,3  writes a plain shell loop for a machine without SLURM: one background worker per GPU, groups dealt out round robin
  wf.py batch pack   BATCH --out RESULTS [--pack-dir DIR] [--note TEXT]   collects what we need from the finished groups (not the 600 MB endpoint.pt of a training), runs `pack-results` on
                     it and writes DIR.tar and its sha256: the one file to send back
Rollouts need the CPU environment of README 3.2: pass --cpu-python $ENV/bin/python (default: the interpreter that runs this command).
"""
from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

from . import RELEASE_NAME, paths, util
from .util import emit, fail, now, read_sums, sha256_file, tree_files

SCHEMA = "wf-batch-1"
STEP_SCHEMA = "wf-step-1"
SKIP_IN_PACK = {"endpoint.pt"}
DEFAULT_MAX_PACK_FILE_MB = 200
DEFAULT_ROLL_WORKERS = 8


# ---------------------------------------------------------------------------------------------------------------- the batch folder
def load_batch(folder) -> tuple:
    """(BATCH.json as a dict, the folder). The folder must equal its SHA256SUMS: an issued batch is never edited."""
    folder = Path(folder).resolve()
    f = folder / "BATCH.json"
    if not f.is_file():
        fail(f"{folder} is not an issued batch (there is no BATCH.json in it)")
    batch = json.loads(f.read_text())
    if batch.get("schema") != SCHEMA:
        fail(f"{f}: schema {batch.get('schema')!r}, this release reads {SCHEMA!r}")
    sums = folder / "SHA256SUMS"
    if not sums.is_file():
        fail(f"{folder} has no SHA256SUMS: unpack the archive we sent again")
    want = read_sums(sums)
    have = {p.relative_to(folder).as_posix() for p in tree_files(folder)} - {"SHA256SUMS"}
    bad = sorted(r for r, h in want.items() if r not in have or sha256_file(folder / r) != h)
    extra = sorted(have - set(want))
    if bad or extra:
        fail(f"the batch folder differs from its SHA256SUMS (changed or missing: {bad[:5]}; unlisted: {extra[:5]}): unpack the archive we sent again, an issued batch is never edited")
    return batch, folder


def check_release(batch: dict) -> dict:
    """Refuse to run a batch that was issued for another release than the one this command belongs to (every result carries the release's digest)."""
    from . import pack_cmd
    mine = pack_cmd.release_record()
    want = batch["release"]
    problems = []
    if mine["sha256sums_sha256"] != want["sha256sums_sha256"]:
        problems.append(f"the release digest (sha256 of SHA256SUMS) is {str(mine['sha256sums_sha256'])[:16]}, the batch was issued for {want['sha256sums_sha256'][:16]}")
    if mine["assets_json_sha256"] != want["assets_json_sha256"] or mine["asset_manifests"] != want["asset_manifests"]:
        problems.append("assets.json or an asset manifest differs from the one the batch was issued with")
    for g in batch["groups"]:
        for p in list(g.get("protocols", [])) + [q for ro in g.get("readouts", []) for q in ro.get("protocols", [])]:
            pj = paths.CODE / "protocols" / f"{p['name']}.json"
            if not pj.is_file() or json.loads(pj.read_text()).get("digest") != p["digest"]:
                problems.append(f"group {g['id']}: protocol {p['name']} is not the protocol the batch was issued with (digest differs or the protocol is unknown)")
    if problems:
        fail("this batch cannot run on this copy of the release:\n  - " + "\n  - ".join(problems) +
             f"\n  The batch was issued for release {want['name']}, git commit {str(want.get('git_commit'))[:10]}; yours is commit {str(mine['git'].get('commit'))[:10]} (dirty: {mine['git'].get('dirty')}). "
             "Update your clone to the tag named in the batch's INSTRUCTIONS.txt (git fetch --tags && git checkout <tag>) and run the setup check again (README section 4).")
    if want.get("git_commit") and mine["git"].get("commit") and want["git_commit"] != mine["git"]["commit"]:
        print(f"wf: note: the batch names git commit {want['git_commit'][:10]}, this checkout is {mine['git']['commit'][:10]}; the release digests are equal, so the files are the same", file=sys.stderr)
    return mine


# ---------------------------------------------------------------------------------------------------------------- groups -> steps
def find_input_set(rel: str, assets: Path) -> str:
    for base in (paths.RELEASE, assets):
        if (base / rel).is_file():
            return str(base / rel)
    fail(f"the input set {rel} is neither in the release nor in the asset root {assets}")


def expand(batch: dict, bdir: Path, out: Path, assets: Path, gpu_py: str, cpu_py: str, workers: int, nice) -> list:
    """The groups of a batch with their steps: [{id, index, kind, dir, steps: [{name, kind, gpu, dir, argv, prepare, protocol?}]}]."""
    W = str(paths.CODE / "wf.py")
    groups = []
    for i, g in enumerate(batch["groups"]):
        gid, kind = g["id"], g["kind"]
        gdir = out / gid
        protos = [p["name"] for p in g.get("protocols", [])]
        steps = []

        def rolls(jobs_path, plist=None, tag=""):
            return [dict(name=f"roll{tag}_{p}", kind="roll", gpu=False, protocol=p, prepare=True, dir=gdir / f"roll{tag}_{p}",
                         argv=[cpu_py, W, "rollout", "run", "--protocol", p, "--jobs", str(jobs_path), "--out", str(gdir / f"roll{tag}_{p}" / "results.npz"), "--workers", str(workers)]
                         + (["--nice", str(nice)] if nice is not None else [])) for p in (protos if plist is None else plist)]

        if kind == "decode_rollout":
            inp = find_input_set(g["input_set"], assets)
            steps.append(dict(name="decode", kind="decode", gpu=True, prepare=False, dir=gdir / "decode",
                              argv=[gpu_py, W, "decode", "run", "--prior", g["prior"], "--inputs", inp, "--draws", str(g["draws"]), "--base-seed", str(g["base_seed"])]
                              + (["--units", ",".join(g["units"])] if g.get("units") else []) + ["--out", str(gdir / "decode")]))
            if protos:
                steps.append(dict(name="jobs", kind="jobs", gpu=False, prepare=True, dir=gdir / "jobs",
                                  argv=[gpu_py, W, "decode", "make-jobs", "--decoded", str(gdir / "decode"), "--inputs", inp, "--protocols", ",".join(protos), "--out", str(gdir / "jobs" / "jobs.npz")]))
                steps += rolls(gdir / "jobs" / "jobs.npz")
        elif kind == "dsrl_readout":
            cfg = bdir / g["config"]
            steps.append(dict(name="train", kind="train", gpu=True, prepare=False, dir=gdir / "train",
                              argv=[gpu_py, W, "dsrl", "train", "--config", str(cfg), "--seed", str(g["seed"])] + (["--calls", str(g["calls"])] if g.get("calls") else []) + ["--out", str(gdir / "train")]))
            readouts = g.get("readouts")
            if readouts is None and g.get("arms"):                                      # one read-out given at the group level
                readouts = [{"arms": g["arms"], "sets": g.get("sets"), "input_set": g["input_set"], "protocols": g.get("protocols", [])}]
            for j, ro in enumerate(readouts or [], 1):
                tag = "" if len(readouts) == 1 else f"_{j}"
                rprotos = [q["name"] for q in ro.get("protocols", [])]
                if not rprotos:
                    fail(f"group {gid}: a read-out needs the protocols the read-out jobs will be rolled under (their known-answer jobs are made for them)")
                inp = find_input_set(ro["input_set"], assets)
                steps.append(dict(name=f"readout{tag}", kind="readout", gpu=True, prepare=False, dir=gdir / f"readout{tag}",
                                  argv=[gpu_py, W, "dsrl", "readout", "--train-out", str(gdir / "train"), "--inputs", inp, "--arms", ",".join(ro["arms"])]
                                  + (["--sets", ",".join(str(x) for x in ro["sets"])] if ro.get("sets") else []) + ["--protocols", ",".join(rprotos)] + ["--out", str(gdir / f"readout{tag}")]))
                steps += rolls(gdir / f"readout{tag}" / "jobs_readout.npz", rprotos, tag)
        elif kind == "jobs":
            steps += rolls(bdir / g["jobs_file"])
        elif kind == "cem_readout":
            cfg = bdir / g["config"]
            gens = int(g["generations"])
            if gens < 1 or not protos:
                fail(f"group {gid}: a cem_readout group needs `generations` and the protocol the search and the read-out are rolled under")
            for k in range(gens):
                dd, ud = gdir / f"g{k}_decode", gdir / f"g{k}_update"
                steps.append(dict(name=f"g{k}_decode", kind="cem_decode", gpu=True, prepare=False, dir=dd,
                                  argv=[gpu_py, W, "cem", "decode", "--config", str(cfg), "--seed", str(g["seed"]), "--gen", str(k)] + (["--prev", str(gdir / f"g{k - 1}_update")] if k else []) + ["--out", str(dd)]))
                steps += rolls(dd / "jobs.npz", protos[:1], f"_g{k}")
                steps.append(dict(name=f"g{k}_update", kind="cem_update", gpu=False, prepare=False, dir=ud,
                                  argv=[cpu_py, W, "cem", "update", "--config", str(cfg), "--seed", str(g["seed"]), "--gen", str(k), "--decoded", str(dd),
                                        "--results", str(gdir / f"roll_g{k}_{protos[0]}" / "results.npz"), "--out", str(ud)]))
            steps.append(dict(name="constants", kind="cem_constants", gpu=False, prepare=False, dir=gdir / "constants",
                              argv=[cpu_py, W, "cem", "constants", "--config", str(cfg), "--seed", str(g["seed"]), "--search", str(gdir), "--out", str(gdir / "constants")]))
            for j, ro in enumerate(g.get("readouts") or [], 1):
                tag = "" if len(g["readouts"]) == 1 else f"_{j}"
                rprotos = [q["name"] for q in ro.get("protocols", [])]
                if not rprotos:
                    fail(f"group {gid}: a read-out needs the protocols the read-out jobs will be rolled under (their known-answer jobs are made for them)")
                inp = find_input_set(ro["input_set"], assets)
                steps.append(dict(name=f"readout{tag}", kind="readout", gpu=True, prepare=False, dir=gdir / f"readout{tag}",
                                  argv=[gpu_py, W, "dsrl", "readout", "--train-out", str(gdir / "constants"), "--inputs", inp, "--arms", ",".join(ro["arms"]),
                                        "--const-file", str(gdir / "constants" / "constant_mean.npy")]
                                  + (["--sets", ",".join(str(x) for x in ro["sets"])] if ro.get("sets") else []) + ["--protocols", ",".join(rprotos)] + ["--out", str(gdir / f"readout{tag}")]))
                steps += rolls(gdir / f"readout{tag}" / "jobs_readout.npz", rprotos, tag)
        elif kind == "nn_readout":
            if not g.get("noise_file") or not g.get("forbidden_ids_file") or not g.get("forbidden_ids_sha256") or not g.get("readouts"):
                fail(f"group {gid}: an nn_readout group needs `noise_file`, `forbidden_ids_file`, `forbidden_ids_sha256` and `readouts`")
            for j, ro in enumerate(g["readouts"], 1):
                tag = "" if len(g["readouts"]) == 1 else f"_{j}"
                rprotos = [q["name"] for q in ro.get("protocols", [])]
                if not rprotos:
                    fail(f"group {gid}: a read-out needs the protocols the read-out jobs will be rolled under (their known-answer jobs are made for them)")
                inp = find_input_set(ro["input_set"], assets)
                steps.append(dict(name=f"readout{tag}", kind="readout", gpu=True, prepare=False, dir=gdir / f"readout{tag}",
                                  argv=[gpu_py, W, "dsrl", "readout", "--prior", g.get("prior", "hug2"), "--inputs", inp, "--arms", ",".join(ro["arms"]), "--noise-file", str(bdir / g["noise_file"]),
                                        "--forbidden-ids", str(bdir / g["forbidden_ids_file"]), "--forbidden-ids-sha256", g["forbidden_ids_sha256"]]
                                  + (["--sets", ",".join(str(x) for x in ro["sets"])] if ro.get("sets") else []) + ["--protocols", ",".join(rprotos)] + ["--out", str(gdir / f"readout{tag}")]))
                steps += rolls(gdir / f"readout{tag}" / "jobs_readout.npz", rprotos, tag)
        else:
            fail(f"group {gid}: unknown kind {kind!r} (this release knows decode_rollout, dsrl_readout, jobs, cem_readout, nn_readout)")
        groups.append(dict(id=gid, index=i, kind=kind, dir=gdir, steps=steps))
    return groups


# ---------------------------------------------------------------------------------------------------------------- state of a step
def step_state(st: dict, batch: dict, release_digest: str, recheck: bool = False) -> str:
    """'done' (valid STAMP.json, outputs as stamped), 'partial' (a folder without a valid stamp), 'pending'."""
    d = st["dir"]
    if not d.exists():
        return "pending"
    stamp = d / "STAMP.json"
    if not stamp.is_file():
        return "partial"
    try:
        s = json.loads(stamp.read_text())
    except ValueError:
        return "partial"
    if s.get("status") != "ok" or s.get("batch_id") != batch["batch_id"] or s.get("release_sha256sums_sha256") != release_digest:
        return "partial"
    for o in s.get("outputs", []):
        p = d / o["file"]
        if not p.is_file() or p.stat().st_size != o["bytes"] or (recheck and sha256_file(p) != o["sha256"]):
            return "partial"
    return "done"


def group_state(g: dict, batch: dict, release_digest: str) -> dict:
    states = [step_state(st, batch, release_digest) for st in g["steps"]]
    failed = (g["dir"] / "logs").is_dir() and any((g["dir"] / "logs").glob("*.failed.json"))
    done = sum(s == "done" for s in states)
    state = "done" if states and done == len(states) else ("failed" if failed else ("partial" if done or "partial" in states else "pending"))
    return {"group": g["id"], "index": g["index"], "state": state, "steps_done": done, "steps": len(states)}


def set_aside(out: Path, gid: str, name: str, d: Path) -> Path:
    dest = out / "_failed" / f"{gid}__{name}__{time.strftime('%Y%m%dT%H%M%S')}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(d), str(dest))
    return dest


# ---------------------------------------------------------------------------------------------------------------- running
class Context:
    def __init__(self, batch, bdir, out, assets, args, release):
        self.batch, self.bdir, self.out, self.assets, self.args = batch, bdir, out, assets, args
        self.release = release
        self.digest = release["sha256sums_sha256"]
        self.stop = threading.Event()
        self.procs = set()
        self.lock = threading.Lock()
        self.gpu_prefix = shlex.split(args.gpu_prefix) if getattr(args, "gpu_prefix", None) else []
        self.nice = args.nice

    def log(self, gid, msg):
        with self.lock:
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {gid}: {msg}", flush=True)


def _kill(proc, ctx):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < 20:
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def output_list(d: Path) -> list:
    return [{"file": p.relative_to(d).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in tree_files(d) if p.name != "STAMP.json"]


def run_step(ctx: Context, g: dict, st: dict, gpu) -> bool:
    out, gid, name, d = ctx.out, g["id"], st["name"], st["dir"]
    state = step_state(st, ctx.batch, ctx.digest, getattr(ctx.args, "recheck", False))
    if state == "done":
        ctx.log(gid, f"{name}: done before, skipped")
        return True
    if state == "partial":
        dest = set_aside(out, gid, name, d)
        ctx.log(gid, f"{name}: an unfinished folder was moved to {dest} (kept as evidence); running the step again")
    logs = g["dir"] / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{name}.failed.json").unlink(missing_ok=True)
    if st["prepare"]:
        d.mkdir(parents=True)
    env = dict(os.environ)
    env.update(WF_ASSETS=str(ctx.assets), PYTHONDONTWRITEBYTECODE="1")
    argv = list(st["argv"])
    nice = [] if ctx.nice is None or st["kind"] == "roll" else ["nice", "-n", str(ctx.nice)]
    if st["gpu"]:
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        prefix = [t.replace("{gpu}", str(gpu)) for t in ctx.gpu_prefix] if gpu is not None else []
        full = prefix + nice + argv
    else:
        full = nice + argv
    out_path = d / "results.run.json" if st["kind"] == "roll" else logs / f"{name}.out"
    err_path = logs / f"{name}.err"
    ctx.log(gid, f"{name}: start" + (f" on GPU {gpu}" if st["gpu"] and gpu is not None else ""))
    t0, started = time.time(), now()
    with open(out_path, "w") as fo, open(err_path, "w") as fe:
        try:
            proc = subprocess.Popen(full, env=env, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:                                                            # the command itself cannot start (a wrong --gpu-prefix, a missing interpreter)
            fe.write(f"cannot start {full[0]!r}: {e}\n")
            proc, rc = None, 127
        if proc is not None:
            with ctx.lock:
                ctx.procs.add(proc)
            try:
                while True:
                    try:
                        rc = proc.wait(timeout=1.0)
                        break
                    except subprocess.TimeoutExpired:
                        if ctx.stop.is_set():
                            _kill(proc, ctx)
                            rc = proc.wait()
                            break
            finally:
                with ctx.lock:
                    ctx.procs.discard(proc)
    wall = round(time.time() - t0, 1)
    if rc != 0:
        interrupted = ctx.stop.is_set()
        tail = err_path.read_text(errors="replace").splitlines()[-25:]
        (logs / f"{name}.failed.json").write_text(json.dumps({"group": gid, "step": name, "rc": rc, "interrupted": interrupted, "started": started, "finished": now(), "wall_s": wall,
                                                              "argv": full, "stderr_tail": tail}, indent=1) + "\n")
        ctx.log(gid, f"{name}: {'interrupted' if interrupted else 'FAILED'} (exit code {rc}, {wall} s); the last lines of {err_path}:")
        for line in tail[-8:]:
            ctx.log(gid, "    " + line)
        return False
    stamp = {"schema": STEP_SCHEMA, "status": "ok", "batch_id": ctx.batch["batch_id"], "group": gid, "step": name, "kind": st["kind"], "protocol": st.get("protocol"), "argv": full,
             "gpu": gpu if st["gpu"] else None, "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"), "started": started, "finished": now(), "wall_s": wall, "rc": rc,
             "release": ctx.release, "release_sha256sums_sha256": ctx.digest, "host": util.host_fingerprint(gpu=True), "runner": "wf.py batch run",
             "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME") if os.environ.get(k)} or None,
             "outputs": output_list(d)}
    (d / "STAMP.json").write_text(json.dumps(stamp, indent=1, default=str) + "\n")
    ctx.log(gid, f"{name}: ok ({wall} s)")
    return True


def take_lock(ctx: Context, g: dict):
    lock = g["dir"] / "RUNNING.lock"
    g["dir"].mkdir(parents=True, exist_ok=True)
    me = f"{socket.gethostname()} {os.getpid()} {now()}"
    if lock.exists():
        host, pid = (lock.read_text().split() + ["", ""])[:2]
        same_host_dead = host == socket.gethostname() and pid.isdigit() and not Path(f"/proc/{pid}").exists()
        if same_host_dead or getattr(ctx.args, "force_unlock", False):
            ctx.log(g["id"], f"stale lock of {lock.read_text().strip()} removed")
            lock.unlink()
        else:
            raise SystemExit(f"wf: group {g['id']} is being run by another process ({lock.read_text().strip()}). If that run died, remove {lock} or pass --force-unlock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, me.encode())
    os.close(fd)
    return lock


def run_group(ctx: Context, g: dict, gpu) -> dict:
    lock = take_lock(ctx, g)
    t0 = time.time()
    ok = True
    try:
        for st in g["steps"]:
            if ctx.stop.is_set():
                ok = False
                break
            if not run_step(ctx, g, st, gpu):
                ok = False
                break
    finally:
        lock.unlink(missing_ok=True)
    return {"group": g["id"], "ok": ok, "wall_s": round(time.time() - t0, 1)}


def select(groups: list, a) -> list:
    if getattr(a, "task", None) is not None:
        if not 0 <= a.task < len(groups):
            fail(f"--task {a.task}: the batch has {len(groups)} groups (0 to {len(groups) - 1})")
        return [groups[a.task]]
    if getattr(a, "only", None):
        want = [x for x in a.only.split(",") if x]
        missing = [x for x in want if x not in {g["id"] for g in groups}]
        if missing:
            fail(f"--only: no group {missing} in the batch (groups: {[g['id'] for g in groups]})")
        return [g for g in groups if g["id"] in want]
    return groups


def common_setup(a):
    batch, bdir = load_batch(a.batch)
    out = util.refuse_inside(a.out, paths.RELEASE)
    assets = paths.assets_root(a.assets)
    util.refuse_inside(a.out, assets)
    return batch, bdir, out, assets


def run(a) -> int:
    batch, bdir, out, assets = common_setup(a)
    release = check_release(batch)
    slots = [s for s in (a.gpus or "").split(",") if s.strip() != ""] or [None]
    workers = a.workers or int(os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 2) // len(slots)) or DEFAULT_ROLL_WORKERS)
    groups = select(expand(batch, bdir, out, assets, sys.executable, a.cpu_python or sys.executable, workers, a.nice), a)
    if a.dry:
        plan = []
        for k, g in enumerate(groups):
            gs = group_state(g, batch, release["sha256sums_sha256"])
            plan.append({"group": g["id"], "kind": g["kind"], "state": gs["state"], "slot_gpu": slots[k % len(slots)],
                         "steps": [{"step": st["name"], "state": step_state(st, batch, release["sha256sums_sha256"]), "gpu": st["gpu"], "command": shlex.join(st["argv"])} for st in g["steps"]]})
        emit({"batch": batch["batch_id"], "release_ok": True, "results": str(out), "gpu_slots": slots, "rollout_workers": workers, "groups": plan})
        return 0
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context(batch, bdir, out, assets, a, release)

    def on_signal(signum, frame):
        ctx.stop.set()
        print(f"wf: signal {signum}: stopping the running steps; run the same command again to resume", file=sys.stderr, flush=True)
    for sg in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sg, on_signal)
    q = queue.Queue()
    for g in groups:
        q.put(g)
    results, errors = {}, []

    def worker(gpu):
        while not ctx.stop.is_set():
            try:
                g = q.get_nowait()
            except queue.Empty:
                return
            try:
                results[g["id"]] = run_group(ctx, g, gpu)
            except SystemExit as e:
                errors.append(str(e))
                ctx.stop.set()

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in slots[:max(1, len(groups))]]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive():
            t.join(timeout=1.0)
    for e in errors:
        print(e, file=sys.stderr)
    states = [group_state(g, batch, release["sha256sums_sha256"]) for g in groups]
    emit({"batch": batch["batch_id"], "results": str(out), "wall_s": round(time.time() - t0, 1), "groups": states})
    return 0 if all(s["state"] == "done" for s in states) else 1


def status(a) -> int:
    batch, bdir, out, assets = common_setup(a)
    release = check_release(batch)
    groups = expand(batch, bdir, out, assets, sys.executable, sys.executable, 1, None)
    states = [group_state(g, batch, release["sha256sums_sha256"]) for g in groups]
    if a.array:
        print(",".join(str(s["index"]) for s in states if s["state"] != "done"))
        return 0
    emit({"batch": batch["batch_id"], "results": str(out), "groups": states, "done": sum(s["state"] == "done" for s in states), "of": len(states)})
    return 0


# ---------------------------------------------------------------------------------------------------------------- scripts for a cluster
def _env_lines(a, assets) -> list:
    return [f"export WF_ASSETS={shlex.quote(str(assets))}", "export PYTHONDONTWRITEBYTECODE=1"]


def slurm(a) -> int:
    batch, bdir, out, assets = common_setup(a)
    check_release(batch)
    n = len(batch["groups"])
    gpu_py, cpu_py = a.gpu_python or sys.executable, a.cpu_python or sys.executable
    L = ["#!/bin/bash", f"# SLURM array for the batch {batch['batch_id']}: one array task per group ({n} groups). Written by `wf.py batch slurm`; edit the setup lines for your cluster.",
         f"#SBATCH --job-name=wf-{batch['batch_id'][:40]}", f"#SBATCH --partition={a.partition}", f"#SBATCH --array=0-{n - 1}" + (f"%{a.array_parallel}" if a.array_parallel else ""),
         "#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={a.cpus}"]
    L.append(f"#SBATCH --gres={a.gres}" if a.gres else f"#SBATCH --gpus-per-task={a.gpus_per_task}")
    L += [f"#SBATCH --time={a.time}"]
    for flag, val in (("mem", a.mem), ("account", a.account), ("qos", a.qos), ("constraint", a.constraint)):
        if val:
            L.append(f"#SBATCH --{flag}={val}")
    L += [f"#SBATCH --output={out}/slurm/%x_%A_%a.out", "", "set -euo pipefail", *_env_lines(a, assets)]
    L += list(a.setup or [])
    L += ["", f"GPUPY={shlex.quote(gpu_py)}", f"CPUPY={shlex.quote(cpu_py)}", f"WF={shlex.quote(str(paths.CODE / 'wf.py'))}", f"BATCH={shlex.quote(str(bdir))}", f"RESULTS={shlex.quote(str(out))}", "",
          'echo "task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
          '$GPUPY $WF batch run "$BATCH" --out "$RESULTS" --task "$SLURM_ARRAY_TASK_ID" --workers "${SLURM_CPUS_PER_TASK:-8}" --cpu-python "$CPUPY" --force-unlock', "",
          "# After the array has finished: `$GPUPY $WF batch status $BATCH --out $RESULTS --array` prints the groups that are not done; run them again with",
          "#   sbatch --array=<that list> <this script>   (finished steps are skipped), then `$GPUPY $WF batch pack $BATCH --out $RESULTS`.", ""]
    text = "\n".join(L)
    if a.dry:
        print(text)
        return 0
    script = Path(a.script) if a.script else out / "run_batch.sbatch"
    (out / "slurm").mkdir(parents=True, exist_ok=True)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(text)
    script.chmod(0o755)
    emit({"script": str(script), "array_tasks": n, "submit": f"sbatch {script}", "note": "check the #SBATCH and setup lines against your cluster's documentation before submitting"})
    return 0


def shell(a) -> int:
    batch, bdir, out, assets = common_setup(a)
    check_release(batch)
    gpus = [s for s in a.gpus.split(",") if s.strip() != ""]
    if not gpus:
        fail("--gpus: list the GPU ids to use, e.g. --gpus 0,1,2,3")
    n = len(batch["groups"])
    gpu_py, cpu_py = a.gpu_python or sys.executable, a.cpu_python or sys.executable
    workers = a.workers or DEFAULT_ROLL_WORKERS
    prefix = f" --gpu-prefix {shlex.quote(a.gpu_prefix)}" if a.gpu_prefix else ""
    L = ["#!/bin/bash", f"# Plain loop for the batch {batch['batch_id']} on a machine without SLURM: one background worker per GPU, the {n} groups dealt out round robin.",
         "# Run it again after an interruption: finished steps are skipped. Written by `wf.py batch shell`.", "set -u", *_env_lines(a, assets), "",
         f"GPUPY={shlex.quote(gpu_py)}", f"CPUPY={shlex.quote(cpu_py)}", f"WF={shlex.quote(str(paths.CODE / 'wf.py'))}", f"BATCH={shlex.quote(str(bdir))}", f"RESULTS={shlex.quote(str(out))}", f"WORKERS={workers}", "",
         "gpu_worker() {", "  local gpu=$1; shift", "  for task in \"$@\"; do",
         f"    CUDA_VISIBLE_DEVICES=$gpu $GPUPY $WF batch run \"$BATCH\" --out \"$RESULTS\" --task $task --gpus $gpu --workers $WORKERS --cpu-python \"$CPUPY\"{prefix} || echo \"group $task did not finish (see $RESULTS)\"",
         "  done", "}", ""]
    for k, gpu in enumerate(gpus):
        mine = " ".join(str(t) for t in range(k, n, len(gpus)))
        if mine:
            L.append(f"gpu_worker {gpu} {mine} &")
    L += ["wait", f'$GPUPY $WF batch status "$BATCH" --out "$RESULTS"', f'echo "when every group says done: $GPUPY $WF batch pack $BATCH --out $RESULTS"', ""]
    text = "\n".join(L)
    if a.dry:
        print(text)
        return 0
    script = Path(a.script) if a.script else out / "run_batch.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(text)
    script.chmod(0o755)
    emit({"script": str(script), "groups": n, "gpus": gpus, "run": f"bash {script}"})
    return 0


# ---------------------------------------------------------------------------------------------------------------- the return
def pack(a) -> int:
    from . import pack_cmd
    batch, bdir, out, assets = common_setup(a)
    release = check_release(batch)
    groups = expand(batch, bdir, out, assets, sys.executable, sys.executable, 1, None)
    states = {g["id"]: group_state(g, batch, release["sha256sums_sha256"]) for g in groups}
    todo = [g for g in groups if states[g["id"]]["state"] == "done"]
    open_ = [g["id"] for g in groups if states[g["id"]]["state"] != "done"]
    if open_ and not a.partial:
        fail(f"groups not done: {open_}. Run `wf.py batch run` (or the SLURM array) again for them, look at {out}/<group>/logs/ if one keeps failing and send us that; "
             "`--partial` packs only the finished groups")
    pack_dir = Path(a.pack_dir) if a.pack_dir else out.parent / f"{out.name}_pack"
    util.refuse_inside(pack_dir, paths.RELEASE, assets)
    if pack_dir.exists():
        fail(f"{pack_dir} exists; a pack is never overwritten (use --pack-dir for a new folder)")
    pack_dir.mkdir(parents=True)
    max_bytes = int(a.max_file_mb) * 2 ** 20
    skipped = []
    for g in todo:
        for p in tree_files(g["dir"]):
            rel = p.relative_to(out)
            if p.name == "RUNNING.lock" or p.name in SKIP_IN_PACK or p.stat().st_size > max_bytes:
                if p.name != "RUNNING.lock":
                    skipped.append(rel.as_posix())
                continue
            dst = pack_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy2(p, dst)
    shutil.copy2(bdir / "BATCH.json", pack_dir / "BATCH.json")
    note = (a.note or "").strip() + f" | batch {batch['batch_id']}; groups packed {[g['id'] for g in todo]}; not packed {open_}; files left out (endpoint.pt of a training, files over {a.max_file_mb} MB): {len(skipped)}"
    ns = type("NS", (), {})()
    ns.folder, ns.note, ns.no_canaries, ns.assets = str(pack_dir), note, False, a.assets
    rc = pack_cmd.main(ns)
    if rc != 0:
        return rc
    tar_path = Path(str(pack_dir) + ".tar")
    with tarfile.open(tar_path, "w") as tf:
        tf.add(pack_dir, arcname=pack_dir.name)
    emit({"pack_folder": str(pack_dir), "archive": str(tar_path), "archive_sha256": sha256_file(tar_path), "archive_bytes": tar_path.stat().st_size, "groups": [g["id"] for g in todo], "not_packed": open_,
          "next": "upload the archive to returns/<your name>/<date>/ in the shared drive folder and send us its name and the sha256"})
    return 0


def fingerprint(a) -> int:
    """The host fingerprint file that users who run our batches send us before a batch: CPU, GPU, driver, versions, and the numerical class of the CPU."""
    from . import pack_cmd
    fp = util.host_fingerprint(gpu=not a.no_gpu)
    rec = {"schema": "wf-host-1", "created": now(), "release": pack_cmd.release_record(), "host": fp, "numerical_host_class": util.numerical_host_class(fp["cpu"])}
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    emit({"written": str(a.out), "cpu": fp["cpu"].get("model_name"), "numerical_host_class": rec["numerical_host_class"], "gpus": fp.get("gpus")})
    return 0


def main(a) -> int:
    return {"run": run, "status": status, "slurm": slurm, "shell": shell, "pack": pack}[a.cmd](a)
