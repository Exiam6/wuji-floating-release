"""CPU rollouts of predicted grasps in the HUG-Bench evaluator (MuJoCo 3.10.0), from a jobs npz to labels.

One rollout is single-threaded: a predicted MANO grasp (wrist pose + 15 finger rotations, for the robot hands also the 21 hand
landmarks) is executed by the evaluator's floating wrist: approach, finger curl, settle, a 0.5 m lift; the native label is
success when the object no longer touches the table at t = 4.0 s in a valid simulation. No GPU is used anywhere.

Three protocol families, one command set (`--protocol NAME`; the protocol file is code/protocols/NAME.json):
  mano    mano-1111-p1, mano-1111-p2 (the wrist carries the hand's weight), mano-1111-p2s-dzm1, mano-1111-p2-dzm1 / -dzm2 / -dzp1
          (the commanded wrist height offset by k free sags)
  hands   sharpa-float-p1 / -p2, wuji1hr-float-p1 / -p2 (a robot hand retargeted from the landmarks by the multihand retargeter)
  wuji1   wuji1-float-p1 (Wuji Hand 1 in the frozen October evaluator, glove retargeter of wuji-retargeting)

Commands (code/wf.py rollout ...):
  verify   level W1: the protocol file's own digest; every file digest of the protocol (evaluator files, MANO model, code, hand
           files, retargeter files); the evaluator commit marker; the MuJoCo version; the release folder against SHA256SUMS and
           the `core` asset part against its manifest. Prints a JSON report; exit 1 on any mismatch.
  run      verify, then roll every job of --jobs with --workers processes and write --out (outside the release and the asset
           root, never overwritten): per job the key and the record fields of tools/float/records.METRICS, the protocol stamp and
           run_info. MuJoCo's own log (MUJOCO_LOG.TXT) goes to a temporary working folder and is counted in run_info.
  compare  levels W2 / W3: --results against --expected: same keys in the same order, equal native labels, and every other common
           field reported. status pass_bitwise = labels and every field equal; pass_near = every label equal and at most 2 % of the
           jobs (at least one) differ in a diagnostic, by at most 1e-3 in the field's unit (floating-point noise of another CPU model
           in a rare rollout; accepted only when comparing with results stored on another host, never at levels L1 / L1h);
           labels_only = labels equal but more or larger differences (this CPU computes slightly different physics: treat it as a
           separate numerical host and tell us); fail = a label differs.
  check    load --jobs against the asset root without rolling anything.
  slice    pick --n rows spread evenly over a batch (and over its stored results) to spot-check it in minutes.

Jobs npz (one row per job; the fields of a RolloutPool job):
  key (n,) str, unique        returned unchanged
  sim_assets_dir (n,) str     object folder relative to <assets>/assets, e.g. hugbench60_assets/val/<group>/<object>/sim_assets
  body_name (n,) str          optional (mano family): object body in object.xml; empty = the folder above sim_assets
  T_base_camera (n,4,4) f8    camera pose of the input (OpenCV optical axes)
  T_camera_wrist (n,4,4) f8   HUG's wrist pose in the camera frame
  finger_aa (n,15,3) f8       finger axis-angles
  landmarks_camera (n,21,3)   HUG's landmarks (hands and wuji1 families: the retargeter's input)
  settled_qpos7 (n,7) f8      settled object pose x y z qw qx qy qz in the bench world; a row of NaN = the evaluator's own initialisation
  bench_table_z (n,) f8       optional, default 0
A job whose wrist pose, finger angles or landmarks are not finite is not run: it is recorded as a failure with zero diagnostics
(stage invalid_input).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import resource
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import RELEASE_NAME, manifest, paths
from .util import cpu_model, emit, host_fingerprint, loadavg, now, results_digest, sha256_file, versions

CODE = paths.CODE
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from tools.float import collab_native as native  # noqa: E402
from tools.float import protocol as base_protocol  # noqa: E402
from tools.float import records, rollout  # noqa: E402

for _m in (native, base_protocol, records, rollout):
    if CODE not in Path(_m.__file__).resolve().parents:
        raise SystemExit(f"{_m.__name__} was imported from {_m.__file__}, not from this release's {CODE}")

MANO, HANDS, WUJI1, WUJI22 = "mano", "hands", "wuji1", "wuji22"     # r8f: wuji22 = the floating Wuji Hand 2.2 (tools/float_wuji22, asset part `hands22`)
INT_FIELDS = ("success_native", "success_strict_v1", "sim_valid", "stage_code", "prelift_fingers_n", "final_fingers_n", "final_thumb",
              "final_palm", "prelift_finger_mask", "final_finger_mask", "init_contacts", "prelift_contacts", "final_contacts",
              "lift_min_contacts", "badqacc_warnings", "n_steps", "fit_nfev")
HAND_EXTRA_FIELDS = ("tipfit_rms_mm", "fit_rms_all_mm", "fit_nfev", "mount_check_mm", "mount_check_deg", "prelift_wrist_err_mm",
                     "prelift_wrist_sag_mm", "prelift_track_err_rad")
NOT_COMPARED = {"key", "protocol_name", "protocol_digest", "run_info", "source", "unit", "cluster", "sample", "t_rollout_s"}
NEAR_MAX_ABS = 1e-3            # pass_near: the largest difference of any non-label field (in the field's own unit: mm, m, counts)
NEAR_MAX_JOB_FRACTION = 0.02   # pass_near: the share of jobs that may differ in any field (at least one job)
if not (set(INT_FIELDS) - {"fit_nfev"}) <= set(records.METRICS):
    raise SystemExit("records.METRICS changed: update INT_FIELDS")


def protocol_names() -> list:
    return sorted(p.stem for p in (CODE / "protocols").glob("*.json"))


def family_of(name: str) -> str:
    path = CODE / "protocols" / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"wf: no protocol {name!r}; available: {', '.join(protocol_names())}")
    if name == "wuji1-float-p1":
        return WUJI1
    if name.startswith("wuji22-float-"):
        return WUJI22
    return HANDS if "hand_name" in json.loads(path.read_text()) else MANO


# ------------------------------------------------------------------------------------------------ W1
def _sha_or_none(path: Path):
    return sha256_file(path) if path.is_file() else None


def verify(name: str, layout: paths.Layout, release_check: bool = True) -> dict:
    """Level W1. report['pass'] is False on any mismatch."""
    fam = family_of(name)
    problems, files = [], {}
    try:
        if fam == HANDS:
            from tools.float_hands import protocol_hand as P
            hand, version = P.parse_name(name)
            proto = P.load(hand, root=CODE, version=version)
            stamp = P.stamp(proto)
        elif fam == WUJI1:
            from tools.float_wuji import protocol_wuji as P
            proto = P.load()
            stamp = P.stamp(proto)
        elif fam == WUJI22:
            from tools.float_wuji22 import protocol22 as P
            version = name.rsplit("-", 1)[1]
            proto = P.load(version, root=CODE)
            stamp = P.stamp(proto)
        else:
            proto = base_protocol.load(name)
            stamp = base_protocol.stamp(proto)
        bproto = proto if fam == MANO else base_protocol.load(proto["base"]["name"])
    except Exception as e:                                       # noqa: BLE001 (any load failure is a failed gate)
        return {"pass": False, "protocol_name": name, "problems": [f"protocol: {type(e).__name__}: {e}"], "host": socket.gethostname()}
    if fam != MANO and bproto["digest"] != proto["base"]["digest"]:
        problems.append(f"base protocol digest {bproto['digest']} != {proto['base']['digest']}")
    for rel, want in sorted(bproto["files"].items()):
        path = (layout.mano / rel) if rel.startswith(base_protocol.MANO_FILE_PREFIX) else (layout.evaluator / rel)
        got = _sha_or_none(path)
        files[rel] = got == want
        if got != want and not (rel.startswith(base_protocol.MANO_FILE_PREFIX) and "mano" in manifest.EXTERNAL):      # the MANO model: the asset check below explains it
            problems.append(f"evaluator file {rel}: {got} != {want}")
    code_files = {"adapter": bproto["adapter"]}
    if fam == MANO:
        for block in ("scene", "wrist", "wrist_offset"):
            if block in proto:
                code_files[block] = proto[block]
        used = (proto.get("wrist_offset", {}).get("parameters") or {}).get("wristload")
        if used:
            code_files["wrist_offset.wristload"] = used
    else:
        for rel, want in proto["code"].items():
            code_files[rel] = {"path": rel, "sha256": want}
        if "wrist" in proto:
            code_files["wrist"] = proto["wrist"]
    code_ok = {}
    for block, entry in code_files.items():
        got = _sha_or_none(CODE / entry["path"])
        code_ok[block] = got == entry["sha256"]
        if got != entry["sha256"]:
            problems.append(f"code {block} {entry['path']}: {got} != {entry['sha256']}")
    marker = layout.evaluator / base_protocol.SOURCE_COMMIT_FILE
    commit = marker.read_text() if marker.is_file() else None
    if commit != bproto["evaluator"]["commit"] + "\n":
        problems.append(f"evaluator/SOURCE_COMMIT holds {commit!r}, the protocol needs {bproto['evaluator']['commit']}")
    extra = {}
    if fam == HANDS:
        hand_files = P.hand_files(hand, root=CODE)
        if hand_files != proto["hand"]:
            problems.append(f"hand model files differ from the protocol ({len(set(hand_files.items()) ^ set(proto['hand'].items()))} entries); "
                            f"run `wf.py assets link-hands` after unpacking the `hands` asset part")
        rt = {rel: _sha_or_none(CODE / rel) for rel in P.RETARGETER_FILES}
        if rt != proto["retargeter"]["files"]:
            problems.append("retargeter files differ from the protocol: " + ", ".join(r for r in rt if rt[r] != proto["retargeter"]["files"].get(r)))
        extra = {"hand": hand, "version": version, "hand_files": len(hand_files), "retargeter_files": len(rt)}
    elif fam == WUJI1:
        hand_files = P.hand_files()
        if hand_files != proto["hand"]:
            problems.append(f"Wuji hand model files differ from the protocol ({len(set(hand_files.items()) ^ set(proto['hand'].items()))} entries); "
                            f"run `wf.py assets link-hands` after unpacking the `hands` asset part")
        man = layout.retargeter / "tree_manifest.txt"
        if not man.is_file():
            problems.append("retargeter/tree_manifest.txt is missing (asset part `hands`)")
        else:
            import hashlib
            if hashlib.sha256(man.read_bytes()).hexdigest() != proto["retargeter"]["tree_digest"]:
                problems.append("tree_manifest.txt does not hash to the protocol's retargeter tree digest")
            lines = dict(line.split("  ", 1)[::-1] for line in man.read_text().splitlines() if line.strip())
            if len(lines) != proto["retargeter"]["tree_files"]:
                problems.append(f"tree_manifest.txt lists {len(lines)} files, the protocol {proto['retargeter']['tree_files']}")
            shipped = {p.relative_to(layout.retargeter).as_posix(): sha256_file(p) for p in sorted(layout.retargeter.rglob("*")) if p.is_file()}
            shipped.pop("tree_manifest.txt", None)
            bad = sorted(rel for rel, got in shipped.items() if lines.get(rel) != got)
            if bad:
                problems.append(f"{len(bad)} retargeter files are not in the protocol's tree or differ from it, e.g. {bad[:3]}")
            if sha256_file(layout.retargeter / P.RT_CONFIG) != proto["retargeter"]["config_sha256"]:
                problems.append("retargeter config differs from the protocol")
            extra = {"retargeter_files_shipped": len(shipped)}
    elif fam == WUJI22:
        hand_files = P.hand_files(CODE) if (CODE / P.SRC).is_dir() else {}
        if hand_files != proto["hand"]:
            problems.append(f"Wuji Hand 2.2 model files differ from the protocol ({len(set(hand_files.items()) ^ set(proto['hand'].items()))} entries); "
                            f"unpack the `hands22` asset part into the asset root and run `wf.py assets link-hands`")
        rt = {rel: _sha_or_none(CODE / rel) for rel in P.RETARGETER_FILES}
        if rt != proto["retargeter"]["files"]:
            problems.append("retargeter files differ from the protocol: " + ", ".join(r for r in rt if rt[r] != proto["retargeter"]["files"].get(r)))
        if not problems:
            if P.drive_parameters(version) != proto["drive"]:
                problems.append("the drive parameters of tools/float_wuji22 differ from the protocol's")
            from tools.float_wuji22 import model22
            chk = model22.check()
            if not chk["pass"]:
                problems.append(f"the stored Wuji Hand 2.2 MJCF is not a fresh conversion of its inputs: {chk}")
        extra = {"hand": P.HAND, "version": version, "hand_files": len(hand_files), "retargeter_files": len(rt)}
    import mujoco
    if mujoco.__version__ != bproto["evaluator"]["mujoco"]:
        problems.append(f"mujoco {mujoco.__version__} != {bproto['evaluator']['mujoco']}")
    release = manifest.check_release() if release_check else {"pass": True, "skipped": True}
    if not release["pass"]:
        problems += [f"release folder: {p}" for p in release["problems"]]
    parts = ["core", "mano"] + ({MANO: [], WUJI22: ["hands22"]}.get(fam, ["hands"]))
    assets = manifest.check_parts(layout.root, parts)
    for part, rep in assets.items():
        if not rep["pass"]:
            problems += [f"asset part {part}: {p}" for p in rep["problems"]]
    return {"pass": not problems, "protocol": stamp, "protocol_name": name, "family": fam, "evaluator_commit": bproto["evaluator"]["commit"],
            "protocol_files_ok": sum(files.values()), "protocol_files": len(files), "code_files": code_ok, "mujoco": mujoco.__version__,
            "sha256sums": {"sha256": release.get("sha256sums_sha256"), "files": release.get("files")},
            "assets": {p: {k: v for k, v in r.items() if k in ("pass", "files", "bytes", "manifest_sha256")} for p, r in assets.items()},
            "base_sha256sums": {"sha256": (assets.get("core") or {}).get("manifest_sha256")},
            "asset_root": str(layout.root), "release": str(paths.RELEASE), "host": socket.gethostname(), "time": now(), **extra,
            "problems": problems}


# ------------------------------------------------------------------------------------------------ pools
ARENA_MEMORY = "128M"       # MuJoCo's arena of one scene. The default (16 MB for these scenes) overflows on grasps of objects with about 120 or more collision pieces
                            # ("mj_stackAlloc: out of memory"). The size does not enter the physics: the same jobs give bit-identical results with it (census_cmd README, arena).


def _enlarge_arena() -> None:
    # every scene compiled in a roll worker gets <size memory=ARENA_MEMORY/> unless its XML sets a size itself
    import mujoco
    orig = mujoco.MjModel.from_xml_string

    def with_arena(xml, assets=None):
        if "<size" not in xml and "<mujoco" in xml:
            i = xml.index(">", xml.index("<mujoco")) + 1
            xml = xml[:i] + f'<size memory="{ARENA_MEMORY}"/>' + xml[i:]
        return orig(xml) if assets is None else orig(xml, assets)
    mujoco.MjModel.from_xml_string = staticmethod(with_arena)


def _init_mano_worker(shim, evaluator, mano, nice, trace, wrist, wrist_offset):
    _enlarge_arena()
    rollout._init(shim, evaluator, mano, nice, trace, wrist, wrist_offset)
    root = Path(evaluator).resolve()
    for mod in (rollout._H.SP, rollout._H.SC, rollout._H.EP):
        if root not in Path(mod.__file__).resolve().parents:
            raise RuntimeError(f"{mod.__name__} was imported from {mod.__file__}, not from the asset root's {root}")


class ManoPool(rollout.RolloutPool):
    """tools/float/rollout.RolloutPool on the asset root's evaluator and MANO model (`run` and the job function are unchanged;
    only the git-checkout verification is replaced by `verify`)."""

    def __init__(self, workers: int, layout: paths.Layout, name: str, *, nice: int = 19, trace: bool = True):
        self.proto = base_protocol.load(name)
        self.stamp = base_protocol.stamp(self.proto)
        self.workers = int(workers)
        self._tmp = tempfile.TemporaryDirectory(prefix="wf_roll_")
        self.work_dir = Path(self._tmp.name)
        shim = self.work_dir / "shim"
        native.make_shim(shim, layout.evaluator)
        self.wrist = self.proto.get("wrist")
        self.wrist_offset = self.proto.get("wrist_offset")
        self._ex = cf.ProcessPoolExecutor(max_workers=self.workers, initializer=_init_mano_worker,
                                          initargs=(str(shim), str(layout.evaluator), str(layout.mano), int(nice), bool(trace),
                                                    self.wrist, self.wrist_offset))

    def close(self):
        try:
            super().close()
        finally:
            self._tmp.cleanup()


class HandPoolBase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self._ex.shutdown(wait=True)
        finally:
            self._tmp.cleanup()


def make_hand_pool(workers: int, layout: paths.Layout, name: str, nice: int):
    from tools.float_hands import protocol_hand as P
    from tools.float_hands import rollout_hand

    class HandPool(HandPoolBase):
        run = rollout_hand.HandPool.run

        def __init__(self):
            self._tmp = tempfile.TemporaryDirectory(prefix="wf_roll_hand_")
            work = Path(self._tmp.name)
            shim = work / "shim"
            native.make_shim(shim, layout.evaluator)
            os.environ["FLOAT_EVALUATOR_CHECKOUT"] = str(layout.evaluator)      # hand_eval: the MANO shape file
            os.environ["FLOAT_MANO_ASSETS"] = str(layout.mano)                  # hand_eval: the MANO layer
            hand, version = P.parse_name(name)
            proto = P.load(hand, root=CODE, version=version)
            self.stamp = P.stamp(proto)
            self._ex = cf.ProcessPoolExecutor(max_workers=int(workers), initializer=rollout_hand._init,
                                              initargs=(str(shim), str(work), hand, int(nice), P.gravity_compensation(proto), P.wrist_rule(proto)))

    return HandPool()


def make_wuji22_pool(workers: int, layout: paths.Layout, name: str, nice: int):
    """r8f: tools/float_wuji22/rollout22's workers (HandHarness22: tools/float_hands/rollout_hand.run_job with the Hand 2.2), its `run` unchanged."""
    from tools.float_wuji22 import protocol22 as P
    from tools.float_wuji22 import rollout22

    class Wuji22Pool(HandPoolBase):
        run = rollout22.HandPool22.run

        def __init__(self):
            self._tmp = tempfile.TemporaryDirectory(prefix="wf_roll_wuji22_")
            work = Path(self._tmp.name)
            shim = work / "shim"
            native.make_shim(shim, layout.evaluator)
            os.environ["FLOAT_EVALUATOR_CHECKOUT"] = str(layout.evaluator)      # hand_eval: the MANO shape file
            os.environ["FLOAT_MANO_ASSETS"] = str(layout.mano)                  # hand_eval: the MANO layer
            proto = P.load(name.rsplit("-", 1)[1], root=CODE)
            self.stamp = P.stamp(proto)
            self._ex = cf.ProcessPoolExecutor(max_workers=int(workers), initializer=rollout22._init22,
                                              initargs=(str(shim), str(work), int(nice), P.gravity_compensation(proto), P.wrist_rule(proto)))

    return Wuji22Pool()


def make_wuji_pool(workers: int, layout: paths.Layout, name: str, nice: int):
    from tools.float_wuji import protocol_wuji as P
    from tools.float_wuji import rollout_wuji

    class WujiPool(HandPoolBase):
        run = rollout_wuji.WujiPool.run

        def __init__(self):
            self._tmp = tempfile.TemporaryDirectory(prefix="wf_roll_wuji_")
            work = Path(self._tmp.name)
            shim = work / "shim"
            native.make_shim(shim, layout.evaluator)
            os.environ["FLOAT_EVALUATOR_CHECKOUT"] = str(layout.evaluator)
            os.environ["FLOAT_MANO_ASSETS"] = str(layout.mano)
            os.environ["FLOAT_WUJI_RETARGETING"] = str(layout.retargeter)
            self.stamp = P.stamp(P.load())
            self._ex = cf.ProcessPoolExecutor(max_workers=int(workers), initializer=rollout_wuji._init,
                                              initargs=(str(shim), str(layout.retargeter), str(work), int(nice)))

    return WujiPool()


# ------------------------------------------------------------------------------------------------ jobs
def load_jobs(path, objects: Path, fam: str) -> list:
    """Every array is read from the npz once and every job gets its own copies of its rows."""
    with np.load(path, allow_pickle=False) as Z:
        J = {f: np.array(Z[f]) for f in Z.files}
    need = ["key", "sim_assets_dir", "T_base_camera", "T_camera_wrist", "finger_aa", "settled_qpos7"] + (["landmarks_camera"] if fam != MANO else [])
    if [f for f in need if f not in J]:
        raise SystemExit(f"{path}: missing fields {[f for f in need if f not in J]}")
    n = len(J["key"])
    shapes = {"sim_assets_dir": (n,), "T_base_camera": (n, 4, 4), "T_camera_wrist": (n, 4, 4), "settled_qpos7": (n, 7)}
    if fam != MANO:
        shapes["landmarks_camera"] = (n, 21, 3)
    for f, s in shapes.items():
        if J[f].shape != s:
            raise SystemExit(f"{path}: {f} has shape {J[f].shape}, expected {s}")
    if J["finger_aa"].size != n * 45:
        raise SystemExit(f"{path}: finger_aa has shape {J['finger_aa'].shape}, expected ({n}, 15, 3)")
    if len(set(J["key"].tolist())) != n:
        raise SystemExit(f"{path}: keys are not unique")
    finger = np.asarray(J["finger_aa"], np.float64).reshape(n, 15, 3)
    body = J["body_name"] if "body_name" in J else np.array([""] * n)
    table = J["bench_table_z"] if "bench_table_z" in J else np.zeros(n)
    jobs = []
    for k in range(n):
        rel = Path(str(J["sim_assets_dir"][k]))
        if rel.is_absolute() or ".." in rel.parts:
            raise SystemExit(f"job {k}: sim_assets_dir must be relative to {objects} without '..': {rel}")
        d = objects / rel
        if not (d / "object.xml").is_file():
            raise SystemExit(f"job {k}: no object.xml in {d}")
        q = np.array(J["settled_qpos7"][k], np.float64)
        job = {"sim_assets_dir": str(d), "body_name": (str(body[k]) or None) if fam == MANO else None,
               "T_base_camera": np.array(J["T_base_camera"][k], np.float64), "T_camera_wrist": np.array(J["T_camera_wrist"][k], np.float64),
               "finger_aa": finger[k].copy(), "settled_qpos7": None if np.isnan(q).all() else q, "bench_table_z": float(table[k]),
               "key": str(J["key"][k])}
        if fam != MANO:
            job["landmarks_camera"] = np.array(J["landmarks_camera"][k], np.float64)
        if fam == MANO and "official_check" in J:
            job["official_check"] = bool(J["official_check"][k])
        jobs.append(job)
    return jobs


def column(ms: list, field: str) -> np.ndarray:
    vals = [m[field] for m in ms]
    if field in INT_FIELDS:
        if any(not isinstance(v, int) for v in vals):
            raise ValueError(f"{field}: a value is not an integer")
        return np.array(vals, np.int64)
    return np.array(vals, np.float64)


def mujoco_log(folder: Path) -> dict:
    """MuJoCo's warnings written to <folder>/MUJOCO_LOG.TXT during the run, counted by message body (the wall-clock time of each
    warning is dropped, so the summary is the same from run to run for the same jobs). It goes into run_info only."""
    path = folder / "MUJOCO_LOG.TXT"
    if not path.is_file():
        return {"warnings": 0, "messages": [], "bytes": 0}
    blocks, cur = [], None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(("WARNING ", "ERROR ")):
            if cur:
                blocks.append(cur)
            cur = [line.split(": ", 1)[1] if ": " in line else line]
        elif line.strip():
            if cur is not None:
                cur.append(line.strip())
        else:
            if cur:
                blocks.append(cur)
            cur = None
    if cur:
        blocks.append(cur)
    counts = {}
    for b in blocks:
        counts[" | ".join(b)] = counts.get(" | ".join(b), 0) + 1
    return {"warnings": len(blocks), "messages": [{"message": m, "count": n} for m, n in sorted(counts.items())][:20], "bytes": path.stat().st_size}


# ------------------------------------------------------------------------------------------------ run
def run(a, layout: paths.Layout) -> int:
    name, fam = a.protocol, family_of(a.protocol)
    jobs_path, out = Path(a.jobs).resolve(), Path(a.out).resolve()      # resolved before the working directory moves
    report = verify(name, layout)
    if not report["pass"]:
        emit(report)
        return 1
    for root in (paths.RELEASE, layout.root):
        if out == root or root in out.parents:
            raise SystemExit(f"--out must be outside {root}")
    if out.exists():
        raise SystemExit(f"{out} exists; results are never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(jobs_path, layout.objects, fam)
    if a.nice and os.nice(0) < a.nice:
        os.nice(a.nice - os.nice(0))
    need_finite = ("T_camera_wrist", "finger_aa") + (("landmarks_camera",) if fam != MANO else ())
    live = [k for k, j in enumerate(jobs) if all(np.isfinite(j[f]).all() for f in need_finite)]
    cwd0, scratch = os.getcwd(), tempfile.TemporaryDirectory(prefix="wf_roll_cwd_")
    os.chdir(scratch.name)                       # MuJoCo's MUJOCO_LOG.TXT goes here, never into the release (the workers inherit it)
    samples, stop = [], threading.Event()

    def sample():                                # load during the run, every 10 s
        while not stop.wait(10.0):
            samples.append([round(time.time() - t0, 1)] + loadavg())

    started, load_before, t0 = now(), loadavg(), time.time()
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        if fam == MANO:
            pool_cm = ManoPool(a.workers, layout, name, nice=a.nice)
        elif fam == HANDS:
            pool_cm = make_hand_pool(a.workers, layout, name, a.nice)
        elif fam == WUJI22:
            pool_cm = make_wuji22_pool(a.workers, layout, name, a.nice)
        else:
            pool_cm = make_wuji_pool(a.workers, layout, name, a.nice)
        with pool_cm as pool:
            stamp = dict(pool.stamp)
            res = pool.run([jobs[k] for k in live], chunk=a.chunk)
    finally:
        stop.set()
        thread.join()
        mlog = mujoco_log(Path(scratch.name))
        os.chdir(cwd0)
        scratch.cleanup()
    wall = time.time() - t0
    ru_self, ru_kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    rss = {"runner": round(ru_self.ru_maxrss / 1024, 1), "largest_worker": round(ru_kids.ru_maxrss / 1024, 1)}
    cpu = {"runner": round(ru_self.ru_utime + ru_self.ru_stime, 1), "workers": round(ru_kids.ru_utime + ru_kids.ru_stime, 1)}
    cpu["total"] = round(cpu["runner"] + cpu["workers"], 1)
    ms = [None] * len(jobs)
    october = np.zeros(len(jobs), np.int64)
    extra = {f: np.zeros(len(jobs), np.float64) for f in HAND_EXTRA_FIELDS} if fam in (HANDS, WUJI22) else {}
    for k, r in zip(live, res):
        if r.get("key") != jobs[k]["key"]:
            raise SystemExit(f"result order differs from job order at job {k}")
        ms[k] = records.metric_fields(r)
        october[k] = int(bool(r.get("october_success", False)))
        for f in extra:
            extra[f][k] = float(r.get(f, 0.0))
    for k in range(len(jobs)):
        if ms[k] is None:                        # never ran: a failure with zero diagnostics
            ms[k] = records.metric_fields({"success": False, "stage": "invalid_input"})
    fields = {f: column(ms, f) for f in records.METRICS}
    if "fit_nfev" in extra:
        extra["fit_nfev"] = extra["fit_nfev"].astype(np.int64)
    info = {"host": socket.gethostname(), "cpu": cpu_model(), "cpus": os.cpu_count(), "workers": int(a.workers), "nice": os.nice(0),
            "family": fam, "protocol_name": name, "jobs_file": str(jobs_path), "jobs_sha256": sha256_file(jobs_path), "jobs": len(jobs),
            "rolled": len(live), "not_finite": len(jobs) - len(live), "started": started, "finished": now(), "wall_s": round(wall, 2),
            "rollouts_per_s": round(len(live) / wall, 3) if wall > 0 else None, "loadavg_before": load_before, "loadavg_after": loadavg(),
            "load_samples_t_l1_l5_l15": samples, "max_rss_mib": rss, "cpu_s": cpu,
            "sha256sums_sha256": report["sha256sums"]["sha256"], "base_sha256sums_sha256": report["base_sha256sums"]["sha256"],
            "assets": report["assets"], "release": RELEASE_NAME, "protocol": stamp, "versions": versions(), "fingerprint": host_fingerprint(), "mujoco_log": mlog,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    arrays = dict(key=np.array([j["key"] for j in jobs]), protocol_name=np.array(stamp["name"]), protocol_digest=np.array(stamp["digest"]), **fields)
    if fam != MANO:
        arrays["october_success"] = october
    arrays.update(extra)
    info["results_digest"] = results_digest(arrays)              # over every array but run_info: a results file edited afterwards no longer matches its own record
    arrays["run_info"] = np.array(json.dumps(info, sort_keys=True))
    tmp = out.with_name(out.name + ".partial")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, out)
    emit({"out": str(out), "out_sha256": sha256_file(out), "jobs": len(jobs), "rolled": len(live),
          "success_native": int(fields["success_native"].sum()), "wall_s": info["wall_s"], "rollouts_per_s": info["rollouts_per_s"],
          "workers": info["workers"], "max_rss_mib": rss, "cpu_s": cpu, "host": info["host"], "protocol": stamp,
          "sha256sums_sha256": info["sha256sums_sha256"], "mujoco_warnings": mlog["warnings"]})
    return 0


# ------------------------------------------------------------------------------------------------ W2 / W3
def compare(a) -> int:
    R = np.load(a.results, allow_pickle=False)
    E = np.load(a.expected, allow_pickle=False)
    problems = []
    if R["key"].shape != E["key"].shape or not np.array_equal(R["key"], E["key"]):
        emit({"W2_pass": False, "pass": False, "status": "fail", "problems": ["keys differ (content or order)"]})
        return 1
    for f in ("protocol_name", "protocol_digest"):
        if f in E.files and str(R[f]) != str(E[f]):
            problems.append(f"{f}: {R[f]} != {E[f]}")
    same = R["success_native"] == E["success_native"]
    diag = {}
    for f in sorted((set(R.files) & set(E.files)) - NOT_COMPARED - {"success_native"}):
        x, y = np.asarray(R[f], np.float64), np.asarray(E[f], np.float64)
        diag[f] = {"equal": int((x == y).sum()), "max_abs_diff": float(np.max(np.abs(x - y))) if len(x) else 0.0}
    n = len(R["key"])
    all_equal = all(v["equal"] == n for v in diag.values())
    labels_ok = bool(same.all()) and not problems
    differ = np.zeros(n, bool)
    for f in diag:
        differ |= np.asarray(R[f], np.float64) != np.asarray(E[f], np.float64)
    n_differ, max_abs = int(differ.sum()), max((v["max_abs_diff"] for v in diag.values()), default=0.0)
    near = labels_ok and n_differ <= max(1, int(NEAR_MAX_JOB_FRACTION * n)) and max_abs <= NEAR_MAX_ABS
    status = "pass_bitwise" if (labels_ok and all_equal) else ("pass_near" if near else ("labels_only" if labels_ok else "fail"))
    report = {"W2_pass": status == "pass_bitwise", "pass": status == "pass_bitwise", "status": status, "jobs": n, "labels_equal": int(same.sum()),
              "jobs_differing_in_any_field": n_differ, "max_abs_diff_any_field": max_abs, "keys_differing": [str(k) for k in R["key"][differ]][:20],
              "near_rule": {"max_abs": NEAR_MAX_ABS, "max_job_fraction": NEAR_MAX_JOB_FRACTION},
              "labels_differ": [str(k) for k in R["key"][~same]][:50], "success_results": int(R["success_native"].sum()),
              "success_expected": int(E["success_native"].sum()), "diagnostics_all_equal": all_equal,
              "diagnostics_jobs_all_fields_equal": int(np.all([np.asarray(R[f], np.float64) == np.asarray(E[f], np.float64) for f in diag], axis=0).sum())
              if diag else None, "diagnostics": diag, "problems": problems,
              "results_sha256": sha256_file(a.results), "expected_sha256": sha256_file(a.expected)}
    emit(report)
    return 0 if status == "pass_bitwise" else 1


def slice_batch(a) -> int:
    """Pick --n rows spread evenly over a batch (jobs and, optionally, our stored results) so that it can be spot-checked in minutes."""
    out = Path(a.out_dir).resolve()
    if paths.RELEASE in out.parents or out == paths.RELEASE:
        raise SystemExit(f"--out-dir must be outside the release folder ({paths.RELEASE})")
    if out.exists():
        raise SystemExit(f"{out} exists")
    J = np.load(a.jobs, allow_pickle=False)
    n = len(J["key"])
    idx = np.unique(np.linspace(0, n - 1, min(int(a.n), n)).round().astype(int))
    out.mkdir(parents=True)

    def cut(Z, name):
        arrays = {f: (np.array(Z[f])[idx] if Z[f].ndim >= 1 and Z[f].shape[0] == n else np.array(Z[f])) for f in Z.files}
        np.savez_compressed(out / name, **arrays)

    cut(J, "jobs_slice.npz")
    if a.expected:
        E = np.load(a.expected, allow_pickle=False)
        if not np.array_equal(E["key"], J["key"]):
            raise SystemExit("the expected file's keys differ from the jobs file's (content or order)")
        cut(E, "expected_slice.npz")
    emit({"out_dir": str(out), "rows": int(len(idx)), "of": int(n), "first_keys": [str(k) for k in J["key"][idx][:3]]})
    return 0


def check(a, layout: paths.Layout) -> int:
    fam = family_of(a.protocol)
    jobs = load_jobs(a.jobs, layout.objects, fam)
    objects = sorted({Path(j["sim_assets_dir"]).parent.name for j in jobs})
    emit({"pass": True, "jobs": len(jobs), "objects": len(objects), "jobs_sha256": sha256_file(a.jobs), "protocol_name": a.protocol, "family": fam})
    return 0


def main(a, layout: paths.Layout) -> int:
    if a.cmd == "verify":
        report = verify(a.protocol, layout)
        emit(report)
        return 0 if report["pass"] else 1
    if a.cmd == "run":
        if a.workers < 1:
            raise SystemExit("--workers must be at least 1")
        return run(a, layout)
    if a.cmd == "compare":
        return compare(a)
    if a.cmd == "slice":
        return slice_batch(a)
    return check(a, layout)
