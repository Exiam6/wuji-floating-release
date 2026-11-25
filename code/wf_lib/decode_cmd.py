"""GPU prior decode (job type 2): stored input pictures x noise rows -> predicted grasps, with HUG2, HUG1 or the dumbbell 50k prior.

Per input (a rendered RGB-D picture of an object, its condition pixel) the prior decodes `--draws` grasps from fixed noise rows:
row i is torch.randn((1, 99)) from a CPU generator seeded with derived_seed(base_seed, "fresh_hug_benchmark_noise", noise_key, i), so the
same noise rows serve every input and prior ("codebook" noise: sample j is the same noise everywhere). The computation is the
mano-1111 campaign's sampling service in process (tools/float/decode.py: centre crop, 4,096-point cloud, frozen scene encoding, Euler
integration, MANO decode). It needs CUDA: the encoder's farthest-point sampler is only seeded on the GPU.

Input set (json): {"schema": "wf-input-set-1", "units": [{"unit", "cluster", "unit_key", "asset", "noise_key"?}, ...]}
  unit_key   the picture: <store>/<key[:2]>/<key>.npz (rgb, depth_mm, camera_k, condition_uv, T_base_camera, settled_object_qpos)
             with <key>.json holding its sha256; the store is --store (default <assets>/inputs/input_store)
  asset      the object folder relative to <assets>/assets (hugbench60_assets/val/<group>/<object>/sim_assets)

Outputs (new folder --out):
  preds_<prior>.npz   unit, cluster, candidate_index, noise_seed, T_camera_wrist, mano_pose_aa, landmarks_camera, mano_shape,
                      mano_params99, noise, finite       (one row per unit x draw; the columns of the project's census files)
  decode.json         settings (checkpoint sha256, steps, torch, device), counts, seconds, convention check
`make-jobs` turns that folder into the jobs npz of the CPU stage (`wf.py rollout run`), keys "<prior>/<unit>#<draw>".
`check` decodes the reference pictures and compares with the stored reference decode (level L2 of the verification suite).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

from . import RELEASE_NAME, canary, manifest, paths
from . import util
from .util import emit, now, sha256_file, versions

SCHEMA = "wf-input-set-1"
NOISE_KEY = "codebook"
SHAPES = {"T_camera_wrist": (4, 4), "mano_pose_aa": (15, 3), "landmarks_camera": (21, 3), "mano_shape": (10,), "mano_params99": (99,),
          "noise": (99,), "finite": ()}
# tolerances of the cross-GPU decode check (level L2): the measured differences between two GPU models are below 2e-5 (README section 4, level L2)
TOL = {"mano_params99": 1e-3, "landmarks_camera_m": 1e-3, "wrist_translation_m": 1e-3, "wrist_rotation_max_abs": 1e-3, "mano_pose_aa_rad": 1e-3}


def load_input_set(path, store: Path, units=None) -> list:
    spec = json.loads(Path(path).read_text())
    if spec.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: not an input set (schema {spec.get('schema')!r}, expected {SCHEMA})")
    out = []
    keep = set(units) if units else None
    for u in spec["units"]:
        if keep is not None and u["unit"] not in keep:
            continue
        k = u["unit_key"]
        npz, js = store / k[:2] / f"{k}.npz", store / k[:2] / f"{k}.json"
        if not npz.is_file() or not js.is_file():
            raise SystemExit(f"unit {u['unit']}: picture {k} is not in the store {store}")
        row = json.loads(js.read_text())
        if sha256_file(npz) != row["case_file_sha256"]:
            raise SystemExit(f"unit {u['unit']}: {npz} differs from its stored sha256")
        out.append(dict(u, case_path=npz, noise_key=u.get("noise_key", NOISE_KEY)))
    if keep is not None and len(out) != len(keep):
        raise SystemExit(f"units not in the input set: {sorted(keep - {u['unit'] for u in out})}")
    return out


def make_decoder(layout: paths.Layout, prior: str, mem_fraction=None, checkpoint=None, steps=None):
    """The HUG decoder for one prior (checkpoint and hug source verified by sha256 / git commit)."""
    os.environ.update(manifest.hf_env(layout.root))
    ckpt, sha, default_steps = layout.checkpoint(prior)
    if checkpoint:
        ckpt, sha = Path(checkpoint), None
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint for {prior!r} not found: {ckpt} (asset part for this prior; see ASSETS.md)")
    from tools.float import decode
    import atexit
    import shutil
    if prior in paths.OBJECT_FRAME_PRIORS:                          # EX396: the object-frame prior has its own code checkout and decoder class; its pin lives in tools/float/decode.py
        os.environ["WF_MANO_ROOT"] = str(layout.mano)                   # r8: the model's code is our integrated package tools/float/objframe (PROVENANCE.json): no checkout, no shim, no extra part
        return decode.ObjectFrameDecoder(ckpt, int(steps or default_steps), hug_commit=paths.HUG2OBJ_SOURCE_COMMIT, expected_sha256=sha, mem_fraction=mem_fraction, prior=prior)
    if not (layout.hug_source / ".git").exists():
        raise SystemExit(f"{layout.hug_source} is not a git checkout; run `wf.py assets fetch-hug-source`")
    work = Path(tempfile.mkdtemp(prefix="wf_decode_"))              # the import shim (a link to the HUG source): never inside an output folder
    atexit.register(shutil.rmtree, work, True)
    return decode.Decoder(ckpt, int(steps or default_steps), hug_source=layout.hug_source, hug_commit=paths.HUG_SOURCE_COMMIT,
                          shim_dir=work / "hugpkg", expected_sha256=sha, mem_fraction=mem_fraction)


def decode_units(dec, units, draws: int, base_seed: int, request_seed: int, batch_inputs: int):
    """-> (columns dict, seconds). Rows are unit-major, draw-minor, as the project's census files."""
    import numpy as np
    from tools.float import decode
    frames, noises, seeds = [], [], []
    for u in units:
        with np.load(u["case_path"]) as z:
            frames.append(decode.make_frame(z["rgb"], z["depth_mm"], z["camera_k"], z["condition_uv"], int(request_seed), object_mask=decode.case_object_mask(z)))
        sd, w = decode.case_noise(int(base_seed), u["noise_key"], int(draws))
        seeds.append(sd)
        noises.append(w)
    import torch
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    recs = dec.decode_many(frames, noises, batch_inputs=int(batch_inputs), return_encoding=False)
    torch.cuda.synchronize()
    secs = time.time() - t0
    cols_peak = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    cols = dict(unit=np.array([u["unit"] for u in units for _ in range(draws)], dtype=str),
                cluster=np.array([u.get("cluster", u["unit"]) for u in units for _ in range(draws)], dtype=str),
                candidate_index=np.tile(np.arange(draws), len(units)), noise_seed=np.array([x for ss in seeds for x in ss], np.int64))
    for k, shape in SHAPES.items():
        cols[k] = np.concatenate([r[k] for r in recs], 0) if recs else np.zeros((0, *shape), bool if k == "finite" else float)
    cols["_peak_gpu_mib"] = cols_peak
    return cols, secs


def run(a) -> int:
    import numpy as np
    layout = paths.Layout(paths.assets_root(a.assets))
    manifest.require_external(layout.root)               # the MANO model is downloaded by the user: stop early with a clear message if it is not in place
    out = util.refuse_inside(a.out, paths.RELEASE, layout.root)
    if out.exists():
        raise SystemExit(f"{out} exists; decode outputs are never overwritten")
    store = Path(a.store) if getattr(a, "store", None) else layout.input_store
    units = load_input_set(a.inputs, store, a.units.split(",") if a.units else None)
    t_load = time.time()
    dec = make_decoder(layout, a.prior, a.mem_fraction)
    t_load = time.time() - t_load
    out.mkdir(parents=True)                                # only now: a failed set-up leaves nothing behind
    cols, secs = decode_units(dec, units, a.draws, a.base_seed, a.request_seed, a.batch_inputs)
    peak = cols.pop("_peak_gpu_mib")
    fin = cols["finite"]
    conv = None
    try:
        from tools.float import rollout
        if fin.any():
            conv = float(rollout.landmark_convention_error_mm(cols["T_camera_wrist"][fin], cols["mano_pose_aa"][fin], cols["mano_shape"][fin],
                                                              cols["landmarks_camera"][fin], mano_assets=layout.mano).max())
    except Exception as e:                                          # noqa: BLE001 (the convention check is a diagnostic, not a gate)
        conv = f"not computed: {type(e).__name__}: {e}"
    np.savez_compressed(out / f"preds_{a.prior}.npz", **cols)
    report = dict(prior=a.prior, units=len(units), draws=int(a.draws), base_seed=int(a.base_seed), request_seed=int(a.request_seed),
                  rows=int(len(fin)), finite_rows=int(fin.sum()), seconds=round(secs, 2), model_load_s=round(t_load, 2), peak_gpu_memory_mib=peak,
                  convention_error_mm_max=conv, settings=dec.settings, inputs=str(Path(a.inputs).resolve()),
                  inputs_sha256=sha256_file(a.inputs), release=RELEASE_NAME, versions=versions(), host=socket.gethostname(), finished=now(),
                  preds=str(out / f"preds_{a.prior}.npz"), preds_sha256=sha256_file(out / f"preds_{a.prior}.npz"))
    (out / "decode.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    dec.close()
    emit({k: report[k] for k in ("prior", "units", "rows", "finite_rows", "seconds", "model_load_s", "peak_gpu_memory_mib", "convention_error_mm_max", "preds_sha256")})
    return 0


def make_jobs(a) -> int:
    import numpy as np
    layout = paths.Layout(paths.assets_root(a.assets))
    folder = Path(a.decoded).resolve()
    out = util.refuse_inside(a.out, paths.RELEASE, layout.root)
    if out.exists():
        raise SystemExit(f"{out} exists; jobs files are never overwritten")
    rep = json.loads((folder / "decode.json").read_text())
    prior = rep["prior"]
    z = np.load(folder / f"preds_{prior}.npz")
    store = Path(a.store) if getattr(a, "store", None) else layout.input_store
    units = {u["unit"]: u for u in load_input_set(a.inputs, store)}
    cases = {}
    for name in sorted(set(str(x) for x in z["unit"])):
        with np.load(units[name]["case_path"]) as c:
            cases[name] = (np.asarray(c["T_base_camera"], np.float64), np.asarray(c["settled_object_qpos"], np.float64))
    n = len(z["unit"])
    key = np.array([f"{prior}/{z['unit'][i]}#{int(z['candidate_index'][i])}" for i in range(n)])
    arrays = dict(key=key,
                  sim_assets_dir=np.array([units[str(u)]["asset"] for u in z["unit"]]),
                  T_base_camera=np.stack([cases[str(u)][0] for u in z["unit"]]),
                  T_camera_wrist=np.asarray(z["T_camera_wrist"], np.float64),
                  finger_aa=np.asarray(z["mano_pose_aa"], np.float64),
                  landmarks_camera=np.asarray(z["landmarks_camera"], np.float64),
                  settled_qpos7=np.stack([cases[str(u)][1] for u in z["unit"]]),
                  bench_table_z=np.zeros(n))
    out.parent.mkdir(parents=True, exist_ok=True)
    side = None
    if not getattr(a, "no_canaries", False):         # known-answer jobs (README section 8): spread over the protocols you will roll the file under
        protocols = [x for x in a.protocols.split(",") if x] if getattr(a, "protocols", None) else canary.MANO_PROTOCOLS
        arrays, side = canary.add(arrays, protocols, layout.root, seed=f"make-jobs/{sha256_file(folder / f'preds_{prior}.npz')}")
    np.savez_compressed(out, **arrays)
    if side is not None:
        canary.write_sidecar(out, side)
    emit({"out": str(out), "jobs": n, "finite_rows": int(np.asarray(z["finite"]).sum()), "prior": prior, "out_sha256": sha256_file(out),
          "canaries": side["n_canary"] if side else 0, "canary_protocols": side["protocols"] if side else []})
    return 0


def compare_decode(got: dict, want: dict) -> dict:
    """Cross-GPU comparison of two decode column sets (same rows). Reports the maxima and whether each is within TOL."""
    import numpy as np
    for k in ("unit", "candidate_index", "noise_seed"):
        if not np.array_equal(got[k], want[k]):
            return {"pass": False, "problems": [f"rows differ in {k} (the noise rows or the input order changed)"]}
    problems = []
    if not np.array_equal(got["noise"], want["noise"]):
        problems.append("the noise rows differ from the reference (torch CPU generator): decode inputs are not comparable")
    fin = np.asarray(got["finite"], bool) & np.asarray(want["finite"], bool)
    m = {"finite_equal": bool(np.array_equal(got["finite"], want["finite"])), "rows": int(len(got["unit"])), "finite_rows": int(fin.sum()),
         "mano_params99": float(np.abs(got["mano_params99"][fin] - want["mano_params99"][fin]).max()),
         "landmarks_camera_m": float(np.abs(got["landmarks_camera"][fin] - want["landmarks_camera"][fin]).max()),
         "wrist_translation_m": float(np.abs(got["T_camera_wrist"][fin][:, :3, 3] - want["T_camera_wrist"][fin][:, :3, 3]).max()),
         "wrist_rotation_max_abs": float(np.abs(got["T_camera_wrist"][fin][:, :3, :3] - want["T_camera_wrist"][fin][:, :3, :3]).max()),
         "mano_pose_aa_rad": float(np.abs(got["mano_pose_aa"][fin] - want["mano_pose_aa"][fin]).max())}
    for k, tol in TOL.items():
        m[f"{k}_within_tol"] = bool(m[k] <= tol)
    ok = m["finite_equal"] and all(m[f"{k}_within_tol"] for k in TOL) and not problems
    return dict(m, tolerances=TOL, problems=problems, **{"pass": bool(ok)})


def check(a) -> int:
    import numpy as np
    layout = paths.Layout(paths.assets_root(a.assets))
    manifest.require_external(layout.root)
    ref = paths.REFERENCE / "decode"
    out = util.refuse_inside(a.out, paths.RELEASE, layout.root)
    out.mkdir(parents=True, exist_ok=True)
    units = load_input_set(ref / "input_set.json", ref / "input_store")
    requested = [p for p in a.priors.split(",")] if a.priors else list(paths.PRIORS)
    unknown = [p for p in requested if p not in paths.PRIORS]
    if unknown:
        raise SystemExit(f"unknown prior(s) {unknown}; choose from {list(paths.PRIORS)}")
    report, ran = {}, 0
    for prior in requested:
        expected = ref / f"expected_{prior}.npz"
        ckpt, sha, _ = layout.checkpoint(prior)
        if not ckpt.is_file():
            report[prior] = {"pass": None, "skipped": f"checkpoint {ckpt.relative_to(layout.root)} is not in the asset root"}
            continue
        if not expected.is_file():
            report[prior] = {"pass": False, "problems": [f"no reference {expected.name}"]}
            continue
        dec = make_decoder(layout, prior)
        cols, secs = decode_units(dec, units, 32, 20261116, 0, 8)
        peak = cols.pop("_peak_gpu_mib")
        cols2, _ = decode_units(dec, units, 32, 20261116, 0, 8)              # the same GPU twice: how repeatable is it here
        cols2.pop("_peak_gpu_mib")
        dec.close()
        want = dict(np.load(expected))
        rep = compare_decode(cols, want)
        rep["same_gpu_repeat_max_abs_params99"] = float(np.abs(cols["mano_params99"][cols["finite"]] - cols2["mano_params99"][cols["finite"]]).max())
        rep["seconds"] = round(secs, 2)
        rep["peak_gpu_memory_mib"] = peak
        rep["device"] = dec.settings["device"]
        rep["torch"] = dec.settings["torch"]
        rep["checkpoint_sha256"] = dec.settings["checkpoint_sha256"]
        report[prior] = rep
        np.savez_compressed(out / f"preds_{prior}.npz", **cols)
        ran += 1
    verdicts = [r["pass"] for r in report.values() if r.get("pass") is not None]
    ok = bool(verdicts) and all(verdicts)
    emit({"pass": ok, "priors": report, "reference": str(ref), "release": RELEASE_NAME, "versions": versions(), "host": socket.gethostname()})
    (out / "decode_check.json").write_text(json.dumps({"pass": ok, "priors": report}, indent=1, default=str) + "\n")
    return 0 if ok else 1


def main(a) -> int:
    return {"run": run, "make-jobs": make_jobs, "check": check}[a.cmd](a)
