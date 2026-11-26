"""The verification suite: run it first on every new machine. A machine's runs count only after the levels you need pass.

  L0   files: the release folder against its SHA256SUMS, and the asset parts needed by the other levels against asset_manifests/
  L1   CPU physics, MANO protocols: the 200 HUG-Bench reference grasps under mano-1111-p1 and mano-1111-p2, and the 220 dumbbell
       census reference grasps under the six MANO protocols; every label and every other record field must equal the stored
       reference bit for bit (status pass_bitwise). `labels_only` (labels equal, a diagnostic differs) means this CPU computes
       slightly different physics: that is not a pass, tell us before running any job there.
  L1h  CPU physics, robot hands: sharpa-float-p1/p2 and wuji1hr-float-p1/p2 (200 + 64 reference grasps each) and wuji1-float-p1
       (200 grasps); needs the `hands` asset part, `wf.py assets link-hands` and requirements/cpu_hands.txt. r8f: wuji22-float-p2 (the Wuji Hand 2.2,
       200 known-answer grasps) is checked too when the asset part `hands22` is unpacked (or named with --protocols)
  L1b  CPU physics on our production batches (needs the `batches` asset part): 100 rows spread over each batch are rolled and compared with
       our stored results (every label equal; bit for bit, or `pass_near`: at most 2 % of the jobs differ in a diagnostic by at most 1e-3,
       the floating-point noise between CPU models that a few Wuji rollouts show); robot-hand batches need the `hands` part
  L2   GPU decode: the reference pictures decoded with each available prior; agreement with the stored reference decode within the
       stated tolerance (GPU kernels are not bit-identical across GPU models)
  L3   GPU DSRL-NA: the port's exactness tests (CPU, bit for bit) and a short training run on the reference mini data against the
       stored reference log within the stated tolerance

Every level writes its records into --out; the last line printed is `VERIFY PASS` or `VERIFY FAIL`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import RELEASE_NAME, manifest, paths, util
from .util import emit, now, sha256_file, versions, host_info

LEVELS = ("L0", "L1", "L1h", "L1b", "L2", "L3")

WF = str(paths.CODE / "wf.py")
MANO_PROTOCOLS = ["mano-1111-p1", "mano-1111-p2", "mano-1111-p2s-dzm1", "mano-1111-p2-dzm1", "mano-1111-p2-dzm2", "mano-1111-p2-dzp1"]
HAND_PROTOCOLS = ["sharpa-float-p1", "sharpa-float-p2", "wuji1hr-float-p1", "wuji1hr-float-p2"]
REF = paths.REFERENCE / "cpu"


def _last_json(text: str):
    """The last JSON document in a command's stdout (the HUG loader prints a console panel before it)."""
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("{"):
            try:
                return json.loads("\n".join(lines[i:]))
            except ValueError:
                continue
    return None


def _run(args, env=None) -> tuple:
    p = subprocess.run([sys.executable, WF, *args], capture_output=True, text=True, env=env)
    return p.returncode, _last_json(p.stdout), p.stderr[-2000:]


def _cpu_check(name, protocol, jobs: Path, expected: Path, out: Path, assets, workers: int, results: list, strict: bool = True) -> None:
    """Roll `jobs` under `protocol` and compare with `expected`; append one result record."""
    res = out / f"{name}.npz"
    base = ["--assets", str(assets)] if assets else []
    rec = {"check": name, "protocol": protocol, "jobs": jobs.name, "expected": expected.name}
    if not expected.is_file():
        rec.update(status="skipped", note=f"no reference {expected.name}")
        results.append(rec)
        return
    rc, run_doc, err = _run(base + ["rollout", "run", "--protocol", protocol, "--jobs", str(jobs), "--out", str(res), "--workers", str(workers)])
    (out / f"{name}.run.json").write_text(json.dumps(run_doc, indent=1) + "\n" if run_doc else err)
    if rc != 0 or not run_doc or "out" not in run_doc:
        rec.update(status="fail", note="the rollout did not run: " + (", ".join(run_doc.get("problems", [])) if run_doc else err.strip().splitlines()[-1] if err.strip() else "no output"))
        results.append(rec)
        print(f"  {name}: FAIL {rec['note']}", flush=True)
        return
    rc, cmp_doc, err = _run(["rollout", "compare", "--results", str(res), "--expected", str(expected)])
    (out / f"{name}.compare.json").write_text(json.dumps(cmp_doc, indent=1) + "\n" if cmp_doc else err)
    status = cmp_doc["status"] if cmp_doc else "fail"
    if strict and status == "pass_near":          # levels L1 / L1h compare with reference records that three hosts reproduced bit for bit
        status = "labels_only"
    rec.update(status=status, labels_equal=cmp_doc.get("labels_equal") if cmp_doc else None,
               jobs_differing=cmp_doc.get("jobs_differing_in_any_field") if cmp_doc else None, max_abs_diff=cmp_doc.get("max_abs_diff_any_field") if cmp_doc else None,
               jobs_n=cmp_doc.get("jobs") if cmp_doc else None, diagnostics_all_equal=cmp_doc.get("diagnostics_all_equal") if cmp_doc else None,
               rollouts_per_s=run_doc.get("rollouts_per_s"), wall_s=run_doc.get("wall_s"), workers=run_doc.get("workers"), out_sha256=run_doc.get("out_sha256"),
               mujoco_warnings=run_doc.get("mujoco_warnings"))
    results.append(rec)
    extra = f"  ({rec['jobs_differing']} job(s) differ, largest difference {rec['max_abs_diff']:.2g})" if rec["status"] == "pass_near" else ""
    print(f"  {name}: {rec['status']}  labels {rec['labels_equal']}/{rec['jobs_n']}  all fields equal: {rec['diagnostics_all_equal']}  "
          f"{rec['rollouts_per_s']} rollouts/s{extra}", flush=True)


def main(a) -> int:
    assets = paths.assets_root(a.assets)
    out = util.refuse_inside(a.out, paths.RELEASE, assets)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out} is not empty: every check writes its own result files and never overwrites one; use a new folder")
    out.mkdir(parents=True, exist_ok=True)
    levels = [x.strip() for x in a.levels.split(",") if x.strip()]
    unknown = [l for l in levels if l not in LEVELS]
    if unknown or not levels:
        raise SystemExit(f"unknown level(s) {unknown or levels}: choose from {', '.join(LEVELS)}")
    sets = [x.strip() for x in a.set.split(",") if x.strip()]
    bad_priors = [p for p in (a.priors.split(",") if a.priors else []) if p not in paths.PRIORS]
    if bad_priors:
        raise SystemExit(f"unknown prior(s) {bad_priors}: choose from {', '.join(paths.PRIORS)}")
    results, t0 = [], time.time()
    need_parts = {"core", "mano"} if any(l in levels for l in ("L1", "L1h", "L1b")) else set()
    if "L1h" in levels:
        need_parts.add("hands")
        if "wuji22-float-p2" in (a.protocols or "").split(",") or (not a.protocols and "hands22" in manifest.present_parts(assets)):
            need_parts.add("hands22")
    if "L1b" in levels:
        need_parts.update({"core", "mano", "batches"})
    if "L2" in levels:
        need_parts.update({"dinov2", "mano"})
        layout = paths.Layout(assets)
        for prior in (a.priors.split(",") if a.priors else list(paths.PRIORS)):
            if layout.checkpoint(prior)[0].is_file():
                need_parts.add({"hug1": "hug1", "hug2": "hug2", "dumbbell50k": "dumbbell50k"}[prior])
    ok = True
    print(f"wf verify {RELEASE_NAME}: levels {','.join(levels)}, asset root {assets}", flush=True)
    ext_bad = {p: r for p, r in manifest.check_parts(assets, sorted(need_parts & set(manifest.EXTERNAL))).items() if not r["pass"]}
    if ext_bad:          # a part the user downloads themselves (the MANO model) is missing or wrong: say so once and stop, instead of failing every check
        for p, r in ext_bad.items():
            for prob in r["problems"]:
                print(f"  {p}: {prob}", flush=True)
        (out / "verify_report.json").write_text(json.dumps({"release": RELEASE_NAME, "levels": levels, "pass": False, "finished": now(), "host": host_info(),
                                                            "checks": [{"check": f"external asset part {p}", "status": "fail", "problems": r["problems"]} for p, r in ext_bad.items()]}, indent=1) + "\n")
        print(f"VERIFY FAIL: {', '.join(sorted(ext_bad))} must be put in the asset root first (see above)", flush=True)
        return 1

    if "L0" in levels:
        rel = manifest.check_release()
        wanted = sorted(need_parts) if need_parts else manifest.present_parts(assets)       # L0 alone or L0,L3: whatever parts are present
        if getattr(a, "parts", None):
            wanted = [p for p in a.parts.split(",") if p]
        parts = manifest.check_parts(assets, wanted) if wanted else {}
        bundle = manifest.check_bundle(assets)
        (out / "l0_release.json").write_text(json.dumps(rel, indent=1) + "\n")
        (out / "l0_assets.json").write_text(json.dumps({"parts": parts, "bundle": bundle}, indent=1) + "\n")
        if bundle["pass"] is not None:
            results.append({"check": "L0 asset bundle SHA256SUMS", "status": "pass_bitwise" if bundle["pass"] else "fail", "problems": bundle.get("problems")})
            print(f"  L0 asset bundle SHA256SUMS: {'PASS' if bundle['pass'] else 'FAIL'}", flush=True)
        results.append({"check": "L0 release folder", "status": "pass_bitwise" if rel["pass"] else "fail", "files": rel.get("files"),
                        "sha256sums_sha256": rel.get("sha256sums_sha256"), "problems": rel.get("problems")})
        for p, r in parts.items():
            results.append({"check": f"L0 asset part {p}", "status": "pass_bitwise" if r["pass"] else "fail", "files": r.get("files"),
                            "bytes": r.get("bytes"), "problems": r.get("problems")})
        print(f"  L0 release folder: {'PASS' if rel['pass'] else 'FAIL'} ({rel.get('files')} files)", flush=True)
        for p, r in parts.items():
            print(f"  L0 asset part {p}: {'PASS' if r['pass'] else 'FAIL'} ({r.get('files')} files)", flush=True)
            for prob in ([] if r["pass"] else r.get("problems", [])):
                print(f"    {prob}", flush=True)
        for p in sorted(set(manifest.EXTERNAL) - set(parts)):
            print(f"  note: asset part {p} was not checked (not in the asset root; it is not in the tarballs, README.md section 3.0 says how to get it)", flush=True)

    if "L1" in levels:
        protos = a.protocols.split(",") if a.protocols else MANO_PROTOCOLS
        if "ref200" in sets:
            for p in protos:
                if p in ("mano-1111-p1", "mano-1111-p2"):
                    _cpu_check(f"l1_ref200_{p}", p, REF / "jobs_ref200.npz", REF / f"expected_ref200_{p}.npz", out, assets, a.workers, results)
        if "w3" in sets:
            for p in protos:
                _cpu_check(f"l1_w3_{p}", p, REF / "jobs_w3_220.npz", REF / f"expected_w3_{p}.npz", out, assets, a.workers, results)
        if "bench2000" in sets:
            _cpu_check("l1_bench2000_mano-1111-p1", "mano-1111-p1", REF / "jobs_bench2000.npz", REF / "expected_bench2000_mano-1111-p1.npz", out, assets, a.workers, results)

    if "L1b" in levels:
        idx = json.loads((assets / "batches/INDEX.json").read_text())
        hands_here = (assets / "hand_models").is_dir()
        for bid, b in idx["batches"].items():
            for k, pair in enumerate(b["pairs"]):
                name = f"l1b_{bid}_{pair['expected'].replace('expected_', '').replace('.npz', '')}"
                if pair["protocol"].split("-")[0] in ("sharpa", "wuji1hr", "wuji1") and not hands_here:
                    results.append({"check": name, "status": "skipped", "note": "needs the `hands` asset part"})
                    continue
                sdir = out / "slices" / name
                rc, doc, err = _run(["--assets", str(assets), "rollout", "slice", "--jobs", str(assets / "batches" / bid / pair["jobs"]),
                                     "--expected", str(assets / "batches" / bid / pair["expected"]), "--n", str(a.slice_n), "--out-dir", str(sdir)])
                if rc != 0:
                    results.append({"check": name, "status": "fail", "note": "slice failed: " + err.strip()[-200:]})
                    continue
                _cpu_check(name, pair["protocol"], sdir / "jobs_slice.npz", sdir / "expected_slice.npz", out, assets, a.workers, results, strict=False)
    if "L1h" in levels:
        hp = a.protocols.split(",") if a.protocols else HAND_PROTOCOLS + ["wuji1-float-p1"] + (["wuji22-float-p2"] if "hands22" in need_parts else [])
        for p in hp:
            if p == "wuji22-float-p2":
                _cpu_check(f"l1h_ref200_{p}", p, REF / "hands22/jobs_ref200_wuji22.npz", REF / "hands22/expected_ref200_wuji22-float-p2.npz", out, assets, a.workers, results)
                continue
            if p == "wuji1-float-p1":
                _cpu_check(f"l1h_ref200_{p}", p, REF / "wuji1/jobs_ref200_wuji1.npz", REF / "wuji1/expected_ref200_wuji1-float-p1.npz", out, assets, a.workers, results)
                continue
            for s in ("ref200", "pilot64"):
                _cpu_check(f"l1h_{s}_{p}", p, REF / f"hands/jobs_{s}_hands.npz", REF / f"hands/expected_{s}_{p}.npz", out, assets, a.workers, results)

    env = dict(os.environ)
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    base = ["--assets", str(assets)]
    if "L2" in levels:
        args = base + ["decode", "check", "--out", str(out / "l2_decode")] + (["--priors", a.priors] if a.priors else [])
        rc, doc, err = _run(args, env)
        (out / "l2_decode.json").write_text(json.dumps(doc, indent=1) + "\n" if doc else err)
        if doc is None:
            results.append({"check": "L2 decode", "status": "fail", "note": err.strip().splitlines()[-1] if err.strip() else "no output"})
            print(f"  L2 decode: FAIL {results[-1]['note']}", flush=True)
        else:
            for prior, r in doc["priors"].items():
                st = "skipped" if r.get("pass") is None else ("pass_tolerance" if r["pass"] else "fail")
                results.append({"check": f"L2 decode {prior}", "status": st, "max_params99": r.get("mano_params99"), "max_landmarks_m": r.get("landmarks_camera_m"),
                                "device": r.get("device"), "note": r.get("skipped")})
                print(f"  L2 decode {prior}: {st.upper()} " + (f"max |d params99| {r['mano_params99']:.2e}, landmarks {r['landmarks_camera_m']:.2e} m ({r.get('device')})" if r.get("pass") is not None else r.get("skipped", "")), flush=True)
    if "L3" in levels:
        args = base + ["dsrl", "check", "--out", str(out / "l3_dsrl")]
        rc, doc, err = _run(args, env)
        (out / "l3_dsrl.json").write_text(json.dumps(doc, indent=1) + "\n" if doc else err)
        if doc is None:
            results.append({"check": "L3 dsrl", "status": "fail", "note": err.strip().splitlines()[-1] if err.strip() else "no output"})
            print(f"  L3 dsrl: FAIL {results[-1]['note']}", flush=True)
        else:
            for name, r in doc["checks"].items():
                st = "skipped" if r.get("pass") is None else ("pass_bitwise" if r["pass"] and r.get("exact") else ("pass_tolerance" if r["pass"] else "fail"))
                results.append({"check": f"L3 {name}", "status": st, "note": r.get("note")})
                print(f"  L3 {name}: {st.upper()} {r.get('note', '')}", flush=True)

    good = ("pass_bitwise", "pass_near", "pass_tolerance", "skipped")
    ok = all(r["status"] in good for r in results) and any(r["status"] != "skipped" for r in results)
    summary = {"release": RELEASE_NAME, "levels": levels, "pass": ok, "finished": now(), "seconds": round(time.time() - t0, 1), "host": host_info(),
               "release_sha256sums_sha256": sha256_file(paths.RELEASE / "SHA256SUMS") if (paths.RELEASE / "SHA256SUMS").is_file() else None,
               "versions": versions(), "checks": results}
    (out / "verify_report.json").write_text(json.dumps(summary, indent=1) + "\n")
    bad = [r["check"] + ": " + r["status"] for r in results if r["status"] not in good]
    print(("VERIFY PASS" if ok else "VERIFY FAIL") + f" ({len(results)} checks, {summary['seconds']} s)" + ("" if ok else "; failed: " + "; ".join(bad)), flush=True)
    return 0 if ok else 1
