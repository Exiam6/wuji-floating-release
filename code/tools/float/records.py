"""Result records of the floating-hand line, in the schema of the experiment system `xp`
(docs/wuji_grasplift/EXPERIMENT_SYSTEM_2026-11-16.md, tools/xp/record.py).

One record = one executed sample (one rollout):
  exp, row, seed, attempt     which registered run produced it (stamped by xp from the attempt's launch record)
  protocol {name, digest}     the evaluator protocol stamp (tools/float/protocol.stamp)
  unit                        the configuration id (object pose and camera); the unit of analysis
  cluster                     the object, for object-level resampling
  arm                         the experimental arm (a prior, a policy, a control)
  sample                      index of the sample within (unit, arm)
  metric fields               numbers only, the same set in every record (`metric_fields`)

Registered runs write through `xp.record.write(attempt_dir, protocol=, unit=, cluster=, arm=, sample=,
**metric_fields(result))`. Gate scripts and tests, which are not xp attempts, use `make` / `append` here; the
metric fields are the same, so a gate's records look exactly like a run's.

Metric fields (all numbers):
  success_native   1 when the evaluator's own label is success, else 0
  success_strict_v1  1 when success_native AND thumb plus at least one other finger touch the object at the final
                   frame AND the final object rise is at least 400 mm (fixed 2026-11-16, EX244 row d)
  sim_valid        1 for a valid simulation
  stage_code       index of the failure stage in STAGES (0 invalid_input ... 5 success)
  contacts and fingers at the end of settle and at the final frame (counts; thumb and palm as 0 / 1; the groups
  in contact as a bit mask, FINGER_BITS: index 1, middle 2, pinky 4, ring 8, thumb 16, palm 32)
  penetration (mm) static, at the pre-grasp pose, at the end of settle, at the final frame, maximum over the
  20 Hz states of the whole rollout and of the lift; share of lift states with hand-object contact
  object rise (mm) peak and final; object displacement before the lift
A sample whose input was not finite (stage_code 0) never ran: its diagnostics are written as 0 and it counts as
a failure.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

STAGES = ("invalid_input", "sim_invalid", "no_prelift_contact", "not_lifted", "dropped", "success")
IDENTITY = ("exp", "row", "seed", "attempt", "protocol", "unit", "cluster", "arm", "sample")
KEY = ("exp", "row", "seed", "attempt", "unit", "arm", "sample")
# numeric diagnostics copied from a rollout result (tools/float/rollout.RESULT_FIELDS), 0 when the sample never ran
NUMERIC = ("sim_z_lifted", "obj_rise_peak_mm", "obj_rise_final_mm", "init_contacts", "init_pen_mm", "static_pen_obj_mm",
           "static_pen_table_mm", "prelift_contacts", "prelift_pen_mm", "prelift_obj_disp_mm", "final_contacts",
           "final_pen_mm", "pen_max_mm", "lift_pen_max_mm", "lift_contact_share", "lift_min_contacts",
           "badqacc_warnings", "n_steps", "t_rollout_s")
METRICS = ("success_native", "success_strict_v1", "sim_valid", "stage_code", "prelift_fingers_n", "final_fingers_n", "final_thumb",
           "final_palm", "prelift_finger_mask", "final_finger_mask") + NUMERIC
# strict label v1 (fixed 2026-11-16 from EX244 row ex244-d_strictlabel_1116, ledger NOTE): native success AND the final-frame
# hand-object contact includes the thumb and at least one other finger AND the final object rise is at least STRICT_V1_RISE_MM.
# Penetration is not part of it (reported as a diagnostic).
STRICT_V1_RISE_MM = 400.0
# bit of each contact group in the finger masks (the adapter's group names)
FINGER_BITS = {"index": 1, "middle": 2, "pinky": 4, "ring": 8, "thumb": 16, "palm": 32}


def _num(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _parts(text) -> list:
    return [p for p in str(text or "").split(",") if p]


def metric_fields(result: dict, **extra) -> dict:
    """The numeric metric fields of one rollout result (tools/float/rollout.run_job), plus caller numbers such as
    noise_seed. Every key of METRICS is present in every record."""
    out = {"success_native": int(bool(result["success"])), "sim_valid": int(bool(result.get("sim_valid", False))),
           "stage_code": STAGES.index(result["stage"])}
    final = _parts(result.get("final_fingers"))
    prelift = _parts(result.get("prelift_fingers"))
    out["prelift_fingers_n"] = len(prelift)
    out["prelift_finger_mask"] = sum(FINGER_BITS[p] for p in set(prelift))
    out["final_finger_mask"] = sum(FINGER_BITS[p] for p in set(final))
    others = [p for p in set(final) if p not in ("thumb", "palm")]
    out["success_strict_v1"] = int(bool(out["success_native"]) and "thumb" in final and len(others) >= 1
                                   and float(result.get("obj_rise_final_mm") or 0.0) >= STRICT_V1_RISE_MM)
    out["final_fingers_n"] = len(final)
    out["final_thumb"] = int("thumb" in final)
    out["final_palm"] = int("palm" in final)
    for k in NUMERIC:
        out[k] = _num(result.get(k, 0)) if result.get(k) is not None else 0
    for k, v in extra.items():
        if k in out or k in IDENTITY:
            raise ValueError(f"extra field {k} would overwrite a record field")
        out[k] = _num(v)
    bad = [k for k, v in out.items() if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)]
    if bad:
        raise ValueError(f"metric fields must be finite numbers: {bad}")
    return out


def problems(rec: dict) -> list:
    """What is wrong with one record (the checks of tools/xp/record.problems, plus the fixed metric set)."""
    out = [f"`{k}` is missing" for k in IDENTITY if k not in rec]
    p = rec.get("protocol")
    if "protocol" in rec and not (isinstance(p, dict) and all(isinstance(p.get(k), str) and p[k] for k in ("name", "digest"))):
        out.append("`protocol` must be {name, digest}, both non-empty strings")
    for k in ("unit", "cluster", "arm"):
        if k in rec and not (isinstance(rec[k], (str, int)) and not isinstance(rec[k], bool) and rec[k] != ""):
            out.append(f"`{k}` must be a non-empty string or an integer")
    if "sample" in rec and not (isinstance(rec["sample"], int) and not isinstance(rec["sample"], bool)):
        out.append("`sample` must be an integer")
    out += [f"metric `{k}` is missing" for k in METRICS if k not in rec]
    for k, v in rec.items():
        if k not in IDENTITY and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)):
            out.append(f"metric `{k}` must be a finite number, got {v!r}")
    if rec.get("success_native") not in (0, 1):
        out.append("`success_native` must be 0 or 1")
    return out


def validate(rec: dict) -> dict:
    found = problems(rec)
    if found:
        raise ValueError("invalid result record: " + "; ".join(found))
    return rec


def make(*, exp: str, row: str, seed: int, attempt, protocol: dict, unit: str, cluster: str, arm: str, sample: int,
         result: dict, **extra) -> dict:
    """A full record for a gate script or a test (a registered run lets xp stamp exp, row, seed and attempt)."""
    rec = {"exp": exp, "row": row, "seed": int(seed), "attempt": attempt, "protocol": dict(protocol), "unit": unit,
           "cluster": cluster, "arm": arm, "sample": int(sample)}
    rec.update(metric_fields(result, **extra))
    return validate(rec)


def strict_v1(rec: dict) -> int:
    """strict v1 from a record's numeric fields (for records written before the field existed)."""
    mask = int(rec["final_finger_mask"]) if "final_finger_mask" in rec else None
    if mask is None:          # older records: counts only (thumb flag, palm flag, number of contact groups)
        others = int(rec["final_fingers_n"]) - int(rec["final_thumb"]) - int(rec["final_palm"])
        thumb = int(rec["final_thumb"])
    else:
        names = fingers_of(mask)
        others = len([n for n in names if n not in ("thumb", "palm")])
        thumb = int("thumb" in names)
    return int(bool(rec["success_native"]) and thumb == 1 and others >= 1 and float(rec["obj_rise_final_mm"]) >= STRICT_V1_RISE_MM)


def stage_of(rec: dict) -> str:
    return STAGES[int(rec["stage_code"])]


def fingers_of(mask: int) -> list:
    """The contact groups of a finger mask, by name."""
    return [name for name, bit in FINGER_BITS.items() if int(mask) & bit]


def key_of(rec: dict) -> tuple:
    return tuple(rec[k] for k in KEY)


def read(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path, records) -> int:
    """Append validated records to a results.jsonl; refuses a sample that is already in the file or repeated in
    the batch, and a batch that would mix protocols with the file."""
    path = Path(path)
    old = read(path)
    seen = {key_of(r) for r in old}
    stamps = {(r["protocol"]["name"], r["protocol"]["digest"]) for r in old}
    lines = []
    for rec in records:
        validate(rec)
        k = key_of(rec)
        if k in seen:
            raise ValueError(f"sample already recorded: {k}")
        seen.add(k)
        stamps.add((rec["protocol"]["name"], rec["protocol"]["digest"]))
        lines.append(json.dumps(rec, allow_nan=False))
    if len(stamps) > 1:
        raise ValueError(f"records of different protocols in one file: {sorted(stamps)}")
    if lines:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write("\n".join(lines) + "\n")
    return len(lines)
