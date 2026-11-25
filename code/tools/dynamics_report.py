#!/usr/bin/env python3
"""Release r8c (EX403): learning curves of one DSRL-NA training from its dynamics log and the read-outs of its checkpoints. Reads only; CPU; numpy (matplotlib for the PNGs, optional).

    python code/tools/dynamics_report.py --train-out RUN --inputs INPUT_SET.json --out DIR \
        [--point CALLS=READOUT_DIR:RESULTS.npz ...] [--group NAME=OBJECT_IDS.txt ...] [--arm own]

  RUN            a `dsrl train` folder written with dynamics_log true (RUN/dynamics.jsonl) and, for the success curves, checkpoint_every (RUN/checkpoints/<calls>/)
  --point        one read-out per checkpoint: CALLS (a checkpoint, or the endpoint's calls), READOUT_DIR (`dsrl readout --checkpoint CALLS --out READOUT_DIR`, its readout_<arm>.npz) and
                 RESULTS.npz (the `rollout run` results of that read-out's jobs: `key`, `success_native`); the job keys of the arm are `<arm>/<unit>` (known-answer jobs are ignored)
  --inputs       the input set of the read-out: each unit's `readout_set` (1: unseen configurations of the training objects, 2: unseen objects) names the curves `set 1` and `set 2`
  --group        an extra curve over the units whose object is listed in OBJECT_IDS.txt (one id per line), e.g. `objects188=objects_188.txt`

Writes DIR/curves.json (every curve as lists, with its denominator) and one PNG per curve: success.png (success per set / group vs calls), qw_vs_success.png (the noise critic's mean value
at the played noise against the measured success, per checkpoint and set), norm.png (the played noise's norm, 5/50/95 %, vs calls), auc.png (the within-input AUC of Q^W on the fixed training
views vs calls), losses.png (the training losses vs calls). Nothing is fitted or smoothed; a missing input leaves its curve out and curves.json says why.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def object_of(unit: str) -> str:
    return str(unit).split("#", 1)[0].rsplit("/", 1)[-1]


def success_points(points, arm: str, sets: dict, groups: dict) -> tuple:
    """{curve: [(calls, rate, n)]}, {curve: [(calls, mean q_noise, rate)]} from the read-outs of the checkpoints."""
    succ, qv = {}, {}
    for calls, rdir, res in sorted(points, key=lambda x: x[0]):
        with np.load(res, allow_pickle=False) as z:
            lab = {str(k)[len(arm) + 1:]: float(v) for k, v in zip(z["key"], z["success_native"]) if str(k).startswith(arm + "/")}
        q = {}
        ro = Path(rdir) / f"readout_{arm}.npz"
        if ro.is_file():
            with np.load(ro, allow_pickle=False) as r:
                q = {str(u): float(v) for u, v in zip(r["id"], r["q_noise"])}
        curves = {f"set {s}": [u for u in lab if sets.get(u) == s] for s in sorted(set(sets.values()))}
        curves.update({g: [u for u in lab if object_of(u) in objs] for g, objs in groups.items()})
        curves["all"] = list(lab)
        for name, us in curves.items():
            if not us:
                continue
            rate = float(np.mean([lab[u] for u in us]))
            succ.setdefault(name, []).append((int(calls), rate, len(us)))
            qs = [q[u] for u in us if u in q and np.isfinite(q[u])]
            if qs:
                qv.setdefault(name, []).append((int(calls), float(np.mean(qs)), rate))
    return succ, qv


def plot(out: Path, name: str, series: dict, xlabel: str, ylabel: str, notes: list) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        notes.append(f"{name}.png not written: matplotlib is not installed (curves.json has the numbers)")
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    for label, (x, y) in series.items():
        ax.plot(x, y, marker="o", ms=3, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--point", action="append", default=[], help="CALLS=READOUT_DIR:RESULTS.npz")
    ap.add_argument("--group", action="append", default=[], help="NAME=OBJECT_IDS.txt")
    ap.add_argument("--arm", default="own")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    notes = []
    dyn = read_jsonl(Path(a.train_out) / "dynamics.jsonl")
    if not dyn:
        notes.append("no dynamics.jsonl in the training folder (dynamics_log was not true): the norm, AUC and loss curves are left out")
    sets = {}
    if a.inputs:
        sets = {str(u["unit"]): int(u.get("readout_set", 0)) for u in json.loads(Path(a.inputs).read_text())["units"]}
    groups = {}
    for g in a.group:
        name, f = g.split("=", 1)
        groups[name] = {x.strip() for x in Path(f).read_text().splitlines() if x.strip()}
    points = []
    for p in a.point:
        calls, rest = p.split("=", 1)
        rdir, res = rest.rsplit(":", 1)
        points.append((int(calls), rdir, res))
    succ, qv = success_points(points, a.arm, sets, groups) if points else ({}, {})
    if not points:
        notes.append("no --point: the success and Q^W-against-success curves are left out")
    col = lambda k: [d.get(k) for d in dyn]                                   # noqa: E731
    curves = {"calls": col("calls"), "actor_norm": {q: col(f"actor_norm_p{q}") for q in (5, 50, 95)}, "qw_auc_within": col("qw_auc_within"), "qw_auc_views": col("qw_auc_views"),
              "losses": {k: col(k) for k in ("train_critic_loss", "train_actor_loss", "train_noise_critic_loss", "train_noise_critic_penalty")},
              "q_values": {k: col(k) for k in ("qa_data_mean", "qw_pool_mean", "qw_actor_mean")}, "entropy": col("entropy"), "alpha": col("alpha"),
              "success": {k: [dict(calls=c, rate=r, n=n) for c, r, n in v] for k, v in succ.items()},
              "qw_vs_success": {k: [dict(calls=c, qw_mean=q, rate=r) for c, q, r in v] for k, v in qv.items()}, "arm": a.arm, "notes": notes}
    if succ:
        plot(out, "success", {k: ([c for c, _, _ in v], [r for _, r, _ in v]) for k, v in succ.items()}, "calls", f"native success ({a.arm} read-out)", notes)
    if qv:
        plot(out, "qw_vs_success", {k: ([q for _, q, _ in v], [r for _, _, r in v]) for k, v in qv.items()}, "mean Q^W at the played noise", "measured success", notes)
    if dyn:
        x = col("calls")
        plot(out, "norm", {f"p{q}": (x, col(f"actor_norm_p{q}")) for q in (5, 50, 95)}, "calls", "norm of the played noise", notes)
        ok = [(c, v) for c, v in zip(x, col("qw_auc_within")) if v is not None]
        if ok:
            plot(out, "auc", {"Q^W within-input AUC": ([c for c, _ in ok], [v for _, v in ok])}, "calls", "AUC (fixed training views)", notes)
        else:
            notes.append("auc.png not written: no dynamics line has an AUC (the prep's actions.npz has no `noise`, or no view has both outcomes)")
        plot(out, "losses", {k: (x, [v if v is not None else np.nan for v in col(k)]) for k in ("train_critic_loss", "train_actor_loss", "train_noise_critic_loss")}, "calls", "loss (mean over the last 250 calls)", notes)
    (out / "curves.json").write_text(json.dumps(curves, indent=1) + "\n")
    print(json.dumps({"curves": str(out / "curves.json"), "dynamics_lines": len(dyn), "points": len(points), "notes": notes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
