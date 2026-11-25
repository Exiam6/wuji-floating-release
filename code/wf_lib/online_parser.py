"""Argument parser of `wf.py online`, in its own light module: wf.py imports it for EVERY command, so it imports nothing heavy (no torch, no numpy, no dsrl_port); the code of the command
(online_cmd, online_core) is imported only when `wf.py online ...` runs."""
from __future__ import annotations


def add_parser(sub) -> None:
    p = sub.add_parser("online", help="job type 5: online learner, offline data (the authors' online loop on a bank of stored views)")
    s = p.add_subparsers(dest="online_cmd", required=True)
    for name, hlp in (("l1", "one block end to end, every term printed"), ("train", "the run")):
        q = s.add_parser(name, help=hlp)
        q.add_argument("--config", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--stub", action="store_true", help="stub components: no GPU, no simulator (tests only; the folder is marked stub)")
        q.add_argument("--device", default="cuda")
        q.add_argument("--seed", type=int, default=None)
    try:                                                            # a command that exists only where its module does
        import importlib.util
        has_gate = importlib.util.find_spec("wf_lib.online_live") is not None
    except (ImportError, ValueError):
        has_gate = False
    if has_gate:
        q = s.add_parser("render-gate")
        q.add_argument("--config", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--units", type=int, default=12)
    q = s.add_parser("status", help="the last log line of a run")
    q.add_argument("--out", required=True)
