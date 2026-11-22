"""Floating evaluator for the rl/sim/multihand hands (Sharpa Wave, Wuji Hand 2, Shadow; Wuji Hand 1 for diagnosis
only), protocols `<hand>-float-p1` (first: sharpa-float-p1, 2026-11-19).

hand_eval.py      the harness mechanics of the frozen October Wuji evaluator (tools/float_wuji/wuji_eval.py) with the
                  hand generalised, from tools/ex210_1112/hbport210.py (EX210 harness port v2): scene, floating wrist,
                  kp-30 finger actuators, contact mapping, close rule; the fingers and the hand-root mount from
                  rl/sim/multihand/retarget.HandRetargeter (the retargeter of every table-arm result on these hands)
rollout_hand.py   handoff A2, validity, stages, diagnostics, persistent pool (as tools/float_wuji/rollout_wuji.py)
protocol_hand.py  the protocol file's digests and their verification (base protocol, code, hand files, retargeter files)
Object-agnostic: no object name or per-object branch.
"""
