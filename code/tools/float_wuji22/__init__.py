"""Floating evaluator for the Wuji Hand 2.2 (the hand on the real robot since 2026-11-20): protocols wuji22-float-p1 / -p2.

The scene, the floating wrist, the plan, the lift, the success label and the retarget gate are those of tools/float_hands
(protocol sharpa-float-p1 and its p2 version: mano-1111-p1's evaluator with the MANO capsule hand replaced); what changes
is the hand and how it is driven:

model22.py     the hand model: the rrc / vendor Hand 2.2 URDF (glove variant, the copy EX337 pinned under
               assets/robot_description/hands/multihand/wuji22) converted to a hand-only MJCF with the vendor's self-collision
               policy, written once to assets/robot_description/hands/multihand/wuji22_float/
spec22.py      the retarget configuration (rl/sim/multihand HandSpec + MANO keypoint map for Hand 2.2), registered into the
               retargeter's tables at import time, without editing the files that older protocols digest
hand_eval22.py the scene builder with the vendor impedance actuators of EX337's model (kp = Kt x 1.9115 x 3.0, kv, 1.5 A cap),
               the hand server's 3 rad/s ramp and the deployed over-close profile; every other function is tools/float_hands'
rollout22.py   job runner and pool (tools/float_hands/rollout_hand.run_job with this module's hand and scene)
protocol22.py  the protocol files and their digests
Object-agnostic: no object name or per-object branch.
"""
