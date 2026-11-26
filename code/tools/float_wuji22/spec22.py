"""The Hand 2.2 retarget configuration for rl/sim/multihand/retarget.HandRetargeter: the HandSpec and the MANO keypoint map.

HandRetargeter looks a hand up by name in two module-level tables (rl/sim/multihand/specs.SPECS and retarget.KEYPOINTS). Those two files
are digested by every older floating protocol (sharpa-float-p1 / -p2, wuji1hr-float-p1 / -p2), so they are not edited: importing this
module adds the entry "wuji22" to both tables in the running process, which is all the retargeter, tools/float_hands/hand_eval.Hand and
the 4r gate (tools/float_hands/fit4r) need. The protocol files wuji22-float-p* digest this file.

Keypoint map (MANO index -> body, kind), written from the geometry before any grasp was fitted: MANO's 21 joints are the wrist, then
thumb CMC / MCP / IP / tip, then MCP / PIP / DIP / tip of index..little. 'o' = the body's origin (a joint axis of the hand), 't' = the
farthest collision point of the body from its origin (the fingertip; retarget._tip_offset, as for every other hand).
  fingers   MCP <- r_<finger>_proximal (the MCP flexion axis)    PIP <- r_<finger>_middle    DIP <- r_<finger>_distal
            tip <- the distal body's farthest collision point    (the same names the Hand 2.1 entry "wuji2" uses)
  thumb     CMC <- r_thumb_proximal (the CMC flexion axis)       MCP <- r_thumb_middle      IP <- r_thumb_distal    tip <- r_thumb_distal
The two alternatives with a plausible anatomy (MCP <- the abduction link's origin, 14.5 mm further out; the tip pad body) are compared at
the open start pose by tools/float_wuji22/validate22.py before anything else is fitted, and the choice is recorded in its output.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from rl.sim.multihand import retarget as _rt
from rl.sim.multihand import specs as _specs
from tools.float_wuji22 import model22 as _m

REPO = Path(__file__).resolve().parents[2]
NAME = "wuji22"
FINGER_TOKEN = {"index": "index_finger", "middle": "middle_finger", "ring": "ring_finger", "pinky": "pinky"}
ACT = tuple(_m.actuator_names())                 # r_thumb_cmc_flex_act ... r_pinky_dip_act (the SDK's flat order)


def keypoints(mcp: str = "proximal", tip_body: str = "distal", thumb_cmc: str = "proximal") -> dict:
    """MANO keypoint -> (body, kind). mcp: the finger body whose origin stands for the MCP ('proximal' = the flexion axis, 'proximal_abd' =
    the abduction axis); thumb_cmc likewise; tip_body: 'distal' or 'tip_sensor_frame' (the pad body), whose farthest collision point is the tip."""
    kp = {1: (f"r_thumb_{thumb_cmc}", "o"), 2: ("r_thumb_middle", "o"), 3: ("r_thumb_distal", "o"),
          4: (f"r_thumb_{'distal' if tip_body == 'distal' else 'tip_sensor_frame'}", "t")}
    for k, f in enumerate(("index", "middle", "ring", "pinky")):
        n = FINGER_TOKEN[f]
        kp[5 + 4 * k] = (f"r_{n}_{mcp}", "o")
        kp[6 + 4 * k] = (f"r_{n}_middle", "o")
        kp[7 + 4 * k] = (f"r_{n}_distal", "o")
        kp[8 + 4 * k] = (f"r_{n}_{'distal' if tip_body == 'distal' else 'tip_sensor_frame'}", "t")
    return dict(sorted(kp.items()))


def hand_spec(name: str = NAME) -> _specs.HandSpec:
    flex = tuple(a for a in ACT if "_abd_act" not in a)
    return _specs.HandSpec(
        name=name, mjcf=_m.MJCF, root_body=_m.ROOT_BODY, palm_body="r_palm", actuators=ACT,
        finger_prefixes=(("r_thumb_",), ("r_index_finger_", "r_index_palm_cover"), ("r_middle_finger_", "r_middle_palm_cover"),
                         ("r_ring_finger_", "r_ring_palm_cover"), ("r_pinky_",)),
        palm_bodies=("r_mount", "r_wrist", "r_adductor_pollicis_cover", "r_thenar_cover", "r_midpalm_cover", "r_hypothenar_cover",
                     "r_palm_index_cover", "r_palm_middle_cover", "r_palm_ring_cover"),
        tip_bodies=tuple(f"r_{t}_distal" if t != "thumb" else "r_thumb_distal" for t in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")),
        # r_mount: fingers along -z, palm normal +y, thumb side +x  ->  the wuji palm frame (fingers +z, palm normal +x, thumb +y): x -> y, y -> x, z -> -z
        # (the rotation the URDF gives, 1.6e-5 off an exact permutation because its rpy are rounded to 5 digits; the exact one is used)
        root_to_wuji_palm=((0, 1, 0), (1, 0, 0), (0, 0, -1)), mount_z_m=0.0,
        flex_actuators=flex, thumb_actuators=ACT[:4],
        source_note="Wuji Hand 2.2 glove variant, hand only (assets/robot_description/hands/multihand/wuji22_float); vendor impedance actuators")


def register(kp: dict | None = None, name: str = NAME) -> None:
    """Add `name` to the retargeter's two tables (idempotent in a process; a different map under the same name is refused)."""
    kp = keypoints() if kp is None else kp
    spec = hand_spec(name)
    if name in _specs.SPECS and (_specs.SPECS[name] != spec or _rt.KEYPOINTS[name] != kp):
        raise RuntimeError(f"{name!r} is already registered with another spec or keypoint map in this process")
    _specs.SPECS[name] = spec
    _rt.KEYPOINTS[name] = kp


register()
