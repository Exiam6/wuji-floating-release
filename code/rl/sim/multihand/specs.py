"""Embodiment specs for the hands that can be mounted on the xArm7 in the grasp-lift scene.

`wuji1` is the Wuji WH110 that every existing result uses; its spec only *describes* the existing
hard-wired stack (rl/sim/scene.py, rl/sim/sim_hub.py) and is never used to build it, so the wuji1
path stays bit-identical. The other three hands are built by rl/sim/multihand/scene.py from the
vendored upstream MJCF (assets/robot_description/hands/multihand/<hand>/SOURCE.json) with the
build-time edits listed in each spec.

Conventions shared by all hands
  * Action = 7 xArm joint targets followed by the hand actuators in `actuators` order.
  * Finger ids: 0 palm, 1 thumb, 2 index, 3 middle, 4 ring, 5 little (the env's contact bitmap).
  * Friction groups (the NB "split friction" physics): `soft` geoms take the thumb/palm range
    (NB 1.0), every other hand collision geom takes the fingers range (NB 0.5). For wuji1 the
    vendor says palm + thumb are silicone, index..little links are hard shell. For the new hands:
    sharpa -> palm + thumb + the five fingertip elastomers (vendor class `elastomer_geom`);
    shadow  -> palm + thumb + the five distal (PST tip) geoms; wuji2 -> palm + thumb, as wuji1
    (the bare Hand 2.1 model marks no soft material, so the wuji1 split is assumed).
  * Mount: every hand is turned so that, in the xArm link7 frame, its finger direction and palm
    normal match wuji1's (fingers along +z of link7, palm normal along Rz(45 deg) x), and its
    root sits on the flange (wuji1 sits on its 49.8 mm adapter).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HANDS_DIR = REPO / "assets/robot_description/hands"

HAND_NAMES = ("wuji1", "wuji2", "sharpa", "shadow")


def _rz(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class HandSpec:
    name: str
    mjcf: Path
    root_body: str                      # attached at the link7 mount frame
    palm_body: str                      # pose stored as selected_transform / used by IK and rel_grasp
    actuators: tuple[str, ...]          # hand action order
    finger_prefixes: tuple[tuple[str, ...], ...]   # body-name prefixes for fingers 1..5
    palm_bodies: tuple[str, ...]        # bodies counted as finger 0 (palm)
    tip_bodies: tuple[str, ...]         # distal body of fingers 1..5
    soft_geom_names: tuple[str, ...] = ()           # extra soft geoms (beyond palm + thumb)
    soft_body_names: tuple[str, ...] = ()           # bodies whose geoms are all soft (beyond palm + thumb)
    # R maps hand-root axes into wuji1's palm_link axes (fingers -> +z, palm normal -> +x)
    root_to_wuji_palm: tuple = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    mount_z_m: float = 0.0              # root origin offset along link7 +z
    delete_joints: tuple[str, ...] = ()
    delete_actuators: tuple[str, ...] = ()
    delete_excludes_with: tuple[str, ...] = ()
    actuator_force_limits: dict = field(default_factory=dict)   # name -> |tau| max (N m), if the MJCF has none
    flex_actuators: tuple[str, ...] = ()           # actuators that close the hand (CANON-equivalent squeeze)
    thumb_actuators: tuple[str, ...] = ()
    source_note: str = ""

    @property
    def n_act(self) -> int:
        return len(self.actuators)

    def finger_of_body(self, body_name: str) -> int | None:
        if body_name in self.palm_bodies:
            return 0
        for k, prefixes in enumerate(self.finger_prefixes, start=1):
            if any(body_name.startswith(p) for p in prefixes):
                return k
        return None


_W1 = tuple(f"right_finger{f}_joint{j}_actuator" for f in range(1, 6) for j in range(1, 5))
WUJI1 = HandSpec(
    name="wuji1",
    mjcf=HANDS_DIR / "wuji_right_mjlab/mjcf/right_mjlab.xml",
    root_body="right_palm_link", palm_body="right_palm_link",
    actuators=_W1,
    finger_prefixes=tuple((f"right_finger{k}_",) for k in range(1, 6)),
    palm_bodies=("right_palm_link",),
    tip_bodies=tuple(f"right_finger{k}_link4" for k in range(1, 6)),
    mount_z_m=0.0498,
    flex_actuators=tuple(a for a in _W1 if not a.endswith("joint2_actuator")),
    thumb_actuators=_W1[:4],
    source_note="existing stack; spec is descriptive only",
)

_W2_FING = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
_W2_ACT = tuple(f"r_{p}J{j}" for p in ("TH", "FF", "MF", "RF", "LF") for j in range(4))
WUJI2 = HandSpec(
    name="wuji2",
    mjcf=HANDS_DIR / "multihand/wuji2/hand2/hand2_1/body/mjcf/right.xml",
    root_body="r_mount", palm_body="r_mount",
    actuators=_W2_ACT,
    finger_prefixes=tuple((f"r_{f}_",) for f in _W2_FING),
    palm_bodies=("r_mount", "r_wrist"),
    tip_bodies=tuple(f"r_{f}_distal" for f in _W2_FING),
    # r_mount: fingers along -z, palm normal +y, thumb side +x  ->  (x->y, y->x, z->-z)
    root_to_wuji_palm=((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    mount_z_m=0.0,
    # thumb: cmc_flex(J0), cmc_abd(J1), mcp(J2), ip(J3); fingers: mcp_flex(J0), abd(J1), pip(J2), dip(J3)
    flex_actuators=tuple(a for a in _W2_ACT if not a.endswith("J1")),
    thumb_actuators=_W2_ACT[:4],
    source_note="Wuji Hand 2.1 bare right.xml (MIT), rooted at the arm-flange mount",
)

_SH_FING = ("thumb", "index", "middle", "ring", "pinky")
_SH_ACT = (
    "right_thumb_CMC_FE_ctrl", "right_thumb_CMC_AA_ctrl", "right_thumb_MCP_FE_ctrl",
    "right_thumb_MCP_AA_ctrl", "right_thumb_IP_ctrl",
    *(f"right_{f}_{j}_ctrl" for f in ("index", "middle", "ring") for j in ("MCP_FE", "MCP_AA", "PIP", "DIP")),
    "right_pinky_CMC_ctrl",
    *(f"right_pinky_{j}_ctrl" for j in ("MCP_FE", "MCP_AA", "PIP", "DIP")),
)
# |effort| from the vendor URDF right_sharpa_wave_with_flange.urdf (the MJCF position actuators are
# unlimited; the URDF limits are applied as actuator force ranges).
_SH_EFFORT = {"CMC_FE": 3.3, "CMC_AA": 3.3, "MCP_FE": 1.864, "MCP_AA": 1.864, "IP": 0.638,
              "PIP": 0.638, "DIP": 0.189369, "CMC": 0.5285}
SHARPA = HandSpec(
    name="sharpa",
    mjcf=HANDS_DIR / "multihand/sharpa/wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.xml",
    root_body="right_hand_C_MC", palm_body="right_hand_C_MC",
    actuators=_SH_ACT,
    finger_prefixes=tuple((f"right_{f}_",) for f in _SH_FING),
    palm_bodies=("right_hand_C_MC",),
    tip_bodies=tuple(f"right_{f}_DP" for f in _SH_FING),
    soft_geom_names=tuple(f"right_{f}_elastomer" for f in _SH_FING),
    root_to_wuji_palm=((1, 0, 0), (0, 1, 0), (0, 0, 1)),   # same axes as wuji1 palm_link
    mount_z_m=0.0,
    actuator_force_limits={a: _SH_EFFORT[a.split("_", 2)[2].rsplit("_ctrl", 1)[0]] for a in _SH_ACT},
    flex_actuators=tuple(a for a in _SH_ACT if "_AA_" not in a and "pinky_CMC" not in a),
    thumb_actuators=_SH_ACT[:5],
    source_note="Sharpa Wave right, with flange (Apache-2.0)",
)

_SD_FING = ("rh_th", "rh_ff", "rh_mf", "rh_rf", "rh_lf")
_SD_ACT = ("rh_A_THJ5", "rh_A_THJ4", "rh_A_THJ3", "rh_A_THJ2", "rh_A_THJ1",
           "rh_A_FFJ4", "rh_A_FFJ3", "rh_A_FFJ0",
           "rh_A_MFJ4", "rh_A_MFJ3", "rh_A_MFJ0",
           "rh_A_RFJ4", "rh_A_RFJ3", "rh_A_RFJ0",
           "rh_A_LFJ5", "rh_A_LFJ4", "rh_A_LFJ3", "rh_A_LFJ0")
SHADOW = HandSpec(
    name="shadow",
    mjcf=HANDS_DIR / "multihand/shadow/shadow_hand/right_hand.xml",
    root_body="rh_wrist", palm_body="rh_wrist",
    actuators=_SD_ACT,
    finger_prefixes=(("rh_th",), ("rh_ff",), ("rh_mf",), ("rh_rf",), ("rh_lf",)),
    palm_bodies=("rh_wrist", "rh_palm"),
    tip_bodies=tuple(f"{p}distal" for p in _SD_FING),
    soft_body_names=tuple(f"{p}distal" for p in _SD_FING),
    # filled in by scene.py from the compiled wrist->palm transform (palm: fingers +z, normal -y)
    root_to_wuji_palm=None,
    mount_z_m=0.0,
    # The forearm is dropped and the two wrist joints are removed (the xArm supplies the wrist),
    # so rh_wrist + rh_palm become one rigid link on the flange: 18 hand actuators remain.
    delete_joints=("rh_WRJ2", "rh_WRJ1"),
    delete_actuators=("rh_A_WRJ2", "rh_A_WRJ1"),
    delete_excludes_with=("rh_forearm",),
    flex_actuators=("rh_A_THJ4", "rh_A_THJ1", "rh_A_FFJ3", "rh_A_FFJ0", "rh_A_MFJ3", "rh_A_MFJ0",
                    "rh_A_RFJ3", "rh_A_RFJ0", "rh_A_LFJ3", "rh_A_LFJ0"),
    thumb_actuators=_SD_ACT[:5],
    source_note="MuJoCo Menagerie Shadow Hand E3M5 right (Apache-2.0), forearm and wrist joints removed",
)

SPECS = {s.name: s for s in (WUJI1, WUJI2, SHARPA, SHADOW)}


def get_spec(name: str) -> HandSpec:
    if name not in SPECS:
        raise KeyError(f"unknown hand {name!r}; choose from {HAND_NAMES}")
    return SPECS[name]
