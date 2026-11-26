# Release r8f: the floating Wuji Hand 2.2

r8f is r8e plus a fourth floating hand. Without it, every command is that of r8e, bit for bit (the MANO, Sharpa and Wuji Hand 1 protocols, their
reference results, their canaries and every r8e command are unchanged).

* Protocol `wuji22-float-p2`: the Wuji Hand 2.2 (glove variant, the vendor's impedance actuators, the hand server's 3 rad/s command ramp and the
  deployed over-close) on the floating wrist that carries the hand's weight (mano-1111-p2's rule). Code: `code/tools/float_wuji22/`; protocol file:
  `code/protocols/wuji22-float-p2.json` (its digest covers the code, the hand files, the retargeter files and the drive numbers).
* Hand model: asset part `hands22` (56 files below `hand_models/assets/robot_description/hands/multihand/wuji22*`: the URDF, the vendor actuator
  recipe, the self-collision policy, the right hand's collision meshes, the converted MJCF and the two source records; licences: wuji-description MIT
  (Wuji Technology) for the hand, xarm_description BSD-3-Clause (UFACTORY) for the arm kinematics the URDF keeps, wuji-mjlab Apache-2.0 for the
  actuator numbers; texts in `LICENSES/`). Unpack it into the asset root, then `python code/wf.py assets link-hands` as for the part `hands`.
* `wf.py rollout verify | run --protocol wuji22-float-p2`; `wf.py census plan --protocol wuji22-float-p2` labels a census with the Hand 2.2.
* Known answers: `reference/cpu/hands22/` (200 grasps rolled under this protocol by the project before the release copy; every Hand 2.2 roll
  carries canaries drawn from them, and `wf.py verify --levels L1h` checks them bit for bit when the part `hands22` is present).

Protocol digest in this release: `9dea9c235e184cb26e3c1ff559d4361f3ca7cfc7a06523143df6ed7715868e95`.
