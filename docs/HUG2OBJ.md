# The object-frame HUG2 prior (release revision r8)

Revision r8 (and its pin update r8a) adds a second HUG2 checkpoint family to r7a: a model that reads only the object's point cloud (from a depth picture and an object mask) and predicts the grasp's translation
relative to the object's centroid. Everything of r7a keeps its behaviour: the prior ids `hug2`, `hug1` and `dumbbell50k`, their pins, their defaults and their results are unchanged (the old
decoder is the same code path; `make_frame` carries one more optional field the old decoder never reads).

| Prior id | File (sha256) | What it is |
|---|---|---|
| `hug2obj_100k` | `ckpt/hug2obj/hug_objpcl_aug_100k.pt` (3fd123112931...) | the final checkpoint: step 100000, EMA export (an average over the last 50,000 steps), bf16, 10 Euler steps; release copy as below (the decoder also accepts the author's file, sha256 5dcd845dad91...) |
| `hug2obj_50k` | `ckpt/hug2obj/hug_objpcl_aug_50k.pt` (c4ef201487ad...) | step 50000, EMA export (the EMA had just begun), bf16 weights, 10 Euler steps; the release copy of the author's file: the tensors are bit-equal, four training-config strings that name the author's machines and account are replaced (the decoder also accepts the author's own file, sha256 350f41313b9d...) |
| `hug2obj` | `ckpt/hug2obj/hug_objpcl_aug_15k.pt` (5738b4d9aa2d...) | step 15000, RAW weights, bf16 (release copy; the author's file has sha256 929a58e54b26...): a separate id, never swapped with another checkpoint |

Both ids use the model authors' code at commit `44ad0250779db5366c938d2c71f0a5467e007770`, integrated as our own package `code/tools/float/objframe/` (ten modules under our names, two small data files, the MIT licence text; one header per file
names the origin commit and file; `PROVENANCE.json` records the sha256 of every original and of every copy and the exact import edits, so the fidelity claim can be checked). The decoder refuses a checkpoint whose
sha256 or layout differs from its pin and a prior id it does not know. The user's own MANO model is read from `$WF_MANO_ROOT` (set by the decoder's builder from the asset root's `mano` folder).

What the decoder does differently: the stored picture's object mask selects the cloud (4,096 points, 1-pixel erosion, 3 m depth limit); the query point is the cloud's centroid; the
network's translation is relative to that centroid and the decoder adds it back, so the records (`T_camera_wrist`, `mano_params99`, landmarks) are in the camera frame as for the old
prior. `flow_output99` of a record is the model's own row (translation relative to the centre) because every consumer normalises it with the model's statistics; `flow_output99_camera` holds the
shifted rows. A census, DSRL or read-out job that names a prior of this family needs that prior's own census (the stored labels of another prior do not apply).

## The seven `make_frame` call sites that now pass the stored object mask

1. `code/wf_lib/cem_cmd.py`, `decode_gen()` (line 168)
2. `code/wf_lib/census_cmd.py`, `decode_chunk()` (line 596)
3. `code/wf_lib/census_cmd.py`, `tokens_chunk()` (line 757)
4. `code/wf_lib/census_cmd.py`, `encode_stage()` (line 890)
5. `code/wf_lib/decode_cmd.py`, `decode_units()` (line 94)
6. `code/wf_lib/dsrl_cmd.py`, `fresh_flow()` (line 290)
7. `code/wf_lib/dsrl_cmd.py`, `readout()` (line 610)

(`ast`-extracted from this folder by the builder; the mask is read from the stored picture's `object_mask`; a picture without one makes the object-frame decoder stop.)

## Other changes in `code/wf_lib`

`--prior` on `wf.py decode run` (two more choices) and on every `wf.py census` command (default `hug2`; a census plan made for one prior refuses to run under another),
`paths.OBJECT_FRAME_PRIORS` and `paths.prior_row`, the canary prior list, the census stamp name `decode/preds_<prior>.npz`,
and `na.py`'s `flow_native` / `normalise_params(native=, centers=)` (an object-frame row normalised without its frame is refused).

## Licence status

The authors' MIT licence text and copyright notice are in `LICENSES/hug2obj-source-LICENSE.txt` and `code/tools/float/objframe/LICENSE`. Status of the permission to share this checkpoint and this code: **granted by the model authors (relayed 2026-11-25); MIT, as their repository says**. 
