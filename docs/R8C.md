# Release revision r8c (2026-11-25): steering variants, learning-dynamics logging, and the online job type for the object-frame prior

r8c is r8b plus three things. **With none of the new keys or arms, every command behaves as in r8b, bit for bit**: training a config of r8b gives the same `na/endpoint.pt` and `na/readout_noise.npz`, byte for byte, and the default read-out arms are unchanged. A test checks this on a CPU stand-in run (`tools/r8c_1125/test_r8c.py` of the project's repository; its result is recorded in the tag message).

1. **The online job type** (`wf.py online ...`, docs/ONLINE.md) and the batch kind `online_readout`. Before this revision they were only in r7c (the r7a overlay of batch 4). Four changes make them work with the object-frame prior (`hug2obj*`), listed under [Online](#online-path-with-the-object-frame-prior).
2. **Optional training keys of `wf.py dsrl train`**, one variable each. They are the steering variants (noise map, the noise critic's input, a conservative penalty, the number of noise-critic heads), the opened recipe constants, checkpoints and a dynamics log.
3. **Read-out arms of `wf.py dsrl readout`** that only read a trained checkpoint, plus `--checkpoint CALLS`, `wf.py dsrl library` and the reader `code/tools/dynamics_report.py`.

The registered-recipe check (`assert_recipe` in `code/wf_lib/dsrl_cmd.py`) accepts each new key **only at the values listed here**. So does `r8c_problems`, which runs for every config, including ones without a training list. Any other value stops the command before anything is read.

## Why these variants (the diagnosis they test)

The project's audit of 2026-11-25 and the descriptive reading EX398 Stage A found the following:

- **The steering actor plays off the prior's typical set.** The actor's output is unscaled from a tanh box, w = bound · tanh(x).
  - At bound 1, the box [-1, 1]^99 cannot hold a vector of the norm-9.95 shell (sqrt(99)). Every N(0, I) noise the prior was trained on lies on that shell, as do every census noise and the best searched constants.
  - The trained actors play at norm about 4.3.
- **The noise critic is optimistic there.** Q^W, learned only on shell noises, rates the actor's off-shell point above its measured success.
- **There is headroom.**
  - A perfect per-view chooser over the 32 prior draws would add 16 points over the best single searched constant.
  - A perfect pick among 4 stored constants would add 13.
  - The actor reads its observation: about 10 points against a swapped observation. Yet it captures none of the headroom over the constant.

Each key below changes one of these factors.

## Training keys (`wf.py dsrl train` config)

| Key | Allowed values (first = r8b) | What it changes |
|---|---|---|
| `noise_param` | `"box"`, `"sphere"`, `"anchored"`, `"sphere_anchored"`, `"geodesic"` | the map from the actor's output to the executed noise w |
| `anchor_noise`, `anchor_noise_sha256` | a .npy of 99 floats and its sha256 (the anchored maps only; refused otherwise) | the anchor constant c |
| `max_angle` | `0.52`, `1.05`, `1.57` (radians; `geodesic` only) | the geodesic trust region |
| `critic_input_scale` | `"bound"`, `"raw"` (non-box maps: `"raw"` only) | what the noise critic sees |
| `critic_penalty` | `0.0`, `1.0`, `5.0` | conservative weight on the noise critic |
| `n_noise_critics` | `2`, `10` | heads of the noise critic |
| `learning_rate` | `3e-4`, `1e-4`, `1e-3` | all optimisers' learning rate (recipe deviation) |
| `utd` | `20`, `5`, `50` | gradient steps per call (recipe deviation) |
| `batch` | `256`, `1024` | batch size (recipe deviation) |
| `noise_critic_grad_steps` | `10`, `1`, `50` | distillation steps per call (recipe deviation) |
| `calls_deviation` + `offline_calls` | `false`; `true` with `offline_calls` `25000`, `50000` or `100000` | calls other than the calls rule (recipe deviation) |
| `checkpoint_every` | `2500`, `5000`; `25` only in tiny mode (`offline_calls` at most 100) | checkpoints during training (and resume) |
| `dynamics_log` | `false`, `true` | the dynamics log every 250 calls (every 25 in tiny mode) |

**Keys that already existed in r8b, under other names.** The coordinator's list names some keys differently:

| Asked for | r8b key | What r8b allowed |
|---|---|---|
| `calls` | `offline_calls` | fixed by the calls rule `calls_rule` (anchor 1,000 calls per 23,712 labelled executions, rounded to 250; 25,000 at 1,000 objects) |
| `utd` | `utd` | 20 |
| `batch_size` | `batch` | 256 |
| `noise_critic_steps` | `noise_critic_grad_steps` | 10 |
| `lr` | none | `learning_rate` 3e-4 was a learner constant (`RECIPE_MODEL`), asserted on the built learner but not read from the config; r8c adds the config key `learning_rate` |

`n_noise_critics` and `critic_penalty` are new. r8c keeps the existing names and opens their values; it adds no second name for them.

**Recipe deviations.** `learning_rate`, `utd`, `batch`, `noise_critic_grad_steps` and `calls_deviation` at any value except the first are deviations from the authors' DSRL-NA recipe. So are `critic_penalty` > 0 and `n_noise_critics` 10. They exist to measure the learning dynamics; a row that uses one is not the recipe row.

### `noise_param`: the noise map

The actor is the authors' squashed Gaussian. Write x for its pre-squash output (the Gaussian sample, or its mean when deterministic) and u = tanh(x) ∈ [-1, 1]^99 (the authors' action). R = sqrt(99), c is the anchor, b is the config's `bound`.

| Value | Executed noise w | Start (deterministic, step 0) |
|---|---|---|
| `box` (r8b, the authors') | w = b · u (the unscale to Box(-b, b)) | the authors' initialisation |
| `sphere` | w = R · u / \|u\| | the authors' initialisation |
| `anchored` | w = c + b · u | **c** (mean layer zero) |
| `sphere_anchored` | w = R · (c + b · u) / \|c + b · u\| | R · c/\|c\| |
| `geodesic` | ĉ = c/\|c\|; v = x − (x·ĉ)ĉ; θ = max_angle · tanh(\|v\|); w = R · (cos θ · ĉ + sin θ · v/\|v\|); w = R·ĉ when \|v\| < 1e-8 (a safe sinc form) | **R · ĉ** (mean layer zero) |

What each value tests:

- **`sphere`**: does keeping the executed noise on the shell help?
- **`anchored`**: does starting at the best global noise and steering around it help?
- **`sphere_anchored`**: both at once.
- **`geodesic`** is the preferred form of the steering variant. Every executed noise lies on the prior's typical set: the norm-9.95 shell where the flow was trained and where all census noises, hence all of the noise critic's training data, lie. The critic is therefore never queried off its support, and `max_angle` is a geodesic trust region around the best global noise. Its two ablations are `sphere` (no anchor) and `anchored` (no sphere).

**Which output the maps read.** `sphere`, `anchored` and `sphere_anchored` read u, the tanh output. It is bounded, so \|u\| ≤ R: the sphere's gradient factor R/\|u\| is at least 1, and no radial drift can make it vanish. These maps also keep the authors' entropy term on the same variable as their action. `geodesic` reads x, the raw output, because tanh(\|v\|) is its own squash.

**Where the policy starts.** For the anchored maps, the actor's mean layer (`actor.mu`, weight and bias) is set to zero after the whole learner is built, so no generator draw moves. The deterministic noise at step 0 is then the anchor (tested to 1e-6). The stochastic sample keeps the authors' log-std head.

**How the actor loss backpropagates.** The loss is the authors' (α · log π(u|s) − min_j Q^W_j(s, g(u or x))). The map g has no parameters, and the gradient flows through it by autograd:

| Map | Jacobian of g |
|---|---|
| `anchored` | b · I |
| `sphere` | (R/\|z\|)(I − ẑẑᵀ) · b, tangent to the shell |
| `sphere_anchored` | the same at z = c + b·u |
| `geodesic` | the projection (I − ĉĉᵀ) followed by the angle and direction terms; at v = 0 it is R · max_angle · (I − ĉĉᵀ), finite (tested) |

Then comes the tanh Jacobian of the authors' squash. The entropy term, its target (0) and its temperature are the authors', on u.

**What the noise critic and the replay data see.**

- **The critic.** For every non-box map the critic learns Q^W(s, w) on the executed noise w (`critic_input_scale` "raw", forced). Distillation pairs (N(0, I) pool or fresh draws, decoded by the prior) enter as they are: they are on the shell already. The actor's u is never stored, so no inverse map is needed (none exists on the sphere).
- **The box.** For a non-box map the box is Box(-1, 1). `bound` enters only as b of `anchored` / `sphere_anchored`, and must be 1.0 for `sphere` and `geodesic` (which do not read it).
- **The rest.** DSRL-NA's losses, distillation, schedule, optimisers and hyperparameters are unchanged.

**Saving and loading.** The checkpoint carries the variant (`r8c_variant`), and a read-out refuses a learner built otherwise. The anchor file is copied into the training folder (`anchor_noise.npy`), and its sha256 is checked at every load.

### `critic_input_scale`

r8b's critic sees scale_action(w) = w / bound. Changing `bound` (EX396 row d, bound 3.0) therefore changes two things: the actor's box and the critic's input scale. `"raw"` makes the critic see w itself under the box map, so a bound row changes only the actor's box. For every non-box map it is forced.

The change applies in three places, all in `code/tools/dsrl_port/dsrl.py`:
- the distillation input, `update_noise_critic` (`critic_in_w`);
- the actor loss (`critic_in_u`, a differentiable unscale);
- `q_noise`.

### `critic_penalty`: a conservative noise critic (CQL-style; not the authors')

This tests whether Q^W's optimism is what misleads the actor. It is added in `update_noise_critic` (`code/tools/dsrl_port/dsrl.py`, the block after the distillation loop):

    L_W = sum_j 0.5 · MSE(Q^A_{j mod 2}(s, a(w)), Q^W_j(s, w))
          + alpha · sum_j mean_b [ logsumexp_{k=1..N} Q^W_j(s_b, z_bk) − Q^W_j(s_b, w_b) ]

Here z_bk ~ N(0, I) with N = 10 per row, drawn on the CPU generator like the authors' fresh draws, and w_b is the row's distillation noise (the "data noise").

**Caveat.** The penalised samples are N(0, I) draws, which lie on the shell, and so does w_b. Under the box map the term therefore flattens Q^W *on* the shell. It does not push Q^W down at the actor's off-shell point. Its intended reading is together with the on-shell maps (`geodesic`, `sphere`), where the actor's noise is a shell point too. Under `box` it tests only "is Q^W's spread over the prior's own draws the problem". The training log adds `noise_critic_penalty` when alpha > 0.

### `n_noise_critics`

This is the noise critic's head count: 2 is the authors', 10 gives a pessimistic min over a larger ensemble. Head i regresses head (i mod 2) of the action critic, which is exactly the authors' pairing when there are 2 heads. The actor loss, the read-out scores and q_noise use the min over all heads. It tests whether a larger ensemble's lower bound removes the optimism.

### `learning_rate`, `utd`, `batch`, `noise_critic_grad_steps`, `calls_deviation`

These are the opened recipe constants, for EX403's learning-dynamics rows. They are passed to the learner (`na.build` `learner`), and the training asserts that the built learner carries them. `calls_deviation: true` replaces the calls rule with an explicit `offline_calls` of 25,000, 50,000 or 100,000. The diagnostics record it, and the epochs are reported as before.

### `checkpoint_every` and `dynamics_log`

Both are for EX403, the learning dynamics at 1,000 objects. Every evaluation runs inside `dynamics.preserved_rng()`, with no gradient taken, so a run with them ends at the same endpoint, bit for bit, as without them (tested).

**Checkpoints.** With `checkpoint_every`, `<train-out>/checkpoints/<calls>/` is written at every multiple of `checkpoint_every` below the last call. It holds:
- `endpoint.pt`: networks, optimisers, entropy coefficient and update count;
- `readout_noise.npz`: the actor's noise at every state;
- `checkpoint.json`: the optimisers' step counts.

The registered endpoint stays the last call (`na/endpoint.pt`). `wf.py dsrl readout --checkpoint CALLS` reads a checkpoint instead.

**Resume.** A run started with `checkpoint_every` writes `resume_key.json`: the config file's sha256 and the command-line overrides. Rerunning **the same command** on the same `--out` folder continues from the last complete checkpoint instead of refusing the existing folder. The resume restores:
- the networks and both optimisers;
- the entropy coefficient and the update and call counters;
- the numpy, python, torch CPU and CUDA generators (so the replay sampler and the pool or fresh draws continue their streams);
- the logs, cut back to that call.

A checkpoint is complete when its `checkpoint.json` is written (last); an incomplete one is removed. The replay data are rebuilt from the prep, which uses no random numbers. On the CPU stand-in, a run stopped during call 3 and resumed from checkpoint 2 ends at the same `endpoint.pt` and `readout_noise.npz`, byte for byte, as an uninterrupted run, with the same dynamics and training-log lines (tested). On a GPU the same holds as far as the kernels are deterministic; it is not tested there.

A different config or override is refused, as is a finished run (`train_run.json`), and so is any folder of a run without `checkpoint_every`, exactly as in r8b.

**Resume through `wf.py batch run`.** Rerun the same `wf.py batch run` command after an interruption. An unfinished `dsrl train` step whose folder holds the resume key of its own command is not moved to `_failed/` (`batch_cmd.resumable_in_place`). Its command runs again in place, and the training continues from the last complete checkpoint; the batch log says `train: resumed from checkpoint N`. Every other unfinished step is moved aside and run again, as in r8b. On the CPU stand-in, a batch killed after its first checkpoint and rerun ends at the same `endpoint.pt` and `readout_noise.npz`, byte for byte, as an uninterrupted batch (tested).

**The dynamics log.** `dynamics_log: true` appends one line to `<train-out>/dynamics.jsonl` every 250 calls (every 25 calls in tiny mode, a training of at most 100 calls, so a rehearsal exercises the log, the checkpoint read-out and the resume). The fields are defined in `code/wf_lib/dynamics.py`:

- **Losses.** The per-head TD error of Q^A on 4,096 fixed data rows (one-step MDP, so the target is the reward), the per-head distillation error of Q^W on 2,048 fixed pool pairs, and the training losses averaged over the last 250 calls (with the penalty when on).
- **Values.** Mean and standard deviation of Q^A on data actions, of Q^W on data noises and of Q^W at the actor's noise.
- **The actor's noise.** Its norm (5 / 50 / 95 %) and the share of coordinates at the bound. For a non-box map, that is the share of u's coordinates at ±1.
- **The anchor.** The cosine of the actor's mean noise to the anchor, and the median per state.
- **Entropy and alpha.**
- **AUC.** The within-input AUC of Q^W over the census's own label-noise rows of a FIXED set of 32 training views, drawn with default_rng([403000, seed]). This is in-sample: the critic has seen their labels through Q^A. They are never test objects. It needs the prep's `actions.npz` `noise`.
- **Wall clock and GPU memory.**

## Every new key at a glance

| Key | Allowed values | Default (= r8b) | Diagnosis it tests |
|---|---|---|---|
| `noise_param` | box, sphere, anchored, sphere_anchored, geodesic | box | the actor's box excludes the norm-9.95 shell (actors play at 4.3); start from the global answer |
| `anchor_noise` + `anchor_noise_sha256` | a 99-float .npy and its sha256 | absent | the anchor of the anchored maps (the searched constant) |
| `max_angle` | 0.52, 1.05, 1.57 | absent | geodesic trust region around the anchor |
| `critic_input_scale` | bound, raw | bound (non-box: raw) | `bound` changes the actor's box and the critic's input at once |
| `critic_penalty` | 0.0, 1.0, 5.0 | 0.0 | critic optimism at the played noise |
| `n_noise_critics` | 2, 10 | 2 | pessimism of a larger ensemble |
| `learning_rate` | 3e-4, 1e-4, 1e-3 | 3e-4 | learning dynamics (recipe deviation) |
| `utd` | 20, 5, 50 | 20 | learning dynamics (recipe deviation) |
| `batch` | 256, 1024 | 256 | learning dynamics (recipe deviation) |
| `noise_critic_grad_steps` | 10, 1, 50 | 10 | distillation strength (recipe deviation) |
| `calls_deviation` + `offline_calls` | false; true with 25000, 50000, 100000 | false (the calls rule) | training length at 1,000 objects (recipe deviation) |
| `checkpoint_every` | 2500, 5000; 25 if offline_calls ≤ 100 (tiny mode) | absent | learning curve; resume of long jobs |
| `dynamics_log` | false, true | false | losses, values, norm, AUC every 250 calls |
| read-out `--arms` | rerank32, rerank8, swap_obs, frozen_obs, mean_obs, lib4, lib8 | not run | selection by Q^W; does the actor use its observation; library + selector |
| read-out `--checkpoint` | a checkpoint's calls | the endpoint | the learning curve |

## Read-out arms (`wf.py dsrl readout --arms ...`, read a trained checkpoint, no training)

Each arm writes `readout_<arm>.npz` (`noise` per unit, like `own`) and its rollout jobs `<arm>/<unit>`, so its labels come back like `own`'s.

**Rerank arms.**
- **`rerank32`, `rerank8`.** For each input, the candidates are the K label noises that the census rule draws for that input (`--census-noise shared|fresh|fresh64`, `--census-base-seed`, default 20261116): draws 0..K−1, so rerank8's 8 are a prefix of rerank32's 32. Each candidate is scored by the min over the noise critic's heads, Q^W(s, w), and the argmax is executed (ties: the lower draw index). The file adds `choice` and `cand_q`.
- **Labels without rolling.** The candidates are the census's own rows, so their labels already exist wherever that census was rolled. pass@32 is computed from the prior arm's census rolls and is not an arm.
- **What it tests.** Selection by the learned verifier instead of regression (the EX401 V2 variant).

**Observation-control arms.**
- **`swap_obs`.** The actor's deterministic noise computed from ANOTHER read-out input's observation, decoded at this input. The pairing is a fixed derangement: default_rng(0), within each camera level when possible, never the same object. The file adds `source_id`, and the read-out record adds the same-level share.
- **`frozen_obs`.** The actor's noise for one fixed observation (the first training view) at every input.
- **`mean_obs`.** The actor's noise for the mean observation of the training views.
- **What they test.** Whether the gain uses the observation, and how much of the actor is a constant.

**Library arms.**
- **`lib4`, `lib8`.** A library of K constants, with the selector argmax over the K of min Q^W.
- **The library.** `wf.py dsrl library --search F1,F2,... --k 8 --out lib.npz` builds it from searched-constant folders (every draw of every CEM generation, with its labels on the search inputs). The rule is **greedy coverage** (EX398 Stage A (e)): first the draw with the highest mean training success; then, repeatedly, the draw that succeeds on the most search inputs not yet covered. The order is nested: lib4 is the first 4 entries of lib8.
- **Exclusion.** The library's search units are checked against the forbidden-id file (`--forbidden-ids`, or the training folder's).
- **What it tests.** A small library with a learned selector (+13 points for a perfect pick among 4).

**`--checkpoint CALLS`.** Any arm that reads the actor or the critic reads that checkpoint instead of the endpoint.

**`code/tools/dynamics_report.py`.** It reads `dynamics.jsonl` and, per checkpoint, the read-out and its rollout results, then writes:
- `curves.json`;
- PNGs: success on set 1 / set 2 / any listed object group vs calls, Q^W against measured success, norm, AUC, losses.

## Runtime cost

Ratios to the recipe training of the same row. The rows marked *measured* come from one train() call (utd 20, batch 256, 10 distillation steps) on the CPU stand-in: 3 × 256 network, 2 threads, pool path. The others are estimates.

| Key / arm | Cost |
|---|---|
| `noise_param` (any), `critic_input_scale` | *measured*: `geodesic` 1.01×; the maps are O(batch · 99) |
| `critic_penalty` > 0 | *measured*: 1.81× a pool-path call (10 more Q^W rows per distillation row); on the fresh path the decoder dominates, so the share is smaller (estimate) |
| `n_noise_critics` 10 | *measured*: 2.31× a pool-path call; the noise critic and its optimiser state are 5× larger (at 3 × 2048 about +0.8 GB of GPU memory, estimate) |
| `utd` 5 / 50 | the critic and actor part × 0.25 / × 2.5 |
| `batch` 1024 | up to × 4 per step (less on a GPU that is not saturated) |
| `noise_critic_grad_steps` 1 / 50 | the distillation part × 0.1 / × 5 (fresh path: the decodes × 5) |
| `calls_deviation` | the calls × 2 or × 4 |
| `checkpoint_every` | per checkpoint one forward pass over every state and one file the size of `na/endpoint.pt`: seconds |
| `dynamics_log` | per line about 10,000 forward rows: under a second |
| `rerank32` / `rerank8`, `lib4` / `lib8` | K Q^W evaluations per input (milliseconds) plus one decode and one rollout per input, as `own` |
| `swap_obs`, `frozen_obs`, `mean_obs` | as `own` |
| `dynamics_report.py` | seconds, CPU |

## Online path with the object-frame prior

r7c's online overlay (batch 4; `docs/ONLINE.md`) and its batch patch (group kind `online_readout`) are in r8c unchanged. Four edits make the online loop work with an object-frame prior id. For every other prior each edit is the r7c code path by construction: the branch is taken only when the decoder says `object_frame` or the config's prior is one of `paths.OBJECT_FRAME_PRIORS`.

1. `online_real.BankSource._load`: the bank view dict carries the stored picture's `object_mask` (object-frame prior only).
2. `online_real.RealPrior.encode`: `make_frame(..., object_mask=view["object_mask"])`. A bank picture without a mask stops with one line.
3. `online_real.RealPrior.flow_fn` / `.decode`: the learner's action is `normalize(flow_native(cond, w))`, the model's own row relative to the object centre, like the census's `flow_output99`. The grasp is decoded from the camera-frame rows `shift_translation(raw, object_center)`.
4. `online_cmd.components`: the encoding's L × D is measured on one stored view (`BankSource.probe_view`, `RealPrior.encoding_shape`) instead of the old prior's 256 × 1,024.

The online config has no registered-recipe check of its own that names a prior (`online_cmd.load_cfg`, `online_core.RECIPE`). It runs under the object-frame prior ids as under `hug2`.

**Not changed: the census exploration control.** The stream control `exploration: census` (`RealPrior.census_noise`) still draws the shared-rule noise `<object>_b0`. A census of the object-frame prior made with `fresh64` label noise has other rows. That control is not part of the online cell.

The steering keys above are `dsrl train` keys. The online learner (`online_core.build_learner`) does not read them.
