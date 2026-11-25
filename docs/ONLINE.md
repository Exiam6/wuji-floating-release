# Job type 5: online learner, offline data (`wf.py online`)

In the offline setting the actor is trained on a census made before training: every view and its labelled grasps are stored. In this job type the actor plays its own noise while it learns, as in the
authors' online DSRL loop: each environment step takes a view of a training object, encodes it with the prior, lets the actor choose the starting noise, decodes the grasp, rolls it out under the release
protocol and returns 0 or 1 (native success) as the reward. Training uses the authors' update unchanged, including their fresh-noise distillation of the noise critic.

**The views come from a bank of stored pictures** of the training objects at random camera and placement settings. Nothing is rendered during a run, and no renderer is part of this release. The loop is
online in everything after the picture, and offline in the pictures: *online learner, offline data* (an online learner on stored views). A comparison with the offline census at the same number of labelled
grasps and train calls is a comparison of two data sources for the same learner. It does not say anything about views rendered during the run.

## Use

1. `python code/wf.py online l1 --config C --out DIR` runs one block end to end and prints every term (view, encoding, state, actor noise, decode, rollout labels, reward, one train call, the number of
   fresh decoder calls). Run it first.
2. `python code/wf.py online train --config C --out DIR` runs the training to its budget (environment steps, GPU hours, core hours, wall hours; the first one reached stops the run, and `stop` in
   `online_run.json` says which). When the step budget ends the run, it ends with exactly `floor(transitions / rollouts_per_call)` train calls: the loop requests round(steps / (B x K)) blocks, lets the rollouts in flight finish and makes every call that is due. The hour limits (GPU, core, wall) stop a run at once and can leave calls owed; `online_run.json` records how many were made.
3. `python code/wf.py dsrl readout --train-out DIR ...` reads the actor out exactly as for an offline run (the folder has the files that command reads).

The config keys are listed at the top of `code/wf_lib/online_cmd.py`. The learner constants are the registered offline recipe and a config that changes them is refused. `forbidden_ids_file` and its
sha256 are compulsory: the training objects and the rows used to standardise the observation are checked against it before anything is created, and an empty forbidden list is refused. `exploration`
`prior` and `census` are stream controls (fresh N(0, I) noise, or the census's own noise rows of the object, instead of the actor's): they exist to separate on-policy collection from the data stream.
`online_run.json` records `labelled_grasps`, `calls`, `rollouts_per_call`, `seed`, `exploration`, `stream_control`, `view_source`, `bank_max_slot`, `state_mode`, `noise_source`, `stop`, the number of
rollouts whose simulation was flagged invalid, and the learner constants.

## The bank

The bank is a data part, not part of this overlay: it is delivered with the batch that uses it (64 stored views of each training object, the census views being its first 19 slots, with its input set and the stored pictures).

An input set of the release's format (`wf-input-set-1`: units with `unit`, `cluster`, `unit_key`, `asset`, optionally `params`) and the input store its units name. A unit's `cluster` must be the object its
id names, and a unit's `params` (the five view axes) must lie inside the ranges in `code/wf_lib/view_ranges.json`. Those ranges are the per-axis minimum and maximum over the 19 camera levels; each level
moves one axis, so a bank drawn uniformly over the box contains combinations that none of the 19 levels has. The loop does not check how a bank was drawn, only these properties. An object with fewer than
`views_per_block` views is sampled with replacement (the run prints a notice). `bank_max_slot` keeps only the units `rv000` to `rv<N>` of a bank that numbers its views that way.

## Needs (per run, one GPU)

* GPU: the class used for the census (the known-answer bars of the rollout canaries are measured on two GPU models; another model needs the census self-test). 2 GiB for the prior, 1 MiB per cached view
  (4.3 GB for 4,096 views, `cache_views`), about 0.5 GB for the learner.
* Host memory: the replay window holds `cache_views` x `rollouts_per_view` transitions of about 8.8 KB (observation and next observation, 1,024-D float32, plus the 99-D action): 1.2 GB for 131,072
  transitions; two such buffers are allocated at start (only one is filled): reserve 2.5 GB.
* CPU: the CPU environment of the release (MuJoCo 3.10.0); 4 cores per GPU are enough (1 for the trainer, 2 to 3 for rollouts). Measured by us on a shared GPU and a loaded host, one measurement each: encode
  0.07 to 0.24 s per view, decode of 8 noises 0.1 to 0.2 s, one rollout 0.26 s inside a worker and 0.39 core-s in object-grouped chunks, one train call 5 to 7 s; 6,500 to 10,000 transitions per GPU-hour at
  8 to 32 rollouts per view. A run of 23,712 transitions therefore takes roughly 2.5 to 4 GPU-hours on such a card.
* No private file and no network access are used. Runs are independent: seeds and cells go to different GPUs.

## Deviations from the authors' loop (registered)

* The observation is the standardised mean of the prior's scene encoding (1,024-D), not their images and proprioception; the standardisation rows are named in the config.
* The replay is a sliding window of `cache_views` x `rollouts_per_view` transitions (the authors' buffer is unbounded at these sizes); every transition is terminal. The authors' `sample()` fails once a ring
  has wrapped; here it samples uniformly over the window (identical before the first wrap).
* Training is gated by the data: one train call per `rollouts_per_call` (23.7) new transitions, i.e. 20 / 23.7 = 0.84 gradient steps per transition. The authors' schedule makes 20 per step of 4
  environments, 5 per transition.
* Warm-up: the first `warmup_transitions` completed transitions use noise uniform in the action box (their `learning_starts` is 1); blocks already in flight when the count is reached are warm too.
* Rollouts of one block (several views of one object) are run together, because a worker keeps one object's scene.
* Pacing (`max_lead_calls`, unset by default): data collection may run at most this many train calls ahead of the learner, so the actor that plays a view is at most about `max_lead_calls` + `prefetch_blocks` x
  (views per block x rollouts per view) / `rollouts_per_call` calls behind the learner when that view's labels arrive. The bound is on how far collection may run ahead, not a guarantee on that lag.
* `state_mode` (`full` by default: the standardised 1,024-D pooled encoding; `pca64`: the 64-D projection stored in the prep's `states.npz`) and `object_sampling` (`uniform` by default; `balanced`: the blocks
  go through the objects in random cycles, so every object gets the same number of blocks to within one).
* Reproducibility: block objects and the stream controls are counter-based. The order in which rollouts finish and the actor's noise (global torch generator, shared with the train calls) are not, so a
  run is one sample of a stochastic process; report each seed.

## Limits of the evidence (read before relying on a result)

What was run, with the real prior and real rollouts: one block end to end; two runs of 1,216 transitions on 3 training objects with stored views (K = 8, B = 4, 51 train calls; all checks written beforehand
passed except one prediction, "the noise-critic loss falls", which failed: the loss rose with the spread of its targets); a run of 4,032 transitions (170 train calls) on the 40 training objects with views taken from
another source, paced with `max_lead_calls` 8, in which the actor that played a view was at most 21.0 calls behind the learner (median 6.9; the request-ahead bound 8 + 3 x 96 / 23.7 = 20.15 is not a guarantee on the lag);
and a run of 600 transitions on 3 objects whose folder was read out by `dsrl readout` (arms own, stoch and a random-noise control; 368 inputs each, every simulation valid). The window's mean reward rose from 0.47 to
0.56 in the first runs; no improvement of the read-out is shown by any of them. **Not run before this release: a run with the stored-view bank and the full observation through `wf.py batch run`, at 40 or at 200 objects, and the read-out of
such a run**: the batch that uses this release carries its own tiny check of that path. The known-answer jobs of the release are not run inside the loop (a config asking for them is refused). Timings above are shared-card values.
