# Release r8d: steering as a residual on the prior's output (optional)

r8d is r8c with one optional training key. With the key absent, training and read-out are those of r8c, bit for bit.

* `action_space` (config of `dsrl train`): `noise` (default: the actor's output is the prior's starting noise, the authors' DSRL-NA) or `residual`.
  With `residual`, the prior is decoded from zero noise and the actor's output u in [-1, 1]^99 is added to the prior's normalised output as `res_halfwidth * u`
  (`res_halfwidth`, a positive number, is required). The actor is trained against the action critic at that point; every other step of the learner is the
  authors' `train` (code/wf_lib/residual.py generates it from their source with the one actor line replaced). Needs `noise_source: fresh` and the box map.
* Read-out arm `randres` (residual actors only): a residual of the same length as the actor's, in a random direction fixed by the input's name.
* Read-out files gain a column `delta` (the residual of each row; zeros for noise arms).
