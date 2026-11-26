# Release r8e: a census labelled with a robot hand (optional)

r8e is r8d with one optional census option. Without it, every command is that of r8d, bit for bit.

* `census plan | run --protocol NAME`: the label protocol of a new plan, one of `mano-1111-p2` (default, the shipped census's), `sharpa-float-p2`,
  `wuji1hr-float-p2` (the floating Sharpa Wave and Wuji Hand 1 with the carrying wrist; their hand models come with the asset part `hands`). The plan records
  the protocol; decode, roll and finish read it from the plan, and refuse a `--protocol` that differs from the plan's.
