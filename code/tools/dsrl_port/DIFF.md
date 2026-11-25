# DSRL-NA port: differences from the authors' code

Upstream: github.com/ajwagen/dsrl @8d21b9cf55459f022c0e769bca1343edd9d30e5a (no license file at the top level), DSRL-NA
implementation in its submodule github.com/ajwagen/stable-baselines3-dsrl @10e5d311b5c42fe571ff3eeb72f622ca616a149d
(`stable_baselines3/dsrl/dsrl.py`, MIT; the copyright notice is in LICENSE_stable-baselines3-dsrl). 
Line-by-line check: `check_against_upstream.py` (our lab's L0 check, not shipped).

## Port (`dsrl.py`, `sb3_parts.py`) vs upstream

| Where | Upstream | Port | Why / effect |
|---|---|---|---|
| Algorithm shell | `DSRL(OffPolicyAlgorithm)`: env loop, callbacks, logger, save/load | plain class; explicit replay buffer; `train()` returns a dict | our transitions come from simulator rollouts outside a gym loop; the update math is untouched |
| `train()` backup | always computes the next-state backup: actor sample on next obs → unscale → diffusion policy → target critics → min → minus entropy, × (1 − done) | same, except when every sampled `done` is 1: diffusion call and target critics skipped, next_q = 0 (the actor sample on next obs is still drawn) | one-step MDP: (1 − done) = 0, so the target is r bit for bit (test T1) and the torch RNG stream is unchanged; saves a HUG2 call per gradient step |
| `update_noise_critic` noise | fresh `th.randn` on the CPU generator, denoised by the diffusion policy every step | `noise_source="fresh"` (identical) or `"pool"` | `"pool"` = EX136's registered cost deviation: pairs drawn from 1,024 precomputed N(0, I) draws per input and their HUG2 outputs (HUG2 takes ~0.84 s per 256 draws) |
| `_update_learning_rate` | sets each optimizer's lr from a constant schedule | not called | no-op with a constant learning rate |
| Batch-norm stats Polyak copy | `polyak_update(batch_norm_stats, …, 1.0)` | dropped | the networks have no batch-norm running statistics (empty lists upstream too) |
| `use_sde` / `reset_noise`, `_vec_normalize_env` | present | dropped | gSDE off and no VecNormalize in every DSRL config |
| SB3 base classes | `BasePolicy` / `BaseModel` with `extract_features` + `preprocess_obs` | `nn.Module` with `nn.Flatten()` and `obs.float()` | identical for Box observations (FlattenExtractor, no image normalisation) |
| Actor | gSDE and `standard_gauss_init` branches | only the branch DSRL uses (squashed diagonal Gaussian, log-std clamped to [−20, 2]) | train_dsrl.py never sets either |
| Action space | env's `action_space` Box (scale / unscale) | `low`, `high` arrays passed in | our noise space; same formulas |
| ReplayBuffer | n_envs, optimize_memory_usage, noise_actions field, psutil warning | n_envs = 1, no memory-optimised variant, no noise_actions | not used by DSRL-NA; sampling code (`np.random.choice` with uniform p, then `np.random.randint` env index) kept verbatim |
| Hyperparameters | cfg/robomimic/dsrl_can.yaml: lr 3e-4, batch 256, tau 0.005, gamma 0.99, utd 20, noise_critic_grad_steps 10, 3 × 2048 Tanh + LayerNorm, 2 critics, min backup, ent_coef auto (init 1), target entropy 0 | constructor defaults equal to these | — |

## Our old reimplementation (our earlier reimplementation (not shipped)) vs upstream

| Where | Upstream DSRL-NA | Old `dsrl_na.py` | Effect |
|---|---|---|---|
| **Noise bound** | actor tanh output unscaled to the env action space; every shipped NA config has bounds [−1, 1] (robomimic wrapper Box(−1, 1); gym actions in [−1, 1]); `action_magnitude` is read only by DSRL-SAC (`DiffusionPolicyEnvWrapper`, `collect_rollouts` clipping) | w = 1.5 · tanh(·), noise critic input w / 1.5 | substantive: the old policy could play noises up to 1.5 per dimension; tested by EX150 a vs b |
| Target critics | kept, Polyak τ = 0.005 every step | not kept | inert when every transition is terminal |
| Next-state actor sample | drawn every critic step (torch RNG) | not drawn | changes only the RNG stream |
| Replay sampling | global numpy RNG: `np.random.choice(arange, p=uniform)` + `np.random.randint` env index | `np.random.default_rng(seed).integers` | same distribution, different stream |
| Distillation noise | fresh `th.randn` draws, denoised every step | pool of 1,024 draws per input (registered EX136 cost deviation) | approximation kept in the port's `"pool"` mode for EX150 |
| Initialisation | two full SACPolicy builds (actor, critic, critic target, then the noise policy's actor, critic, critic target) in that order, PyTorch default init | actor, critic, noise critic built directly | different initial weights for the same seed |
| Losses, network shapes, optimizers, entropy tuning, update order (entropy → critic → actor on the noise critic → Polyak; then distillation) | — | identical | test T2 (our L1 test record (not shipped)): one step from the same weights, batch and RNG gives bit-identical entropy / critic / actor losses and actor / critic parameters; the noise critics differ by ≤ 4.7e-5 only because the old code scales the noise as w / b and SB3 as 2((w + b) / 2b) − 1 (float32 difference ≤ 2.4e-7, amplified by Adam's first step); with the same scaled input they are bit-identical (T2b) |

## Release r8c: optional variants (default = the rows above, bit for bit)

Every changed line of `dsrl.py` carries "PORT-r8c". With the defaults (`noise_param="box"`, `critic_input_scale="bound"`, `critic_penalty=0.0`, `n_noise_critics=None`) each code path is the one above: the actor loss passes `actions_pi` itself, the distillation and `q_noise` call `scale_action`, the backup calls `unscale_action`, the noise critic is built with the same arguments, and no generator is drawn from. The meaning and the allowed values are in `docs/R8C.md` of the release.

| Where | Authors | r8c option | What it tests |
|---|---|---|---|
| executed noise w | w = unscale(tanh x) in Box(-b, b) | `noise_param` sphere / anchored / sphere_anchored / geodesic (a fixed differentiable map g; box Box(-1, 1); the anchored maps zero the mean layer so the policy starts at the anchor) | the actor's reachable set excludes the prior's shell; start at the global answer |
| noise critic input | scale_action(w) = w / b | `critic_input_scale` raw: w (forced for the non-box maps) | the bound changes the actor's box and the critic's input at once |
| noise critic loss | distillation MSE | `critic_penalty` alpha: + alpha sum_j mean_b [logsumexp_k Q^W_j(s, z_k) - Q^W_j(s, w_b)], z_k ~ N(0, I), k = 1..10 (CQL-style; not the authors') | critic optimism |
| noise critic heads | n_critics (2) | `n_noise_critics` 10; head i regresses Q^A head i mod 2 | pessimism of a larger ensemble |

