# Survey of the installed POPGym (reference venv `/tmp/popgym_ref`)

Installed: popgym 1.0.7, ray 2.59.0, torch 2.14.1+cpu, gymnasium 1.2.2, Python 3.14.6 (rebuilt from `ref_ppo/requirements-frozen.txt`,
`uv pip install --no-deps`, exactly as `ref_ppo/README.md`). Source read: `popgym/baselines/{ppo.py,ray_models/*,models/*}`.

## Memory models shipped (`popgym.baselines.ray_models`, all subclass `base_model.BaseModel`, a RLlib `TorchModelV2`)

`popgym/baselines/ppo.py` groups them as:

- recurrent: `LSTM`, `GRU`, `Elman`, `LMU`, `IndRNN`, `DiffNC`
- attention / fast-weight: `LinearAttention`, `FastWeightProgrammer`, `DeepFastWeightProgrammer`, `DeepLinearAttention`
- convolution: `S4D`
- no memory / fixed window: `BasicMLP`, `MLP`, `Framestack`, `Frameconv`

Carried state (floats per env, at POPGym's `hidden_size = 256`, preprocessor width 128): GRU 256; LSTM 512; Elman 256; LMU 512;
IndRNN 512; `FastWeightProgrammer` h = round(sqrt(256)) = 16, state 16x16 = **256**; `DeepFastWeightProgrammer` 4 layers x 8x8 = 256;
`LinearAttention` h x h + h with h = round(sqrt(256) - 1) = 15 (240); MLP none. The SSP model in `ssp_model.py` carries `2*dim = 256` floats
(complex bundle `M` of 128 dims), the same as the GRU and the FWP.

## Is POPGym's fast-weight model a delta-rule model? No.

`popgym/baselines/models/fwp.py`, class `FWPBlock`, docstring (verbatim):

> Note that this is a "simplified" version of FWP without the delta update rule and DPFP. We use ReLU instead of DPFP.

The write is plain outer-product accumulation. The update lines:

```python
K = self.key(x).relu();  Q = self.query(x).relu();  V = self.value(x)
if self.sum_normalization:                       # K, Q divided by their sums (the paper's sum-norm)
    K = K / (1e-5 + K.sum(dim=-1, keepdim=True)); Q = Q / (1e-5 + Q.sum(dim=-1, keepdim=True))
kv = torch.einsum("bti, btj -> btij", V, K)      # v k^T, NO (v - M k) correction, no write strength, no gate
state = self.aggregator(kv.flatten(-2), state.flatten(-2)).reshape(shape)
y = torch.einsum("btij, btj -> bti", state, Q)
```

with `SumAggregation.forward` (`popgym/baselines/models/aggregations.py`): `return x.cumsum(dim=1).clamp(-1e20, 1e20) + memory`,
i.e. `M_t = M_{t-1} + v_t k_t^T` (no decay at all, unlike this repo's `FWPArm`, which has a learned scalar `lam`).
Keys/queries are separate (untied) linear maps. So **POPGym ships no delta-rule fast-weight model**: `FastWeightProgrammer` and
`DeepFastWeightProgrammer` are plain linear attention with ReLU features and sum normalisation; `LinearAttention` is the
elu+1 feature-map linear transformer with an explicit denominator `Z`. The RL "`fwp_delta`" arm of handoff 11 therefore
**does not exist in POPGym**; this track compares against POPGym's plain-`FastWeightProgrammer` (called `fwp` below). A delta-rule
arm would have to be written (a new `BaseModel` subclass), which is outside this pilot.

## Environments shipped (`popgym.envs.ALL`: 42 = 14 tasks x {Easy, Medium, Hard})

Autoencode, RepeatPrevious, RepeatFirst, CountRecall, PositionOnlyCartPole, PositionOnlyPendulum, VelocityOnlyCartPole,
NoisyPositionOnlyCartPole, NoisyPositionOnlyPendulum, MultiarmedBandit, HigherLower, Battleship, Concentration, MineSweeper.
The handoff's recalled list (RepeatFirst, Autoencode, Concentration, CountRecall) is correct; ids look like
`popgym-RepeatPreviousEasy-v0`.

`RepeatPrevious` (the pilot env): the observation is the card **suit** (`Discrete(4)`); the agent must output the suit seen `k`
steps ago (`Easy`: k = 4, 1 deck, 51 steps/episode; `Medium`: k = 32, 2 decks, 103 steps; `Hard`: k = 64, 3 decks, 155 steps).
Reward `+-1/(deck - k)` per step once `k` cards have been dealt. POPGym's wrappers `PreviousAction` and `Antialias` make the policy
observation `Tuple(Discrete(4) suit, Discrete(4) previous action, Discrete(2) is_t0)`, one-hot flattened to 10 dims (so the discrete
token is `argmax(obs_flat[:, :4])`). Random / optimal return, measured with `rl/env_returns.py` (`results/rl_env_returns.json`):

- Easy: random -0.512, optimal +1.000
- Medium: random -0.508, optimal +1.000
- Hard: random -0.502, optimal +1.000

## What POPGym's PPO path is, and the API drift (all handled outside the installed package)

`ppo.py` uses `ray.tune.run("PPO", ...)` with the pre-2.x config names; it does not run on ray 2.59. `ref_ppo/run_ref.py` (and `rl/run_pilot.py`
after it) call `PPOConfig().update_from_dict(<ppo.py's own config dict>)` with the old API stack
(`enable_rl_module_and_learner=False`, `enable_env_runner_and_connector_v2=False`) because the shipped models are `ModelV2`.
PPO hyperparameters are therefore RLlib's own PPO defaults (lr 5e-5, `num_epochs` (= old `num_sgd_iter`) 30, clip 0.3, `vf_loss_coeff` 1.0, `entropy_coeff` 0, `kl_coeff` 0.2, `vf_clip_param` 10,
`lambda_` 1.0) plus POPGym's `gamma 0.99`, `train_batch_size = bptt_cutoff x workers x envs/worker`, `sgd_minibatch_size = 8 x bptt_cutoff`,
`batch_mode = complete_episodes`, `max_seq_len = bptt_cutoff = 1024`. POPGym's `ppo.py` has no evaluation loop of its own; its reported number is
the training `episode_reward_mean` (the paper's MMER = max over training of the mean episode reward), which is what is recorded here.
