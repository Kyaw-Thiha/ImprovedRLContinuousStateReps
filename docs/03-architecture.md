# Codebase Architecture

## Three separate codebases — not one merged repo
1. **Legacy repo** (`ImprovedRLContinuousStateReps`) — being trimmed, not extended. Nengo-based agent code (`network/rlnet`, `trial_cartpole.py`) is being retired as it's superseded; old A2C result CSVs are kept as a citable baseline.
2. **Stage-0 repo** — new, lightweight, plain PyTorch + NumPy, no Nengo, no GPU requirement. Hosts DQN-linear/deep and the re-ported A2C on classic CartPole.
3. **mjlab task-plugin package** — new, separate installable Python package (Python 3.10–3.13, `uv`, CUDA-build Torch ≥2.7) containing custom task configs, observation terms, and the centering patch. Registers itself with mjlab via the `mjlab.tasks` entry point (`src/mjlab/__init__.py`'s `_import_registered_packages()`); mjlab's own source is never forked or modified directly.

Kept separate because: incompatible dependency stacks (legacy repo is pinned to Python 3.9 / NumPy 1.22 / Nengo 3.2; mjlab needs Python ≥3.10 / modern Torch / CUDA), and because DQN-on-CartPole has no natural relationship to a MuJoCo/Warp robotics framework.

## `sspspace` reuse strategy
`sspspace` (github.com/ctn-waterloo/sspspace) is the authoritative, Nengo-free, NumPy implementation of the SSP encoding math, maintained by the original HexSSP paper's authors (same lab). Not published to PyPI proper — installed via `pip install git+...`, pinned to a specific commit for reproducibility (it's "Pre-Alpha" and its API has changed since the version vendored in the legacy repo).

- **Stage-0 repo**: real runtime dependency. Called directly, every step, on CPU — this is fine at this task's scale.
- **mjlab package**: NOT a runtime dependency. Importing NumPy code into a GPU-batched per-step training loop would force a CPU round-trip every step, defeating the purpose of moving to mjlab. Instead:
  - The `phase_matrix` (the one-time setup artifact) is built once via real `sspspace`, then handed to a small Torch module (`HexSSPEncoderTorch`) as a `register_buffer`.
  - `HexSSPEncoderTorch.encode()` is a ~15-line Torch translation of the same three ops (matmul → complex exponential → inverse FFT), called every step, on GPU, batched over `num_envs`.
  - A parity test (encode identical inputs with `sspspace` and with `HexSSPEncoderTorch`, assert `np.allclose`) is the only place the two implementations are compared — run once when the Torch encoder is written or changed, not during training.
  - Known API gotcha: `sspspace`'s `encode()` returns a wrapped `SSP` object, not a raw array — the underlying array is `.v`, and `SSP` does not support `.reshape()` or other array-protocol delegation.

## mjlab integration points (verified against source, not assumed)
- Observation terms are plain functions: `ObservationTermCfg(func=my_func, params={...})` where `my_func(env, ...) -> torch.Tensor`. This is where `HexSSPEncoderTorch.encode()` gets called.
- Training uses RSL-RL (PPO + Student-Teacher Distillation only, on the actively maintained branch — no SAC).
- Value-based centering for PPO hooks into `PPO.compute_returns()` in `rsl_rl/algorithms/ppo.py` — the GAE loop already computes `delta = rewards[step] + next_is_not_terminal*gamma*next_values - values[step]`, which *is* the TD-error the centering equations need. Subclass `PPO`, override `compute_returns`, subtract a running `R̄` from `rewards[step]` before computing `delta`, update `R̄` from `delta.mean()`. No other RSL-RL internals need touching.
- RSL-RL already bootstraps on timeouts (`rewards += γ·V(s')·time_outs`) rather than zeroing return at episode length — partial "continuing" behavior is already present; the remaining conversion work is turning genuine failure terminations into reset-with-penalty (see `05-tasks-and-benchmarks.md`).
