# Handoff 05: A2C port (Nengo-free) and its eval_v3 interface

Written 2026-10-06. Scope of this piece of work: port the legacy Nengo A2C to plain NumPy, validate it against the
old results, and make it plug into the existing eval_v3 pipeline. NO evaluation has been run (no 20-seed policy
pass, no robustness, no deployment run, nothing on R2, no pod launched, nothing committed). Whoever runs the
evaluation should not need to change the port.

## 1. What the A2C actually is (read before interpreting any result)

It is NOT a textbook policy-gradient A2C. The port keeps the legacy rule exactly (`rl/networks/acLinear.py`):

- One weight matrix W (n_actions + 1, dim): row 0 = state value V, other rows = linear action PREFERENCES. Behaviour
  policy = epsilon-greedy over the preferences (`force_rho_one=True`, so the softmax target policy never matters).
  The greedy policy = `argmax(preferences)`.
- Every update is divided by `||phi(s)||^2` (normalised step), so lr 0.4-0.8 is a fraction of an NLMS step, not a
  plain SGD rate. Value row += `lr * td * phi * scale`; the TAKEN action's row += `act_dis * td * phi * scale`
  (same TD error as the critic, no pi(1-pi) term).
- Stale bootstrap: V(s) used in the TD target was computed one step earlier, and the preferences used to pick the
  next action come from the weights BEFORE this step's update.
- Termination = one extra step on the same state with reward 0 (`n_done=1`), not a zero bootstrap.
- Dynamic epsilon: `-0.001` after an episode when the mean of the last 10 episodes beats the previous 10 by one std.
- Centering: `simple`: td uses `r - R`, `R += beta (r - R_old)`. `value`: td uses `r - R`, `R += eta * lr * td`. The OLD
  R is used in the TD error in both modes. W starts at zero (no init randomness; the seed only affects env/actions).

## 2. Files

New (Nengo-free; `tests/test_a2c_port.py` asserts `nengo`/`nengo_spa` are never imported):

- `rl/networks/acLinear.py` - `LinearActorCritic` (rule, three centering modes, dense or sparse features).
- `rl/representations/ssp_legacy.py` - `LegacySSPRep`: the OLD SSP encoder (n_scales 8, scales linspace(0.5, 2),
  principal-log phase recovery, scipy-1.7.3 rotations). `legacy_special_ortho_rvs` reimplements scipy 1.7.3's SO(d)
  sampler, see section 4.
- `experiments/trial_cartpole_a2c.py` - `train_a2c(...)`, `load_winner_yaml`, `greedy_episodes`, `probe_stats`.
- `rl/policy_io_a2c.py` - `save_a2c_policy`, `LinearACPolicy` (same surface as the DQN `Policy`).
- `scripts/run_policy_pass_a2c.py`, `scripts/policy_pass_worker_a2c.py` - the A2C policy pass.
- `scripts/gate_a2c_port.py` - the port validation gate (old metric).
- `conf/a2c_winners/<rep>_<centering>.yaml` - tracked copies of the 9 legacy winners (the originals in
  `outputs/hparam_search/best/` are git-ignored and not in the pod code tarball). Read only; never retuned.
- `tests/test_a2c_port.py`, `tests/test_a2c_eval_ready.py`, `tests/data/legacy_ssp_golden.npz`.

Legacy, renamed with `git mv` (staged, not committed; logic unchanged, only names/imports/one header comment):
`rl/networks/acBasic_nengo.py`, `rl/rules/td0_nengo.py`, `rl/rules/td0centering_nengo.py`,
`experiments/trial_cartpole_nengo.py`, `scripts/run_hparam_search_a2c_nengo.py`. The legacy SSP branch of
`trial_cartpole_nengo.py` is NOT runnable any more (the Stage-0 rewrite replaced `SSPRep`/`sspspace`); the old result
CSVs remain the citable baseline.

Small additive edits to existing eval code (DQN behaviour unchanged, full suite passes):
`rl/policy_io.py` (`load_policy` dispatches on `algo == "a2c_linear"`), `scripts/eval_v3_layout.py` (A2C cells),
`scripts/run_policy_pass.py` (delegates `*/a2c` cells to the A2C runner), `scripts/analyze_robustness.py` (pairs for
the `a2c` arch, `--a2c` flag), `scripts/launch_cell_pod.py` (`parse_cell`/`study_name` accept `a2c`).

## 3. Eval interface

- Cells are `rep/centering/a2c` (the third field is the algorithm; there is no linear/deep split). Study =
  `a2c_<rep>_<centering>_port`. `layout.A2C_CELLS` has the 9; `layout.ALL_CELLS` is still the 18 DQN cells.
  Keys and seeds are otherwise identical to the DQN pass (policies `seed_3000..3019.pt`, same files).
- Protocol per run (mirrors the DQN trial): env seeded from the run seed; 15 greedy checkpoints x 10 episodes every
  `1000 // 15 = 66` episodes (first checkpoint env seeded with seed + 1e6); score = mean of the last 5; 100-episode
  greedy eval of the final policy; verify replay on a fresh env seeded seed + 2e6; policy saved LAST and reloaded
  from disk, replay must match exactly (`verify_max_abs_diff == 0.0`).
- Probe diagnostics at every checkpoint (`probe_history`, 256 fixed probe states): `mean_value` (V), `mean_max_q` (mean
  max preference), `mean_action_gap`, `avg_reward` (centering R-bar). The key names `mean_max_q`/`mean_action_gap`
  are the DQN ones on purpose so `summarize_runs` works unchanged; here they are preferences, not Q-values.
- `LinearACPolicy` has the DQN `Policy` surface, so `run_robustness.py`, `perturbations.py`, `run_deployment.py`
  need no A2C branch. Deployment "network" = the ACTOR rows only (`net.weight`, shape (n_actions, dim)); the critic row
  is only needed for learning, is stored in the file as `value_row` for diagnostics and is NOT counted in size.
  Quantization / op counts / latency therefore describe the deployed actor + encoder.
- Commands (same shape as the DQN ones):

```bash
.venv/bin/python scripts/run_policy_pass.py --cell ssp/none/a2c --workers 8 [--n-seeds 10] [--no-r2 --local-root DIR]
.venv/bin/python scripts/run_robustness.py --cell ssp/none/a2c --workers 8 [--no-r2 --local-root DIR]
.venv/bin/python scripts/run_deployment.py --cell ssp/none/a2c --sections size,quant --quant-episodes 20 [...]
.venv/bin/python scripts/analyze_robustness.py --a2c --local-root DIR --out-dir DIR/_analysis
.venv/bin/python scripts/launch_cell_pod.py --job policy_pass --cells ssp/none/a2c --dry-run   # dry-run verified
```

Seeds are independent jobs and the pass is resumable, so `--n-seeds 10` can be extended to 20 later without
redoing anything (same `--seed-offset`). The tiny offline chain (`--tiny`) was run end to end for ssp, tile and
discrete cells (pass -> robustness -> deployment size/quant/latency -> analysis), and is covered by
`test_tiny_policy_pass_end_to_end`. It was NOT run on a real pod (image dependencies: numpy, gymnasium, torch,
pyyaml, the pinned `sspspace` for the unchanged DQN `SSPRep` import).

## 4. Validation gate (old online metric = mean return of the last 100 training episodes, epsilon included)

Old winner params, 10 fresh seeds (1000-1009), pass if the old value is inside the bootstrap 95% CI of the port's mean
or within 2 SE (fixed before running). `scripts/gate_a2c_port.py`, about 3 minutes for 4 cells on 16 cores:

- ssp/none 490.6 (old 498.4), tile_coding/none 492.4 (495.0), discrete/none 112.4 (118.5), tile_coding/value 496.4
  (495.1): all PASS. The other five cells were not run for the gate; `simple`/`value` centering is pinned by exact
  tests against the real legacy rule code.
- First SSP run FAILED (262 vs 498). Cause: `scipy.stats.special_ortho_group.rvs(4, size=n, random_state=1)` draws
  DIFFERENT rotations in scipy 1.18 than in the legacy pin scipy 1.7.3, which changes the SSP kernel. Fixed by
  reimplementing the 1.7.3 algorithm; golden values were generated by the real vendored encoder under Python 3.9 /
  scipy 1.7.3 (a throwaway `uv` env, not in the repo). Independent check: replaying the recorded states/rewards of a
  real old SSP run (80 episodes) through the port reproduces the recorded value series to 2e-6.
- The DQN `SSPRep` uses upstream `sspspace` (n_scales 5, current scipy rotations) and is NOT the A2C encoder. A2C SSP
  dim = `80 * n_rotates + 1`; DQN SSP dim = `50 * n_rotates + 1`. Keep this in mind in any DQN-vs-A2C comparison.

## 5. Caveats and open items

- Checkpoint evals allocate tile-coding IHT entries (as in the DQN trial), which only permutes W's column order; tests
  confirm returns are identical with and without the hooks (weights equal up to that permutation).
- Winner configs come from the legacy 500-trial searches (old online metric, May 2026). Fresh-seed greedy numbers are
  expected to be a bit lower than the old `best_value` (winner's curse). No retune was needed or done.
- The old A2C results used the ONLINE metric; the greedy metric here is new for A2C, so nothing from RUN_003/004 can be
  compared to the old A2C numbers directly. Compare A2C greedy vs DQN greedy.
- Decision-log entry 17 (centering hurts on episodic CartPole) is still DQN-specific; its A2C caveat should be written
  from the new measured results (the `simple` cells matter for that), not before.
- Nothing about cost/time on pods is measured. Locally the old-metric runs took roughly 30-100 s each (tile ~90 s,
  SSP ~40 s at 1000 episodes, discrete ~10 s), 16 runs in parallel on 16 cores.
- `watch_cells.py` still does not understand eval pods (see handoff 04); monitor A2C eval pods by hand and never touch
  `cerb-*` pods.
