# Handoff: Stage 0 Repo Rebuild

Target: new lightweight repo (or a clearly separated module within the legacy repo, e.g. `dqnExperiments/`), plain PyTorch + NumPy, no Nengo, no GPU requirement.

## Tasks

1. **Remove Nengo from the loop entirely.** Confirm no `import nengo`, `nengo.Network()`, or `nengo.Simulator()` anywhere in the new code path. The state representation must be a plain tensor at every step, not an `nengo.Ensemble` population.

2. **Migrate to `sspspace` as a pinned dependency.**
   - Add to requirements: `sspspace @ git+https://github.com/ctn-waterloo/sspspace@<pin-to-a-specific-commit-sha>`
   - Update constructor calls: old vendored API used `HexagonalSSPSpace(N, domain_bounds=..., scale_min=..., scale_max=..., n_scales=..., n_rotates=..., length_scale=...)` (class-based). Current upstream API: `HexagonalSSPSpace(domain_dim, n_rotates=5, n_scales=5, kernel="hypergeom", length_scale=1, even=False)` (factory function, returns `SSPEncoder`). No more `scale_min`/`scale_max` (replaced by `kernel` choice) or `domain_bounds` on the encoder.
   - Fix the unwrap: `encoder.encode(x)` returns a wrapped `SSP` object, not a raw array. Use `.v` for the underlying NumPy array. `SSP` does **not** support `.reshape()` or other array-protocol delegation — code like `ssppos.reshape(-1)` must become `ssppos.v.reshape(-1)`.

3. **Implement DQN-linear and DQN-deep** in plain PyTorch:
   - Linear: `Q(s,a) = W · φ(s)`, single weight matrix.
   - Deep: 2 hidden layers + output.
   - Both variants: standard epsilon-greedy + replay buffer + target network.
   - Centering (mandatory): implement value-based centering per the DQN equations in `docs/04-algorithms-and-centering.md`.

4. **Re-port A2C** through the same plain-PyTorch harness (currently the only existing A2C numbers come from the Nengo path) — needed for a fair, apples-to-apples comparison against the new DQN numbers.

5. **9-config runner**: representation ∈ {none, tile-coding, SSP} × centering ∈ {none, simple, value-based}, applied per algorithm.

6. **Hyperparameter tuning**: implement the 3-full-search + 6-coarse-sweep schedule from `docs/02-experimental-design.md` — don't run 9 independent full searches.

## Expected compute
Minutes per run on CPU, no GPU needed — CartPole's state/action space (4-dim, 2 actions) is small enough that GPU kernel-launch overhead can make it *slower*, not faster. If a run isn't finishing in minutes, suspect a leftover Nengo dependency first.
