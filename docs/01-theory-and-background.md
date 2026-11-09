# Theory and Background

## Reward centering
Source: Naik, Wan, Tomar, Sutton — "Reward Centering" (arXiv:2405.09999, RLC 2024).

- In discounted RL, the value function decomposes (Laurent series) as:
  `v_π^γ(s) = r(π)/(1-γ) + ṽ_π(s) + e_π^γ(s)`
  where `r(π)/(1-γ)` is a large, state-independent offset that grows as γ→1 and carries no information for action selection.
- **Simple centering**: maintain a running average reward estimate `R̄` and subtract it from observed rewards before bootstrapping.
  `R̄_{t+1} = R̄_t + β_t(R_{t+1} - R̄_t)`
  Valid on-policy; converges to `r(behavior policy)`, not `r(target policy)`, off-policy.
- **Value-based centering**: update `R̄` using the TD-error itself, not the raw reward — gives correct off-policy behavior.
  `δ_t = (R_{t+1} - R̄_t) + γV(s_{t+1}) - V(s_t)`
  `R̄_{t+1} = R̄_t + η α ρ_t δ_t`
  This is the form proven to converge for tabular Q-learning in the source paper, and the form this project treats as mandatory.
- **Critical constraint**: reward centering is only valid for **continuing** (non-episodic) problems. Naively applying it to an episodic task can change the optimal policy (e.g. a gridworld with -1-per-step reward becomes trivial if all rewards are centered to zero). Any task used here that is natively episodic needs conversion (see `05-tasks-and-benchmarks.md`).

## Bellman Error Centering (theoretical refinement)
Source: Chen, Gong, Yang, Wang — "Bellman Error Centering" (arXiv:2502.03104, 2025). Not from Sutton's group, but a direct follow-up building on the above.

- Shows that "value-based reward centering" from the original paper is, precisely, **centering the Bellman/TD error**, not the reward — a more accurate characterization than the original paper's own framing.
- Derives a "centered TD fixpoint" and proves convergence for **linear function approximation**, on-policy (CTD) and off-policy (CTDC) — stronger than the original paper's tabular-only proof.
- Explicitly flags actor-critic extensions as open/future work — directly relevant to this project's A2C/PPO implementations. Use this framing (not the naive original-paper equations) when implementing centering for A2C/PPO.

## HexSSP (Hexagonal Spatial Semantic Pointers)
Source: Bartlett, Simone, Dumont, Furlong, Eliasmith, Orchard, Stewart — ICCM 2022 paper; encoding math maintained in `sspspace` (ctn-waterloo).

- Encodes a continuous state vector `x` as a high-dimensional vector via a Fourier-domain construction:
  `φ(x) = F⁻¹{e^{iΘx}}`
  where `Θ` (the phase/encoding matrix) is built once, structured to mimic grid-cell firing (hexagonal tiling, multiple scales and rotations).
- This is **pure linear algebra** (matmul + complex exponential + inverse FFT) — it does not require any neural population simulation. The original paper's implementation used Nengo to *also* build a population of neurons whose firing mimics grid cells, but that's a separate, optional interpretive layer, not required to use the encoding for RL.
- Claimed properties (from original paper, being re-validated): matches/beats tabular binning on continuous control; representation size stays compact as state dimensionality grows (unlike binning); pairs well with shallow function approximators.
- Not yet validated at dimensionalities beyond ~3-4 (original tasks: RatBox nav [3-dim], CartPole [4-dim]). Whether the dimensionality advantage holds for higher-dimensional robot state (e.g. legged locomotion joint state) is an open question this phase is testing.

## Why combine these two ideas
Both address different failure modes of standard discounted RL:
- Reward centering fixes a *value-magnitude* problem (large, uninformative offset in continuing tasks).
- HexSSP fixes a *state-representation* problem (discretization doesn't scale, loses information near bin boundaries).

They are largely orthogonal and testable independently or jointly — hence the 3×3 grid rather than a single combined method.
