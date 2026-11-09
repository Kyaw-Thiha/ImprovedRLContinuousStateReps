# SSP vs tile coding: why a dense continuous encoder can lose, and where it should win

Written 2026-10-06. This is a **hypotheses register with citations**, not a results record. Every measurement
referenced here lives in [`runs/RUN_004.md`](../runs/RUN_004.md); every hypothesis carries an explicit falsifier.
Nothing in this file was measured here except the IHT saturation table in section 6, which is new and is
labelled with its method.

Purpose: our SSP cells read below tile coding on the greedy score under DQN and A2C, and below the legacy A2C
table. The pre-theoretic intuition — a continuous, high-dimensional representation should be at least as
expressive as a coarse discretisation — is not a theorem, and it is not what the literature predicts. This file
records what the literature does predict, which of our measurements already fit, and what would settle it.

---

## 1. Why "continuous" is not "more expressive"

The framing that makes the intuition look necessary is the wrong framing. SSP (random rotations + a scale ladder
through a complex exponential + linear readout) and tile coding (a local piecewise-constant basis + linear readout)
are **the same kind of object**: a fixed feature map with a linear head. Both are kernel machines. Neither is
"more continuous" in any way that grants it an expressivity guarantee at a fixed feature budget.

Four facts replace the intuition:

1. **A fixed random feature map has proven limits.** Yehudai & Shamir (2019) show random features cannot learn
   even a single ReLU neuron under standard Gaussian inputs unless the network is exponentially large. SSP's
   rotations are a random draw; a linear readout on top of it inherits this limit.
2. **Random-feature approximation error is an explicit function of feature count and the kernel's spectrum**,
   not of continuity or of the nominal dimension of the encoding. See Avron et al. (2017) for the ridge-regression
   bounds; the error decays, but with constants that grow with dimension and with the inverse bandwidth.
3. **The kernel / lazy-training regime suffers the curse of dimensionality in `d`.** Ghorbani, Mei, Misiakiewicz &
   Montanari (2019, 2021) show that linearised two-layer networks need exponentially many features in `d` to learn
   generic functions, and characterise when the learned (non-lazy) regime escapes this. "High-dimensional" is a
   cost as often as an asset.
4. **A local basis has a monotone resolution knob; a random basis has a variance knob.** Tile coding approaches a
   piecewise-constant approximation of any continuous `Q` by refining `tiles_per_dim`; the error falls
   predictably. Widening SSP's dimension or changing its random draw gives you a different sample from a
   distribution over feature maps, with no matching guarantee attached to the sample you get.

Consequence for reading our results: for a fixed budget, what matters is **basis-target alignment, per-dimension
resolution, active-feature count and interference** — not continuity. CartPole is a small, bounded, 4-dimensional
control problem where a fine local grid is cheap: that is the regime least favourable to a dense global encoder.

---

## 2. Three established mechanisms for the direction we observe

| # | Mechanism | Statement | Source | Prediction for us |
|---|---|---|---|---|
| M1 | **Interference** | An update to a dense feature set moves the values of *distant* states, degrading previously learned values. Interference correlates with control degradation; larger hidden layers → more interference. RL worsens it over supervised learning because targets are bootstrapped/non-stationary and data is temporally correlated. | Liu, Kumaraswamy, Le & White, *The Utility of Sparse Representations for Control in RL*, AAAI 2019 (arXiv:1811.06626); Liu, Wang, Tao, Javed, White & White, *Measuring and Mitigating Interference in RL*, CoLLAs 2023 ([PMLR v232](https://proceedings.mlr.press/v232/liu23a/liu23a.pdf)); Hernandez-Garcia & Sutton 2019 (arXiv:1912.04002) | SSP's interference measure (change in squared TD error on a held-out probe set before/after an update) is materially higher than tile coding's at matched baseline score. |
| M2 | **Superposition crosstalk / bounded capacity** | A VSA's representation capacity is finite: the dimension required to recover superposed content to a given accuracy grows with the content, and the analysis connects VSAs to sketching and Bloom filters — i.e. crosstalk has a false-positive-like floor. | Clarkson, Ubaru & Yang, *Capacity Analysis of Vector Symbolic Architectures*, arXiv:2301.10352; *High-probability guarantees for linear accessibility in feature superposition*, arXiv:2609.09556 | SSP's 201-1001 dimensions are not 201-1001 independent features; part of the budget is spent absorbing encoder noise. Expect a dimension-vs-performance curve that saturates rather than rising monotonically. |
| M3 | **Metric misspecification and smooth kernels vs cliffs** | SSP's kernel is isotropic over raw state with one `length_scale`; CartPole's dimensions span ~50x in range (theta 0.418 vs omega 20) and theta carries most of the control-relevant information. A single scalar cannot resolve theta finely without over-resolving `x_dot`/`omega`. CartPole's value function also has a cliff at `|theta| > 0.209`; a smooth kernel smooths across it, a local basis can place a hard edge. | RUN_004 section 3.7 finding 2 (the measured asymmetry: `domain_bounds_` is dead, `normalize_state` False in every cell); the `length_scale` floor crowding in 4 of 6 SSP cells | Observation normalisation plus a `length_scale` sweep recovers a visible part of the SSP deficit; a per-dimension length scale recovers more. |

Note on M1's exact claim: Liu et al. use the two-room gridworld as the diagnostic and **tile coding as the explicit
no-interference contrast**, and their interference experiments run on CartPole and Acrobot — the same task family
we are on.

---

## 3. What we have already measured that fits

Pointers only; numbers are in RUN_004.

- **The collapse pattern has the shape of interference** (3.2): every SSP linear seed reaches >= 475 at some
  checkpoint, but only 1 of 20 stays there, against 14 of 20 for `tile_coding/none/linear`. Dense features plus
  bootstrapped targets is the M1 story; our DQN-settings test (3.5) shows it is not a replay/target-network
  artefact, which points the arrow back at the representation.
- **The IHT-saturation measurement** (section 6) shows the tile-coding side is running in a collision-dominated
  regime — so the comparison is bound-limited on *both* sides, and "tile coding is a clean local basis" is itself
  only approximately true in our setup.
- **SSP's advantage is where a distributed code should have one** (3.3, 3.8): SSP wins `action_delay` in all four
  primary pairs by 153-342 points and `random_action` in three of four. A code that spreads information over many
  features degrades smoothly under corruption; a hard-edged grid aliases across bin boundaries. Same mechanism
  (M1/M2), opposite sign.
- **The mechanism is not decision smoothness** (3.8): SSP flips its action *more* often and has a *smaller* Q
  margin; what orders delay retention is `dQ/dx`, the policy's feedback gain.

---

## 4. Hypotheses, with tests and falsifiers

| # | Hypothesis | Test | Falsified if |
|---|---|---|---|
| H1 | Interference explains the SSP deficit. | Implement Liu et al.'s interference measure (difference in squared TD error on the 256 fixed probe states before/after an update) on the saved policies, SSP vs tile coding, matched on baseline score. | Interference is comparable across representations. |
| H2 | The deficit is substantially a domain/metric misspecification, not a representation limit. | Observation normalisation on/off plus a `length_scale` sweep, other parameters held at each winner, 10 fresh seeds. | The sweep is flat for SSP. (Known weakness: holding `lr` fixed biases the design *against* the changed encoder, so a null is inconclusive; a positive is trustworthy.) |
| H3 | Part of the SSP result is a single random-encoder draw. | Re-draw the SSP rotations (and separately the scale ladder) at fixed other parameters; compare the across-draw spread to the across-seed spread. | Across-draw spread is small relative to across-seed spread. |
| H4 | The linear readout is the binding constraint. | SSP -> fixed random ReLU layer (1024/4096, the referenced paper's architecture) -> readout, against the linear readout, same protocol and budget. | The nonlinear arm matches the linear arm. |
| H5 | SSP's real advantage is structure, not resolution. | On a task whose state is relative and unbounded (e.g. `ee_to_cube` / `cube_to_goal` in the Stage-2 arm task), compare SSP against a fixed grid that must be re-bounded or clips. | SSP shows no advantage where the state is relative/unbounded. |

H1 is the cheapest and the most diagnostic, and it is the one a reviewer will ask for: "why would SSP be worse?"
currently has no answer in the write-up other than "we measured it".

---

## 5. Where SSP should be expected to win

The mechanisms in section 2 are not universally anti-SSP; they are anti-SSP *in CartPole-like regimes*. The same
properties predict the opposite sign elsewhere:

| Regime | Why it favours SSP | Where it is testable |
|---|---|---|
| State range unbounded, unknown, or changing across episodes | A grid needs explicit `low`/`high` and clips outside them; fractional binding has no hard edge. | Navigation / reaching where the target distance is not known a priori; Stage-2 arm task. |
| Relative and relational geometry | Binding implements translation, so a learned value function transfers under shifts and can be composed (query "where is the goal relative to me"). A grid must relearn each region. | `ee_to_cube`, `cube_to_goal` (Stage 2, per `02-experimental-design.md`). |
| Compositional / multi-object state (product spaces) | Representation dimension grows linearly with the number of bound pairs, while a grid over the product space explodes. | Multi-object scenes; locomotion observations. |
| Noisy, partial or dropout observations | Distributed codes degrade gracefully; a hard-edged grid aliases across bin boundaries when noise crosses a boundary. | The `obs_noise` family, extended beyond the saturation point; sensor dropout. |
| Medium-to-high-dimensional spatial observation | A grid or tile basis is infeasible (`tiles_per_dim^d`); SSP stays fixed-size. | Stage 3 locomotion (40+ dims), exactly the plan's stated motivation. |
| Continual / transfer settings | One fixed encoder reused across tasks; structured composition rather than a new basis per task. | Continuing-task CartPole, then multi-task. |

And the regime it should **not** be expected to win, which is the one we ran: small, bounded,
low-dimensional, cliff-edged, discrete-action control where a fine local grid is affordable.

Framed this way, the honest claim is not "SSP is worse than tile coding". It is: **SSP's advantage is structural
(unboundedness, relations, composition, noise), and CartPole has none of that structure — while it has exactly
the properties (bounded, tiny `d`, hard termination edge) that let a local grid win.** That is a result about
when a representation pays off, which is more useful than a ranking on one toy task.

---

## 6. Method record for the new measurement (IHT saturation)

Measured 2026-10-06, local, `.venv/bin/python`, using `rl.representations.tilecoding.TileCodingRep` with the
search's bounds (x +-4.8, `x_dot` +-10, theta +-0.418, `omega` +-10), 200,000 states drawn uniformly from that box:

```
iht=  32768 tilings=16 tpd=10 -> dictionary 100% full, 81% of lookups collide
iht=  32768 tilings=32 tpd=16 -> dictionary 100% full, 98% of lookups collide
iht= 131072 tilings=32 tpd=16 -> dictionary 100% full, 92% of lookups collide
iht= 524288 tilings=64 tpd=16 -> dictionary 100% full, 83% of lookups collide
```

`rl/representations/tilecoding.py::IHT.getindex` falls back to `hash(obj) % size` once the dictionary is full, so
every tile-coding config in the search — including the `iht_size` ceiling — is a hash of tile coordinates over the
reachable domain. Limitation: uniform sampling over the bounds box is an over-estimate of what a trained policy
visits, so the collision *rate* during training is lower than these figures; the dictionary still saturates
monotonically over a 1000-episode run. Any re-search that raises `num_tilings` must raise `iht_size` with it.

---

## 7. Related open thread (not part of the theory above)

The legacy A2C reference numbers cannot presently be tied to a surviving run. Recorded here because section 2's
"SSP >= tile coding under A2C" is the claim being contradicted:

- The nine `outputs/hparam_search/best/{rep}_{centering}.yaml` files (which produced handoff 03 section 2.5's
  table, `best_value` 498.351 ~= the reported 498.4) match **no** trial in `optuna_studies_a2c.db` or
  `optuna_studies_old.db` (relative distance 0.44-2.05; the ssp/none yaml's `lr` 0.4775 vs the nearest trial's
  0.0019). The older `hparam_search_old` yamls *do* match their DB exactly, so the matching method works.
- They match no run artefact either (all 1,784 per-trial CSVs and sidecars under `outputs/hparam_search`, and the
  paper-era records under `cartpoleData/` in `origin/main`).
- The legacy A2C trial (`experiments/trial_cartpole_nengo.py`) has **no seed handling at all**
  (`grep -c seed` = 0), so its "10 re-evaluation seeds" were unseeded re-runs, and the reported value is a
  `max` over the top-10 candidates' 10-run means.
- The only surviving record of those exact numbers is DQN study trial 0 (`dqn_ssp_none_hparam_search`), i.e. the
  DQN warm-start copy of whatever the A2C yaml then held.

Consequence: any statement of the form "our SSP A2C result is worse than previously reported" is currently
unfalsifiable, because the reported number's configuration and protocol are neither in the repo nor reproducible.
This is a reporting gap, not evidence of a port defect.

---

## 8. References

| Work | Citation | Used for |
|---|---|---|
| On the Power and Limitations of Random Features for Understanding Neural Networks | Yehudai & Shamir — [arXiv:1904.00687](https://arxiv.org/abs/1904.00687) (NeurIPS 2019) | The hard limit on fixed random features with a linear readout; SSP is a random-feature map. |
| Random Fourier Features for Kernel Ridge Regression: Approximation Bounds and Statistical Guarantees | Avron, Kapralov, Musco, Musco, Velingker, Zandieh — [PMLR v70](https://proceedings.mlr.press/v70/avron17a/avron17a.pdf) (ICML 2017) | Feature-count/error bounds for random Fourier features. |
| Linearized two-layers neural networks in high dimension; Limitations of Lazy Training of Two-layers Neural Networks; When Do Neural Networks Outperform Kernel Methods? | Ghorbani, Mei, Misiakiewicz & Montanari — [arXiv:1904.12191](https://arxiv.org/abs/1904.12191), [arXiv:1906.08899](https://arxiv.org/abs/1906.08899), [arXiv:2006.13409](https://arxiv.org/abs/2006.13409) | The curse of dimensionality in the kernel/lazy regime; when the learned regime escapes it. |
| The Utility of Sparse Representations for Control in Reinforcement Learning | Liu, Kumaraswamy, Le & White — [arXiv:1811.06626](https://arxiv.org/abs/1811.06626) (AAAI 2019) | Sparsity/locality improves control; dense representations interfere. |
| Measuring and Mitigating Interference in Reinforcement Learning | Liu, Wang, Tao, Javed, White & White — [PMLR v232](https://proceedings.mlr.press/v232/liu23a/liu23a.pdf) (CoLLAs 2023) | The interference measure to run as H1; CartPole experiments; tile coding as the no-interference contrast; larger networks interfere more. |
| Learning Sparse Representations Incrementally in Deep Reinforcement Learning | Hernandez-Garcia & Sutton — [arXiv:1912.04002](https://arxiv.org/abs/1912.04002) | Same conclusion from the sparse-coding side. |
| Capacity Analysis of Vector Symbolic Architectures | Clarkson, Ubaru & Yang — [arXiv:2301.10352](https://arxiv.org/abs/2301.10352) (2023) | Bounded VSA capacity; crosstalk as a sketching/Bloom-filter error floor. |
| High-probability guarantees for linear accessibility in feature superposition | [arXiv:2609.09556](https://arxiv.org/abs/2609.09556) | Cross-feature interference constrains what a linear readout can access. |
| Improving RL with Biologically Motivated Continuous State Representations | Bartlett, Simone, Dumont, Furlong, Eliasmith, Orchard, Stewart — 21st ICCM (2023) proceedings | The referenced paper. Its CartPole comparison set was five tabular discretisations and a deep A2C MLP; **tile coding was never tested**, so no published claim of SSP > tile coding exists to contradict. |
| Simulating and Predicting Dynamical Systems With Spatial Semantic Pointers | Komer, Dumont, Eliasmith et al. — Neural Computation 33(8), 2021 | SSP as a continuous-space encoder; its demonstrated strengths are dynamical/spatial, not CartPole-like control. |

**Citation fixes needed in `06-related-work.md`:** it dates the referenced paper as "ICCM 2022"; RUN_004 section 3.7
identifies the PDF as the **21st ICCM (2023)** proceedings, and the author list in the two files should be
reconciled. Both should be corrected together, in one place.
