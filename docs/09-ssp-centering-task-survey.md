# SSP x centering: task survey, duplication map, and recommended testbeds

Written 2026-10-06. Surveyed via alphaXiv (`discover_papers`) and web search, plus full-text reads of the primary
sources. Purpose: choose tasks where an **SSP + centering** result would be informative *and* not already
published. Companion to [`08-ssp-vs-tile-coding-theory-and-hypotheses.md`](08-ssp-vs-tile-coding-theory-and-hypotheses.md).

Confidence labels: **read in full** / **abstract only** / **unchecked**.

---

## 1. What is already taken (do not redo any of this)

| Work | What it already covers | Status |
|---|---|---|
| Naik, Wan, Tomar & Sutton, *Reward Centering*, RLC 2024 ([arXiv:2405.09999](https://arxiv.org/abs/2405.09999)) | Theory of centering; **tabular and linear FA with a tile-coding representation**; deep RL limited to DQN on Pendulum and Catch; PPO on six MuJoCo tasks; the reward-offset invariance result; benefit grows as gamma -> 1 | Read in full (v1 HTML) |
| Wan, Korenkevych & Zhu, *An Empirical Study of Deep RL in Continuing Tasks*, Meta, Jan 2025 ([arXiv:2501.06937](https://arxiv.org/abs/2501.06937)) | The continuing-task testbed suite: MuJoCo (Swimmer, HumanoidStandup, Reacher, Pusher, Ant/SpecialAnt without resets; HalfCheetah, Ant, Hopper, Humanoid, Walker2d with predefined resets; agent-controlled resets) + six Atari games; DDPG/TD3/SAC/PPO/DQN; 10 runs x 3M steps; TD-based centering vs moving-average vs reference-state centering. Code: `github.com/facebookresearch/DeepRL-continuing-tasks` | Read in full (v1 HTML) |
| De Asis, Elsayed & He, *Extending Differential TD Methods for Episodic Problems*, May 2026 ([arXiv:2605.04368](https://arxiv.org/abs/2605.04368)) | Proves a termination-corrected differential TD that preserves policy ordering, i.e. **legitimately extends reward centering to episodic problems**; shows centering improves sample efficiency on episodic benchmarks across base algorithms | Abstract only |
| Chen, Gong, Yang & Wang, *Bellman Error Centering*, 2025 ([arXiv:2502.03104](https://arxiv.org/abs/2502.03104)) | Centered fixpoint for tabular + linear FA; on-policy CTD and off-policy CTDC algorithms | Already cited by this project (`06-related-work.md`) |
| Bartlett et al., ICCM 2023 (the referenced paper) | HexSSP on RatBox (success) and CartPole; comparison set = five tabular discretisations + deep A2C MLP; **tile coding never tested**; online metric; no centering | Read in part (via RUN_004 3.7) |
| Komer 2020 thesis; Komer & Eliasmith 2019; Komer et al., *Simulating and Predicting Dynamical Systems With SSPs*, Neural Computation 33(8) 2021 | SSP as a continuous-space encoder; fractional binding; the dynamical-systems demonstrations | Abstract/title level |
| Masukawa et al., *Vector Symbolic Policy Gradient*, Aug 2026 ([arXiv:2608.18404](https://arxiv.org/abs/2608.18404)) | VSA on the **action** side: each action is a hypervector, policy gradient = advantage-weighted bundling; bit-flip robustness guarantee decaying exponentially in dimension | Abstract only |
| *Geometric Priors for Generalizable World Models via VSA*, Feb 2026 ([arXiv:2602.21467](https://arxiv.org/abs/2602.21467)) | VSA for world models / dynamics generalisation | Abstract only |
| *Geodesic Flow Matching for Denoising High-Dimensional Structured Representations*, May 2026 ([arXiv:2606.00248](https://arxiv.org/abs/2606.00248)) | Denoising SSP/VSA representations for continuous domains | Abstract only |
| *Optimizing Spatial State Representation in RL* (MDPI Drones 10(6):442) | Possibly relevant spatial-state RL work | **Unchecked** — publisher returned HTTP 403 |

Two conclusions from the table:

1. **Centering x a tile-coding/local basis is already published**, at toy scale (Catch, Pendulum), by Naik et al.
   themselves. So "centering + tile coding" is not a novelty claim we can make.
2. **Centering at scale on continuing MuJoCo/Atari is already published** by Wan et al., with five algorithms and
   three reset scenarios — *with MLP encoders and no representation axis*. That paper occupies the
   "does centering generalise in deep RL" slot; we must not re-run it.

---

## 2. The empty intersection

Nobody has combined: **(a) a structured high-dimensional continuous encoder (SSP/VSA), (b) a centering /
average-reward criterion, (c) a continuing task whose state actually has the structure the encoder assumes, and
(d) a matched local-basis baseline.** The two literatures are complementary and have each found the effect size
the other side is missing:

- **Wan et al. measured a state-topology effect of 347%-1608%.** In continuing Swimmer, DDPG/TD3/SAC gained
  1608%/347%/1027% simply by converting the observed joint angles to `x mod 2pi - pi` — "this conversion does not
  change the environment dynamics, and the new state space is equivalent to the original one". They frame it as
  constraining the state space for exploration. It is equally a statement that **a grid-style encoder has a wrap
  discontinuity that costs multiples of a thousand percent**, and it is currently patched by a manual hack, not by
  a representation.
- **Naik et al. and De Asis et al. established where centering matters**: as gamma -> 1, under a common reward
  offset, and (via De Asis) in episodic tasks with termination. All of these are properties of the *criterion and
  the task structure*, and both groups tested them with tabular, tile-coded, or MLP encoders.

SSP binds over the circle group, so a periodic dimension has no seam; it is unbounded, so unknown ranges need no
`low`/`high`; and it is a kernel over distance, so relative geometry transfers. Those are exactly the three
properties Wan et al. had to hack around. That is the novel, non-duplicative claim.

---

## 3. Recommended tasks, ranked

| # | Task | State structure that loads SSP | Why centering is live | Testbed | Cost | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Continuing Reacher with target resampled every 50 steps** (MuJoCo, no resets) | Target position is **relative** to the arm; the optimal policy is a function of relative geometry only, so translation invariance is directly testable (shift the target distribution at eval) | Continuing; gamma can be pushed toward 1, where centering's benefit is largest; dense smooth reward with a common offset | Exists in Wan et al.'s released suite | Moderate (state-based MuJoCo; 1M steps x 10 seeds is enough for a reward-rate comparison) | **Tier 1** |
| 2 | **Continuing Swimmer with wrapped-angle state** | Periodic topology; SSP has no wrap seam where a grid does. The published baseline effect (347-1608% from wrapping) makes this the highest signal-to-noise test available | Continuing, no resets; rewards are dense with an offset | Exists in the same suite | Moderate | **Tier 1** |
| 3 | **Continuing CartPole with termination-corrected differential TD** (our own env) | CartPole itself has no SSP structure — this is the **negative control** | Directly closes our decision-log 17 gap: we reported that naive centering is harmful on episodic CartPole, and De Asis et al. have since shown the correct episodic formulation preserves policy ordering | Already built locally | Minutes (CPU, A2C) | **Tier 1**, as the control |
| 4 | **Continuing Pusher, object resampled every 100 steps** | Object-to-target and ee-to-object **relative** vectors; the Stage-2 manipulation structure with a proper continuing criterion | Same as Reacher | Exists | Moderate; contact makes it noisier | **Tier 2** |
| 5 | **Custom multi-object relative-position navigation** (2-4 objects, randomised layouts, train/test workspace shift) | **Composition**: dimension grows linearly with bound pairs while a grid over the product space explodes; plus unbounded workspace | Continuing by construction; reward defined on relative configurations | Does **not** exist — must be built (small 2-D sim; RatBox-style code is the closest ancestor) | Low to build, cheap to run | **Tier 2** — the only test of the composition claim |
| 6 | Continuing HumanoidStandup / SpecialAnt (weakly communicating, no resets) | Relative height/orientation | Yes, continuing | Exists | Expensive | **Tier 3** — Wan et al. report that no tested algorithm learned these in most runs, so a representation effect is unmeasurable against that failure |
| 7 | mjlab Go1/G1 locomotion (`05-tasks-and-benchmarks.md` Stage 3) | None of SSP's structure: proprioceptive state, no translation invariance, no wrap | The docs already require a continuing conversion for centering | Does not exist yet | Highest in the plan | **Defer** — expensive, and it tests dimensionality rather than structure |

### The specific sub-angle worth stealing from #2

Wan et al. changed the *environment* to fix the topology. The clean experiment is the reverse: leave the
environment alone, keep the raw wrapped angle, and change the *encoder*. Prediction: the periodic encoder
recovers a large part of the 347-1608% with no environment change, and beats a grid encoder near the seam
(where the grid aliases). Falsifier: the periodic encoder performs like the clipped grid.

---

## 4. Design requirements for any of these to be discriminating

1. **Reward rate, not episodic return.** Continuing criterion throughout; the metric is the average reward.
2. **Push gamma up and add a reward offset.** That is where centering is meant to matter
   (Naik et al.); a gamma in {0.99, 0.999} x {offset, no offset} 2x2 is the centering axis.
3. **Structure in the state, explicitly.** Encode only the dims that have it (relative vectors, heading, phase);
   leave raw proprioception raw. The Stage-2 plan's "HexSSP on the Cartesian terms only" is the right shape, but
   it needs to be the *same information* on both arms.
4. **An invariance / shift eval.** Train on region A (or one layout), evaluate on a shifted B. A grid cannot
   transfer; a distance-based kernel predicts it can. This is the discriminating measurement, and no existing
   testbed forces it — it has to be added.
5. **A linear-readout arm.** SSP vs tile coding with a linear head is the clean encoder ablation; an MLP arm on
   both sides answers the separate "can a network absorb the structure?" question.
6. **Matched budget.** Same tuning budget and same search space treatment per arm (Eimer et al.;
   `06-related-work.md`). Centering has no new hyperparameters on the tile side, so this is cheap.
7. **Reuse their testbeds and their continuing conversions.** `DeepRL-continuing-tasks` is open source; the
   conversion work in `05-tasks-and-benchmarks.md` (penalty-and-reset instead of episode end) reimplements
   something that already exists publicly.

---

## 5. Concrete first experiment

Continuing **Reacher with a resampled target**, three encoders (SSP on the relative target vector, tile coding on
the same vector, raw MLP), centering on/off (TD-based), gamma in {0.99, 0.999}, 10 seeds, reward rate over the
last 10k steps, plus a shifted-target evaluation.

Predictions, each with a falsifier:

- **P1** SSP's advantage over the grid is larger at gamma 0.999 than at 0.99 (centering's benefit and the
  encoder's structure interact through the effective horizon).
- **P2** Centering helps all three encoders but helps the *grid* most, because a hard-edged basis is what the
  offset problem hurts most. Falsified if the gain is uniform.
- **P3** Under a shifted target distribution, the grid loses more than SSP. Falsified if they degrade together.
- **P4** (from doc 08) SSP's interference measure is higher than tile coding's; if P1-P3 hold anyway, the claim
  becomes "SSP trades interference for invariance and noise tolerance", which is a sharper and more defensible
  statement than a ranking.

---

## 6. Open items

- The MDPI *Optimizing Spatial State Representation in RL* paper is unchecked (403). Someone should read it
  before we claim the SSP-as-state-encoder-for-control slot is empty.
- Wan et al.'s agent-controlled-reset results suggest resets themselves are a design variable; this project's
  centering work assumes continuing, so the "predefined resets with a penalty" testbeds are the ones to use.
- Our own A2C/SSP provenance issue (`08-...md` section 7) is orthogonal to this survey but blocks any claim about
  SSP on the A2C side.
