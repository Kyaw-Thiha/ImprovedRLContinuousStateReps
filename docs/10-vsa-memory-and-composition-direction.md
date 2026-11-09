# VSA memory and composition: what exists, what the gap is, and whether it deploys

Written 2026-10-06. Companion to [`08-ssp-vs-tile-coding-theory-and-hypotheses.md`](08-ssp-vs-tile-coding-theory-and-hypotheses.md),
[`09-ssp-centering-task-survey.md`](09-ssp-centering-task-survey.md) and `handoffs/07-mujoco-repo-handoff.md`.
Surfaced by alphaXiv `discover_papers` plus web search; confidence labels say how much of each item was actually read.

Motivation: we established that every advantage of SSP/VSA **as a state encoder for a policy** is substitutable —
Fourier features for smoothness, learned encoders for capacity, Deep Sets/attention for variable-count inputs. The
one ingredient with no substitute is the **algebra**: bind, unbind, bundle, permute. That ingredient is only worth
something where you need to *store, retrieve or compose* rather than merely *encode*.

---

## 1. Correct the conflation: encoder / memory / world model are three different uses

| Use | What VSA does | Status in the literature |
|---|---|---|
| **Encoder** (this project's current thread) | Maps a state to a fixed high-dimensional vector that a linear/learned head consumes | Crowded, substitutable; our CartPole evidence is negative on the headline |
| **Memory** | Stores experience or structure as bundles; retrieves by unbinding / similarity; composes by binding | Partially occupied, but not in deep RL control — see section 3 |
| **World model** | Predicts dynamics in a structured representation | Exists as of 2026: *Geometric Priors for Generalizable World Models via VSA* (arXiv:2602.21467, abstract only) |

So "memory framing" is not world models. Memory is about **storage, retrieval and composition**; a world model is
about **prediction**. They are separable and the memory one is the one with the unique ingredient.

## 2. What "memory" concretely means in RL

| Role | Mechanism | Established competitor (the thing to beat) |
|---|---|---|
| **Episodic / associative memory of experience** | Bundle `(key ⊗ value)` pairs; retrieve a value by unbinding with a query key; O(1) writes and reads, no training, bounded size, graceful degradation | RNN / transformer memory; episodic control with a learned key encoder and kNN (MFEC, Blundell et al. 2016; NEC, Pritzel et al. 2017); prioritised replay |
| **Compositional task specification** | Define a new task by binding/bundling known primitives → zero-shot generalisation | Successor features / successor measures; the Boolean Task Algebra for goal-reaching tasks; the unified zero-shot-RL framework; goal-conditioned RL + HER; LfD skill composition |
| **Structured queries over experience** | Algebraic retrieval: "states where I held X near Y" | Filtered replay buffers; learned retrieval; knowledge-graph memory with a learned meta-policy |
| **Continual / lifelong learning** | Frozen codebook; per-task protected memories; no backprop in the representation | HDC continual learning itself (this is HDC's actual deployed niche); replay; EWC; loss-of-plasticity methods |

## 3. What is already taken (do not redo)

| Work | What it covers | Status |
|---|---|---|
| Kanerva, *Sparse Distributed Memory* (NASA TR, 1988); Plate, *Holographic Reduced Representations* (1995) | The origin of high-dimensional associative memory and of binding/unbinding | Foundational; not read here |
| *On separating long- and short-term memories in hyperdimensional computing* (PMC9869149) | Memory architecture inside HDC: item memory vs associative memory, capacity per storage location | Title/abstract |
| *Autonomous Learning with High-Dimensional Computing* (arXiv:2503.23608) | HDC + Marr-Albus/cerebellum, associative and working memory, autonomous learning | Title/abstract |
| **HD-CB** (arXiv:2501.16863, Sapienza 2025) | "The first exploration of hyperdimensional computing for contextual bandits" — HDC in a decision-making setting | Abstract. **Closest predecessor to VSA-in-RL; must be read in full before claiming novelty** |
| VSPG (arXiv:2608.18404, 2026) | Action *hypervectors* + policy gradient; proves the update is advantage-weighted bundling; bit-flip robustness guarantee | Abstract |
| *Gram-Space* (arXiv:2608.01528, 2026) | VSA codebook compression for memory-efficient neuro-symbolic deployment | Abstract |
| *HyperSpace* (arXiv:2604.15113, 2026) | A generalised framework for spatial encoding in HD representations | Abstract |
| Neuro-symbolic meta-policies for temporal-knowledge-graph memory under partial observability (arXiv:2607.18368) | Learning *which* memory heuristic to use in POMDPs | Abstract |
| *A Unified Framework for Zero-Shot RL* (arXiv:2510.20542); *Switching Successor Measures for Hierarchical Zero-Shot RL* (arXiv:2605.13207); *A Goal-Set Characterization of Task Composition in the Boolean Task Algebra* (arXiv:2606.04053) | Zero-shot task composition in RL is an established framework whose algebraic tool is **linear decomposition over task features**, not VSA | Abstract |
| *Jointly Learning Predicates and Actions Enables Zero-Shot Skill Composition* (arXiv:2605.20648); **ATOM-Bench** (arXiv:2606.16826, "a real-world benchmark for atomic skills and compositional generalization in manipulation policies") | Zero-shot composition for manipulation, including a real-world benchmark | Abstract |
| ImageHD (arXiv:2604.21280, FCCM 2026); *Lifelong Intelligence Beyond the Edge using HDC* (arXiv:2403.04759) | HDC's **actual deployment niche**: energy-efficient on-device *continual learning* on FPGA/edge, without backprop | Abstract |

## 4. Where the gap actually is

Three statements, in decreasing confidence:

1. **VSA as *episodic memory for deep RL control* is unoccupied.** HDC memory architectures exist outside RL; HD-CB
   covers bandits; VSPG covers action memories. Nobody appears to have used the binding algebra as the agent's
   experience memory and measured it against RNN/transformer memory and episodic control.
2. **Composition via the binding algebra in RL is unoccupied, but the competitor set is strong and mature.**
   Successor features/measures and the Boolean Task Algebra already give zero-shot task composition with
   guarantees. VSA cannot win on "compose tasks" in general. It can plausibly win on **relational and geometric
   predicates** — "left of", "between", "near X while holding Y" — because successor-feature methods decompose
   *linearly* over task features, which handles weighted goal conjunctions well and relational predicates poorly.
   That mismatch is the defensible niche, and the binding algebra is exactly the operation relational predicates
   need.
3. **The deployment hook is on-device continual learning, not control.** HDC is genuinely deployed for
   energy-efficient, backprop-free, on-device *continual* learning on FPGAs and edge hardware. If real-world
   relevance is the criterion, that is where VSA has industrial evidence today — and it is a different application
   class from "better controller encoder", which is why it is interesting and why it has a different competitor
   set (sparse backprop, replay, EWC).

The honest risk: the RL-side field is thin (one bandit paper), which means both opportunity and little support;
and the composition side is a crowded, high-quality competitor set with theoretical guarantees that VSA does not
have.

## 5. Candidate tasks

| Task | What it tests | Env | Cost |
|---|---|---|---|
| **Boolean/relational goal composition** | Compose trained primitives into unseen relational goals ("reach A while staying right of B") by algebraic operations on the representation | 2-D point maze, or MiniGrid (this group's own earlier setting) | Low; cheap to build, cheap to run |
| **Delayed-cue / memory maze (POMDP)** | Episodic memory: recall an observation from many steps ago to act correctly | Memory-maze or cue-and-choice task | Low |
| **Rare-event imprint** | One-shot storage and later recall of a rare failure; measured under a fixed memory budget | Any env with rare terminations; our CartPole robustness data already has rare events by construction | Low |
| **Continual task sequence** | Forgetting and plasticity with a frozen codebook vs a learned encoder; the deployment-relevant axis | A sequence of MuJoCo/PointMaze tasks, or HDC's own edge-CL protocol | Medium |
| **Real-world compositional manipulation** | Compositional generalisation with real robot evidence | ATOM-Bench (arXiv:2606.16826) | High; use only if the result needs real-robot credibility |

## 6. Cheap deciders before spending anything

Two local experiments, hours each, no RL training:

1. **Kernel/expressivity test** (from `handoffs/07` section 6): characterise SSP's induced kernel and fit the best
   Gaussian. If it is a Gaussian with some bandwidth, SSP is an RFF and the *encoder* thread is dead — but the
   *memory/composition* thread is untouched, because it depends on the algebra, not the kernel.
2. **Algebra test:** on a toy relational-predicate task, does composition by binding recover a held-out predicate
   composition better than (a) linear/successor-feature arithmetic over task vectors and (b) a small MLP on raw
   features? This measures whether the algebra buys anything *before* an RL agent is involved, and it is the
   cheapest possible falsification of the whole direction.

Only if (2) shows the algebra does something successor features cannot should any RL compute be spent.

---

## 7. Ranked: where SSP x centering has a real gap

Checked after the fact against the primary sources. The strongest intersection is **not** in the memory framing.

### 7.1 Best: centering's benefit as a function of the approximator's ability to represent a constant

Naik et al. state their mechanism in exactly these terms — mean-centering "removes a state-independent constant
(that scales inversely with `1 - gamma`) from the value estimates, **enabling the value-function approximator to
focus on the relative differences** between states' values." So the mechanism is stated as a *limitation of the
approximator*, and yet the benefit has never been measured as a function of the approximator. That is the gap.

The corollary is sharp and testable: **the benefit of centering should scale with how badly the basis represents
the constant function.** A basis with a bias feature represents the offset exactly; a basis without one must
approximate a constant of size `r(pi)/(1-gamma)` using non-constant features. Our own CartPole data already shows
the sign: adding `value` centering moved SSP from 461.65 to 470.21 (+8.6) and tile coding from 499.64 to 493.04
(-6.6) — the representation that can represent a constant exactly is the one that does not benefit.

Experiment: basis in {tabular, tile coding, SSP, random Fourier features} x bias feature in {present, absent}
x centering in {none, TD-based} x gamma in {0.99, 0.999} x reward offset in {0, c}, on a cheap continuing control
task, 10 seeds. The result is the **interaction term**, and the gamma axis separates the two mechanisms centering
mixes: offset removal (basis-dependent, cancelled by a bias feature) versus return-scale/conditioning at high
gamma (plausibly basis-independent). Cost is small — a tabular/linear setup makes most of it nearly free, and the
representation comparison fits in a 2-4 dimensional task.

Why this is the most reliable and highest-yield option: the theory is already written down, our data already
points the right way, the effect should be monotone in the offset magnitude (so it is hard to miss), the cost is
hours-to-days rather than GPU-weeks, and it yields a practitioner-relevant rule ("if your network has a bias term,
centering buys you less") regardless of which way the interaction falls. Representation and criterion are
genuinely co-equal here, which is the framing this project wanted.

### 7.2 If it must be memory: centering as drift tracking for stored values

The genuine memory x centering interaction is **non-stationarity of the reward rate**. A memory (VSA bundle,
replay buffer, or episodic store) holds returns computed against the average reward in force when they were
written. If the reward rate drifts — payload, battery, demand baseline, changing task mix — stored values become
stale and biased in a way recency weighting only partly fixes. Centering is a drift-tracking device, so it should
keep stored values comparable across time. Deployment-relevant and apparently unstudied; but it needs a
drifting-reward-rate task to be built, and the effect may be small against a well-tuned baseline.

### 7.3 Weakest: episodic memory as such

Centering has little to say about storage or retrieval quality. Do not force this pairing.

## 8. World models: why this family fits SSP (full survey in section 9)

The one VSA world model that exists (arXiv:2602.21467, NeurIPS 2025 workshop, Imani's group) is instructive:
**learnable** FHRR encoders with a learned group structure, transitions modelled by **element-wise complex
multiplication**, trained to be approximately invariant. On a discrete grid world: 87.5% zero-shot accuracy on
unseen state-action pairs, +53.6% accuracy on 20-step rollouts, 4x noise robustness versus an MLP baseline.

Two observations. First, it works because the encoders are *learned* and the structure is *trained for* — i.e. it
is not the fixed-random SSP that this project has been testing, which is consistent with everything in `08-...md`
about fixed random features. Second, and more important for the direction question: **a world model makes the
algebra load-bearing in a way a policy encoder does not.** A transition function is a map you want to *compose*
(multi-step rollouts = repeated element-wise products), and a geometric prior on that map is a defensible
generalisation claim about the *dynamics* rather than about a value function. That is why the VSA world model
reports zero-shot accuracy and long-horizon rollout gains as its headline metrics — the very metrics a planning
system cares about, and metrics that a policy-encoder comparison never produces.

So: if SSP is to be the protagonist with the algebra earning its keep, the world-model family is the strongest of
the three. **But see section 9 before acting on this**: the mechanism is mature and already deployed under the
name equivariant / group-structured world models, so the lane is open under the VSA *label*, not as an unexplored
mechanism. What is left is continuous control, whether the group structure survives contact with real dynamics,
whether the advantages hold against a modern sequence-model baseline rather than only an MLP, and the three
differentiators listed in section 9.2.

### 8.1 One correction

SSP the *encoder* has no memory module. Memory comes from the VSA framework — superposition (bundling) as storage
and unbinding as retrieval — plus the HDC memory-architecture literature (item memory vs associative memory,
long/short-term separation) and the grid-cell memory models (Vector-HaSH, TEM). So the intuition is right at the
framework level and wrong at the encoder level, and the practical consequence is the same as section 1: the
encoder framing is the one that keeps coming up empty, because it uses only the part of VSA that has substitutes.

---

## 9. World models: what exists, what is already established under another name, and what to target

Surveyed 2026-10-06 via alphaXiv `discover_papers` and web search. This section answers "has SSP x world models been
done, and is there a real problem to solve".

### 9.1 What exists under the name "VSA world model"

Thin. One paper: *Geometric Priors for Generalizable World Models via VSA* (arXiv:2602.21467, NeurIPS 2025
workshop). Learnable FHRR encoders, learned group structure, transitions by element-wise complex multiplication,
trained for approximate invariance. Discrete grid world, MLP baseline: 87.5% zero-shot accuracy on unseen
state-action pairs, +53.6% on 20-step rollouts, 4x noise robustness. Plus hardware work that enables the
deployment angle: *qFHRR* (arXiv:2604.25939) moves FHRR from floating point to quantized phase and integer
arithmetic. Adjacent: VSPG (action side), *Geodesic Flow Matching* (SSP denoising).

### 9.2 The crucial finding: the idea is already established and already deployed — under a different name

The VSA world model's core mechanism (a latent whose transitions are **group actions**, i.e. multiplication by a
group element) is exactly what the equivariant-deep-learning community calls a **group-structured / equivariant
world model**, and that line is mature and on real robots:

| Work | What it establishes |
|---|---|
| Park et al., *Learning Symmetric Embeddings for Equivariant World Models* (ICML 2022, arXiv:2204.11371) | Latent transition given by multiplication with a group element — the same mechanism, 4 years earlier |
| *Learning Abstract World Models with a Group-Structured Latent Space* (arXiv:2506.01529, 2025) | Geometric priors in the latent space for decision-making |
| *Exact equivariance, kept through training, buys zero-shot generalisation across the symmetry group* (arXiv:2606.03003, 2026) | **Provable** zero-shot generalisation across the group for an equivariant encoder + predictor |
| *Flow Equivariant World Models: Structured Memory for POMDPs* (FloWM, 2026) | Structured dynamic memory under partial observability |
| **SWAP: Symmetric Equivariant World-Model for Agile Robot Parkour** (arXiv:2606.19928, 2026) | The family is deployed on a **real robot** for agile locomotion |
| Wang et al., *On-Robot Learning With Equivariant Models* (PMLR v205, 2023) | Real-robot evidence that the symmetry prior transfers |

Consequence for how a VSA world-model paper must position itself: "world model with group-structured transitions
that generalises zero-shot" is **not novel**, and the equivariant line has *provable* guarantees that VSA does not.
A VSA/FHRR contribution has to be something the equivariant line cannot do, and there are exactly three candidates:
**algebraic composition and query in the latent** (bind/unbind/retrieve, compose unseen transformations by algebra
rather than by group action alone), **integer-arithmetic cheapness** (qFHRR — a fixed-point world model for
on-device inference), and **fixed/lightly-learned encoders** (sample efficiency where the group is not known a
priori).

### 9.3 The problems the field agrees are real (and measures)

Long-horizon prediction is a named, actively measured failure mode of world models — these are the problems a
paper should solve, not "improve a benchmark number":

| Failure mode | Evidence it is taken seriously |
|---|---|
| **Compounding error over recursive rollouts** | *Beyond a single latent space* (arXiv:2609.37644): "recursive rollouts accumulate errors"; *WorldCycle* (2608.04964): long-horizon video world models "suffer from compounding errors"; *Beyond Compression* (2609.30198, Sandia): representations trained for stable long-horizon rollout in physical surrogates |
| **Hallucination / catastrophic prediction failure** | *Hallucination in World Models is Predictable and Preventable* (2606.27326): "rollouts remain visually fluent while drifting from ground-truth dynamics"; *Where World Models Break* (2608.22421): "catastrophic prediction failures could be dangerous" |
| **Intervention / counterfactual fidelity** | *The Intervention Gap in Latent World Models* (2608.29998): whether open-loop transitions move task variables the way matched environment interventions do; *Counterfactual Action Evaluation...* (2610.02860, MIT): "low latent prediction error does not establish that a world model distinguishes the consequences of its actions" |
| **Unseen states / action compositions (coverage)** | The 2602.21467 headline metric; the equivariance literature's zero-shot-across-the-group claim |
| **Cost on device** | qFHRR; HDC's edge/FPGA line (ImageHD, arXiv:2604.21280) |

### 9.4 The proposed thesis

**A compositional, integer-arithmetic world model for on-device use that keeps its fidelity over long horizons and
on unseen state-action pairs.** The claim is tested on the four named failure modes above, not on mean rollout MSE:
long-horizon compounding, zero-shot pairs, intervention fidelity, and cost, on a robot-relevant continuous-control
task, against three baselines: an RSSM/Dreamer-class recurrent world model, an MLP/transformer world model, and —
the direct competitor — an **equivariant group-structured world model**, which is the same idea without the algebra.

What would make it a contribution: if the algebraic latent answers *compositional queries* (compose an unseen
transformation, query a counterfactual, retrieve by content) that the equivariant baseline cannot, and does so at
integer-arithmetic cost. What would kill it: if the equivariant baseline matches it on all four failure modes —
then the algebra buys nothing and the honest conclusion is that group structure is the whole story and VSA is a
particular (and more expensive) implementation of it.

Honest odds: the field's own strongest result in this family is a *proof* (2606.03003), and the deployed version
is on a real quadruped (SWAP). Beating that requires the algebra to do work those cannot, which is a narrower and
more specific claim than "SSP is a better representation" — but it is a real claim, and every one of the four
failure modes is something practitioners already measure.

---

## 10. From the mathematics: what SSP can and cannot do, and where it therefore shines

Written 2026-10-06. This section is derived from the construction, not from empirical results, and it is the
cheapest filter available: everything below is a statement about the encoder we already have in the repo.

### 10.1 What SSP is, exactly

FHRR/SSP encodes a point `x` in `R^d` as a unit-modulus complex vector

```
phi(x) = exp(i * Omega @ x)  in C^D,      Omega in R^{D x d}
```

with rows `omega_j` (frequencies). Three exact identities follow:

| Operation | Algebra | Meaning |
|---|---|---|
| Binding | `phi(x) * phi(y) = phi(x + y)` (element-wise) | The map is a **group character**: `(R^d, +) -> (T^D, *)`. Binding *is* translation. |
| Unbinding | `phi(y) * conj(phi(x)) = phi(y - x)` | Relative position, exactly, with no parameters |
| Similarity | `Re<phi(x), phi(y)> / D = (1/D) sum_j cos(omega_j . (x-y)) = kappa(x - y)` | A **stationary kernel** (Bochner). This is the random-Fourier-feature construction |

Consequences that follow immediately:

1. **The function class of a linear readout is a fixed finite Fourier basis.** `V(x) = w . phi(x) = sum_j w_j
   exp(i omega_j . x)`: band-limited, with support only on the frequencies present. Information about `x` outside
   those frequencies is invisible.
2. **"Resolution" and "range" are both set by `Omega`.** The finest representable detail is `~1/max|omega_j|`; the
   range over which `kappa` is informative is set by `min|omega_j|`. A fixed frequency design trades one against
   the other, and `length_scale` moves that trade-off.
3. **`kappa` does not decay to zero at long range for a finite `D`.** A finite sum of cosines with generic
   (incommensurate) frequencies is almost-periodic: its long-range value is a random quantity with magnitude
   `~1/sqrt(D)`, a *floor*, not a decay. If the frequencies happen to be commensurate — which a scale ladder can
   produce — you get genuine aliasing repetition instead. Either way, "far away" does not look like "no support".
4. **Superposition capacity is finite and noisy.** Bundling `m` items gives per-item retrieval signal-to-noise
   `~ sqrt(D/m)`. Storage is graceful-degradation, not free.
5. **The algebra is exact only for group actions.** `phi` composes translations and rotations exactly; it says
   nothing structural about contact, drag, or any nonlinear map.

### 10.2 Where the mathematics says it shines, and where it must not be expected to

| Property (exact, from 10.1) | Consequence | Task regime where it is load-bearing |
|---|---|---|
| Binding = translation, exactly | Composing `T` steps is one multiplication; **the composed map carries no representation error**, so a group-structured transition does not compound | Kinematic, translation/rotation-dominated dynamics: planar pushing, reaching, mobile navigation, docking |
| Unbinding gives relative position | Relative and relational queries are parameter-free; a model learned at one place transfers to another exactly | Any observation whose sufficient statistic is relative |
| Superposition + similarity | Content-addressable recall, graceful degradation, one-shot storage | Memory/retrieval (but see section 7.3) |
| Band-limited function class | Hard resolution limit; **no** fine-scale or high-frequency structure | Moderate-resolution dynamics; not contact impulses |
| No long-range decay / possible aliasing | "Low similarity means unfamiliar" is only valid within a radius; beyond it there is a floor, or repeat | Never rely on it for OOD beyond the working radius |
| Capacity `~ D/m` | Bounded number of reliably held items | Few entities per bundle |

So the mathematical answer to "where would SSP shine best" is narrower and sharper than "continuous control":
**approximately group-structured, moderate-resolution dynamics expressed on relative coordinates** — mobile
robots, planar manipulation, navigation, reaching — where the algebra is exact and the horizon composition is
free. And the same mathematics says it must *not* be expected to help on contact-rich, high-frequency or
non-symmetric dynamics, nor to deliver trustworthy out-of-support predictions.

### 10.3 The four world-model failure modes, scored by the mathematics *before* any experiment

| Failure mode | Does the algebra address it? |
|---|---|
| **Compounding error over rollouts** | **Yes, for the group part.** Composition is exact, so a translation/rotation-dominant transition does not accumulate representational error. No help at all for the nonlinear part |
| **Intervention / counterfactual fidelity** | **Yes, structurally.** Different actions are different group elements, and distinct group elements are near-orthogonal in high `D`, so the model *necessarily* distinguishes them — precisely the property the MIT probe finds is not guaranteed for MLPs. Weakness: small action differences are only distinguishable if `Omega @ (a1 - a2)` is not near a multiple of `2*pi` |
| **Hallucination / catastrophic failure** | **Weaker than it first looks.** The support argument (far state => low similarity => no confidence) is broken by the `~1/sqrt(D)` floor and by aliasing. A learned readout can amplify that floor into confident nonsense |
| **Coverage / unseen state-action pairs** | **Interpolation yes, extrapolation no.** Kernel expansion interpolates within the frequency band's working radius; outside it, the band limit and the floor govern |
| **On-device cost** | Integer/fixed-point arithmetic is available (qFHRR); unrelated to the algebra |

### 10.4 The cheapest decisive measurement

Everything above is a claim about the encoder, so all of it can be measured on the encoder alone, in an hour,
with no training and no environment:

1. **`kappa(Delta)` as a function of separation**, for the actual `Omega` used in the repo, in 1-D and 4-D. Report:
   the decay rate, whether the envelope decays or oscillates, the long-range floor, and the separation at which
   the floor dominates. This gives the honest working radius and kills or confirms the OOD/hallucination claim.
2. **Action separation** `||phi(a1) - phi(a2)||` against `|a1 - a2|`, to find the action resolution — i.e. whether
   intervention fidelity holds at the action scale the task actually uses.
3. **Frequency coverage**: `max|omega_j|` and `min|omega_j|` against the task's dynamic range, to state the
   resolution limit as a number rather than a hope.

If the long-range floor dominates well inside the task's working range, then the coverage and hallucination claims
are dead and only compounding-error and intervention-fidelity survive — which is still a coherent, narrow, and
genuinely deployment-relevant thesis, but it must be stated that way.

**And the one that matters most, because it is the only genuine differentiator.** Exact composition of group
actions is *not* unique to VSA: an exactly equivariant model (arXiv:2606.03003) composes just as exactly, so
free-horizon composition and structural action separation are table stakes, not an advantage. The capabilities
with no analogue elsewhere are:

- **chained algebraic queries** over the latent — `unbind(bind(A, x), B)` style expressions, i.e. questions like
  "where was the object when the gripper was here", answered by algebra rather than by forward simulation;
- **integer/fixed-point cost** (qFHRR) for on-device inference.

Test 4, therefore, is the crux for the whole direction: **query capacity.** Bind `m` items into one vector,
perform one to three chained unbindings, and measure retrieval accuracy as a function of `m` and `D`. The
mathematics predicts accuracy collapsing as `sqrt(D/m)`, so this single curve tells you how many entities a
queryable world model can hold at realistic `D` — and whether chained queries are a capability or a slogan. It is
an hour of encoder-only work with no training, and if it fails, VSA world models reduce to "group structure,
implemented more expensively", which the equivariant line already owns.

---

## 11. Test battery: encoder-only, no training, one afternoon

Everything here runs on the encoder and a codebook. No environment, no RL, no GPU. Run each block for the three
frequency designs we can build — the legacy encoder (`n_scales=8`, `linspace(0.5, 2)`), the current one
(`n_scales=5`), and a dense-Gaussian-spectrum RFF control — because the comparison between them is itself a
result about the scale ladder.

| # | Test | Protocol | Metric | Pass / fail |
|---|---|---|---|---|
| **T0** | Algebra parity | Sample `x, y`; check `phi(x) * phi(y) == phi(x+y)`, `phi(y) * conj(phi(x)) == phi(y-x)`, and that `kappa` depends only on `x-y` | Relative error | Pass at `~1e-12`. If this fails, nothing downstream means anything (cf. the legacy A2C lesson) |
| **T1** | Kernel shape and floor | `kappa(Delta)` for 10k pairs, `Delta` swept to 5x the task's state range, in 1-D and 4-D | decay half-width, whether the envelope decays or oscillates, the long-range floor level | Pass if floor << 0.1 inside the task's working range. **This decides whether any OOD / hallucination claim is viable** |
| **T2** | Resolution | `kappa` along each axis; extract `max` and `min` frequency magnitudes | finest representable wavelength against the task's dynamic range | Pass if the finest wavelength is well below the smallest feature the task needs to distinguish |
| **T3** | Action separation | `||phi(a1) - phi(a2)||` against `|a1 - a2|` across the action range | the action difference at which the latent change clears the T1 floor | Pass if that resolution is finer than the smallest action difference the task must distinguish. This is intervention fidelity, measured directly |
| **T4** | **Query capacity (the crux)** | Bundle `m` items `key_i (*) value_i`; unbind with `key_j`; retrieve by nearest neighbour in the value codebook. Sweep `m` from 1 to 100 and chain depth from 1 to 3 at `D` in {201, 1001, 1601} | retrieval accuracy vs `m` and vs chain depth | Pass if accuracy is high for `m` equal to the task's entity count **and** for chained queries. Theory predicts collapse as `sqrt(D/m)`, and error compounding with depth |
| **T5** | Group structure of the task | For a candidate dynamics, fit one latent transformation `t` per state and measure its variance across states | the spread of `t` | Tells you how group-structured the task is — i.e. whether this is a task SSP should be applied to at all. Do this *before* choosing the task |
| **T6** | Degraded-cue recall | Zero out a fraction `p` of the query dimensions before retrieval | accuracy vs `p` | Pass if graceful — this is the partial-observability / sensor-dropout claim |
| **T7** | Variable entity count | Encode `N` entities (bound to roles) into one vector and retrieve each individually, `N` in {1, 2, 4, 8} | per-entity accuracy vs `N` | Pass if every entity is retrievable at `N = 8` in one fixed-size vector |

Decision rule: **T1 and T2 gate the coverage/OOD claims; T3 gates intervention fidelity; T4, T6 and T7 gate the
only genuinely unique capability.** If T4 fails — in particular if chained queries collapse at realistic `D` — VSA
world models are group structure implemented more expensively, and the direction should be dropped rather than
rescaled. If T4 passes and T1 shows the floor is far below the working range, there is a real, narrow, cheap
thesis: a *queryable and invertible* world model, with a bounded streaming memory, for kinematic
symmetry-dominated dynamics.

---

## 12. What SSP was actually built for, and where that meets real-world need

Written 2026-10-06. This is the design-intent filter. Two questions have to be answered together: *is it
real-world relevant* and *was SSP made for this*, as opposed to retrofitted.

### 12.1 The design intent

SSP is the Semantic Pointer Architecture (SPA) extended to continuous space. The SPA (Eliasmith's line, including
the large-scale spiking cognitive model Spaun, Science 2012) was built around three things:

1. **A content-addressable declarative memory** — a cleanup/associative memory that recalls a stored item from a
   *partial or noisy* cue.
2. **Structured working memory** — items bound to slots, order preserved, a variable number of items held in a
   fixed-size buffer.
3. **Representation transformation by algebra** — deriving and applying a transformation between two
   representations by binding, rather than learning a separate model.

Komer & Eliasmith then added *continuous* space to this by fractional binding: an SSP is what you get when the SPA
needs to represent a point in a continuum rather than a discrete symbol. So the original purpose of the SSP
construction is **spatial and relational representation with algebraic manipulation**, and the purpose of the
surrounding machinery is **memory and structure**.

### 12.2 The intersection, ranked

Classical geometry owns the known-group spatial case (SE(3), quaternions, DMPs), which is why "robot control" is
the weakest domain despite being the most SSP-native on paper. The memory machinery has no such incumbent.

| Rank | Problem | Real-world relevance | Was SSP built for it? | Who already owns it |
|---|---|---|---|---|
| 1 | **Content-addressable memory with cue completion** — recall from partial, occluded or noisy queries; one-shot storage of rare events; bounded footprint | High: POMDP memory, sensor dropout, occlusion, rare-event recall, on-device budgets | **Yes — this is the SPA's declarative-memory module**, and SSP inherits it | Transformers/RNN memory (strong, not content-addressable); kNN/episodic control (learned key encoder, unbounded store); HDC-CL in the edge literature |
| 2 | **Structured working memory: bind to slots, hold a variable number of items, query one slot** | High: manipulation and instruction following with changing object counts | **Yes — the SPA's working-memory usage pattern** | Attention/Deep Sets over the set |
| 3 | **Spatial/relational algebra** — derive a transformation and apply it elsewhere exactly | High, but partly solved classically | **Yes — the original SSP purpose** | SE(3)/quaternions/DMPs for known groups; equivariant nets for learned groups with a known symmetry |
| 4 | Multi-agent state with changing membership | Medium — swarms/fleets | Not originally; a natural extension of binding | GNNs/attention; concatenation with a mask is a strong trivial baseline |
| 5 | On-device continual learning | High — but a systems/embedded community | Not originally | HDC-CL itself |

**The conclusion the filter gives.** The one problem that is simultaneously maximally real-world relevant and
literally what the architecture was designed for is **content-addressable memory with cue completion** (rank 1),
with **structured working memory** (rank 2) as its natural companion. Both are gated by the same measurement:
query capacity and chain depth at realistic `D` (test T4), plus degraded-cue recall (T6) and variable entity count
(T7). If those pass, the direction is not a retrofit — it is the intended use, tested for the first time in a
control setting.

---

## 13. The concrete benchmarks that already exist (surveyed 2026-10-06)

### 13.1 POPGym — the cheap harness for rank 1

*POPGym: Benchmarking Partially Observable RL* (arXiv:2303.01859, ICLR 2023, `github.com/proroklab/popgym`): 15
POMDP environments, **13 memory baselines**, RLlib-based, low-dimensional observations. Cost is the decisive
feature: **"many of our baseline models converge in under two hours of training on a single consumer-grade GPU"**,
with environment throughput of 1.4k-250k steps/s on a laptop.

Tasks that test exactly rank 1 (content-addressable recall from a cue):

| POPGym task | What it tests | Maps to |
|---|---|---|
| **Concentration** | flip cards, remember value *and* position of previously seen cards | content-addressable recall, cue completion |
| **Count Recall** | maintain running counts per value, answer a *query* for one value's count | bind `key -> value`, retrieve by unbinding — the algebra itself |
| **Repeat Previous (k)** | output the value seen `k` steps ago | memory duration |
| **Autoencode** | watch a sequence, then decode it in order | **superposition capacity** — directly tests the `sqrt(D/m)` prediction |
| Multiarmed Bandit | remember which bandits pay over a long episode | long-horizon memory |

Two POPGym findings that matter for the plan:

- **Plain GRUs beat the modern memory architectures** (FART, FWP, S4D, DNC, LMU, TCN, IndRNN) in RL; the authors
  note the disconnect from supervised learning is unexplained. A cheap, simple memory mechanism is therefore
  competing in a *beatable* field, not against transformers at scale.
- **Navigation tasks are useless for memory claims** (their MLP beat every memory model on Labyrinth), and a
  **positional MLP is a mandatory baseline** (it nearly solves masked stateless Cartpole). Also: on Autoencode and
  Battleship all 13 models converge to nearly the same reward, which the authors attribute to PPO rather than the
  memory models — a task-selection warning.
- Metric caution: POPGym reports **MMER** (max mean episodic reward over epochs), which is selection-inflated.
  Report mean-of-last-`k` under a fixed budget instead, for the reason this project already learned the hard way.

### 13.2 MIKASA — the robotics translation, for rank 1 *and* rank 2

*Memory, Benchmark & Robots* (arXiv:2502.10550, ICLR 2026, `pip install mikasa-robo-suite`, MIT licence):
**MIKASA-Robo = 32 tabletop manipulation memory tasks in 12 categories, built on ManiSkill3** (GPU-parallel
training), plus expert-trajectory datasets, plus published online-RL, offline-RL and VLA baselines. Its taxonomy —
**Object, Spatial, Sequential, Memory Capacity** — is almost a restatement of our rank 1 / rank 2 split.

| MIKASA task | Memory type | Maps to |
|---|---|---|
| **ShellGame** (Touch/Push/Pick) | Object | ball hidden under a cup; recall its location from an earlier observation, then act — content-addressable recall |
| **RememberColor / RememberShape / RememberShapeAndColor** (3/5/9) | Object | observe an attribute, it disappears, then select it among N distractors — cue completion |
| **BunchOfColors / SeqOfColors** (3/5/7) | Capacity | hold a *set* of items and retrieve all of them |
| **ChainOfColors** (3/5/7) | Sequential | hold a set **and its order** — slot binding |
| Intercept / InterceptGrab | Spatial | estimate velocity from remembered positions |
| RotateLenient / RotateStrict | Spatial | remember the initial pose and apply a commanded rotation |
| TakeItBack | Spatial | remember the start position and return there |

Crucially, the authors show **PPO-MLP in state mode (full observability) solves these at 100% success** — so the
failures under partial observability are attributable to memory alone, which is exactly the clean design we want.

### 13.3 Rank 2's competitor set is crowded; rank 1's is not

If the variable-slot / working-memory claim (rank 2) is pursued as its own thesis, the incumbents are the
object-centric and relational lines: **MetaSlot** (arXiv:2505.20772, "break through the fixed number of slots"),
Slot-MPC (2605.14937), COMET (2606.14418), FOCI (2609.08743), World-as-Graph (2609.38927), and GNN relational RL
(2009.12462). That is a fast-moving, well-resourced line attacking the same limitation. Rank 1's competitors are
memory *architectures* (RNN/attention/state-space), where POPGym's own evidence says the simple ones win.

**Therefore: fold rank 2's tasks in as an extra axis of the rank-1 study (BunchOfColors, SeqOfColors,
ChainOfColors, Autoencode), rather than making it a separate thesis.**

### 13.4 The plan, cheapest first

| Stage | What | Cost |
|---|---|---|
| 0 | Encoder-only tests T0-T7 (section 11) — algebra parity, kernel floor, resolution, action separation, **query capacity**, degraded cue, variable entity count | minutes-hour, numpy, no training |
| 1 | **POPGym**, the diagnostic subset where memory demonstrably matters: Concentration, Count Recall, Repeat Previous, Autoencode. Baselines: MLP, **PosMLP**, GRU, LSTM — using POPGym's published numbers where available | ~2 GPU-h per run; a first pass of ~30 runs is a few days on one consumer GPU |
| 2 | **MIKASA-Robo** for robotics credibility: ShellGame (rank 1) and BunchOfColors / SeqOfColors / ChainOfColors (rank 2). Baselines: the published online-RL set plus GRU/attention | ManiSkill3 is GPU-parallel; manageable |
| 3 | Only if stages 0-2 hold: the deployment axis — footprint, writes, latency — where POPGym already reports per-model parameter and memory efficiency, so the comparison is against published numbers | low |

Each stage is a stop point. Stage 0 costs an afternoon and can kill the whole direction; stage 1 costs a few GPU-days
and decides whether the algebra is competitive against a GRU; stage 2 decides whether it survives contact with a
robot.

---

## 14. MIKASA task shortlist, cost, and the other robotics memory benchmarks

Read from `arxiv.org/html/2502.10550v3` sections 6.1-6.4 and the RoboMME abstract, 2026-10-06.

### 14.1 The tasks worth doing, in order

MIKASA-Robo's episode horizons are **short** (timeouts 60-180 steps: 60 for the `Remember*` tasks, 90 for
`ShellGame`/`Intercept`/`Rotate*`, 120 for the colour sets, 180 for `TakeItBack`), which is why these are tractable
where 3-D navigation benchmarks are not.

| Rank | Task(s) | Memory type | Why it is worth doing |
|---|---|---|---|
| 1 | **RememberColor 3/5/9**, **RememberShape 3/5/9** | Object | The cleanest rank-1 test: observe one attribute, it disappears, then select it among 3/5/9 distractors. The published baselines show a **cliff**: PPO-LSTM beats PPO-MLP at 3 colours, and **both collapse to near-zero at 5 and 9**; under sparse rewards both fail even at 3. That cliff is exactly the discriminating interval a memory mechanism should occupy |
| 2 | **ShellGame Touch/Push/Pick** | Object | Ball hidden under a cup — recall its location from an earlier observation, then act. Published VLA numbers show headroom (Octo 0.46, OpenVLA K=8 0.47, π0 0.33) |
| 3 | **ChainOfColors 3/5/7** | Sequential | Rank 2's sharpest test (hold a set *and* its order). Note: **no published baseline solves it**, which is both the opportunity and the risk — it may be unlearnable with current algorithms |
| 4 | **BunchOfColors / SeqOfColors 3/5/7** | Capacity | Direct test of the superposition-capacity prediction (`sqrt(D/m)`). Same "no baseline solves it" caveat |
| — | Intercept / InterceptGrab / Rotate* / TakeItBack | Spatial | Velocity estimation and pose memory — closer to the *world-model* branch than to recall. Defer |
| — | RGB + VLA track | — | Heavy (two camera views, fine-tuned Octo/OpenVLA/π0/SpatialVLA). Use published numbers for comparison, do not re-run |

**The cheap configuration.** MIKASA offers observation modes `state` (full vector + oracle + TCP pose), `RGB`,
`joints`, `oracle`, `prompt`, and any combination. `RGB+joints` is their standard memory-testing configuration and
`state` is described as reserved for MDP tasks because it contains oracle information. For a compute-limited
project the efficient entry is **joints (+ TCP) with dense rewards** — the memory requirement comes from the
*temporal occlusion*, not the modality, and this avoids the vision stack entirely. That choice should be verified
against Appendix I (compute resources) before committing.

### 14.2 What the published baselines establish (and what they do not)

- **Online RL:** PPO-LSTM > PPO-MLP at low difficulty; both fail at high difficulty. SAC and TD-MPC2 are more
  sample-efficient but lack explicit memory and "become a critical shortcoming" as difficulty rises — the authors
  conclude that algorithms common in the robotics community are *inappropriate* for memory-intensive tasks. That
  is a favourable framing for a dedicated memory module.
- **Offline RL:** **none** of Decision Transformer, RATE, BC, CQL or Diffusion Policy solved the majority of
  tasks, and **none solved the Memory Capacity or Sequential Memory categories**. Datasets are 1000 successful
  trajectories per task, released.
- **VLA:** Octo, OpenVLA, π0, SpatialVLA all degrade sharply with memory demand, and a **real-robot experiment with
  π0.5 reproduces the same pattern**, with the authors concluding that "memory, rather than embodiment or
  perception noise, is the primary limitation". Useful one-line justification for the direction.
- **A confound to know about:** VLA action *chunking* (K=8) partially bypasses memory by generating a whole
  trajectory from the early cue. Not applicable to a non-chunked PPO/SAC setup, but it is why the benchmark uses a
  difficulty hierarchy.

### 14.3 The other robotics memory benchmarks: assessment for a compute-limited project

| Benchmark | What it is | Verdict |
|---|---|---|
| **RoboMME** (arXiv:2603.04639) | 16 manipulation tasks under a temporal/spatial/object/procedural taxonomy, plus **14 memory-augmented VLA variants built on the π0.5 backbone**; headline finding is that memory-representation effectiveness is highly task-dependent | **Too heavy to run** (π0.5-scale fine-tuning). Read it for the taxonomy and for positioning; cite its finding that memory design is task-dependent |
| **MemoryBench / SAM2Act** (Fang et al. 2025, arXiv:2501.18564) | Three manipulation tasks, spatial memory only, on **RLBench** | **Skip for compute**: the MIKASA authors note RLBench "does not allow efficient parallelization of training", which is why it has only three tasks. Cite only |
| **Memory-improvable-domains suite** (arXiv:2508.00046) | A suite of domains designed so that memory measurably improves performance; includes the classic passive T-Maze lineage | **Unread** — worth a look as an alternative cheap harness, but POPGym already covers this ground with published per-model baselines |
| **POPGym** (arXiv:2303.01859) | 15 POMDPs, 13 memory baselines, < 2 GPU-h per model | The cheap filter (section 13.1) |

**Order of attack:** POPGym (days) → MIKASA `RememberColor3/5/9` and `ShellGameTouch` in joints+dense mode (the
claim, at the difficulty where the published baselines still discriminate) → the capacity/sequential tasks only if
those work.

### 14.4 Which of these tasks actually match SSP's unique capability

This matters, because the obvious guess is wrong. SSP's unique capability is **holding many items simultaneously in
one fixed-size vector and retrieving them individually** (superposition, capacity `sqrt(D/m)`) and **retrieving by
cue/an algebraic query**. Sorting the MIKASA tasks by that:

| Task | Structural fit for SSP |
|---|---|
| **BunchOfColors / SeqOfColors / ChainOfColors 3/5/7** | **Direct hit.** A *set* of 3/5/7 items held at once, retrieved individually, and for `Chain` retrieved **in order** — that is exactly superposition plus slot binding. These are the only MIKASA tasks where the capability SSP uniquely has is structurally load-bearing. They are also the ones **no published baseline solves**, so the upside and the risk are both high |
| ShellGameTouch / Intercept | Partial: spatial binding of one location — SSP can represent it, but so can a GRU with a state, so there is no *structural* advantage to demonstrate |
| RememberColor / RememberShape 3/5/9 | **Weak fit.** The agent holds **one** attribute across a delay and then matches it. A single-item memory is what a GRU is for; more distractors make the *decision* harder, not the memory wider. Do not expect SSP to shine here, and do not use it to justify the direction |

Consequence: the cheap proxy for the same capability is **POPGym's Autoencode and Count Recall** (encode a series
into one vector and decode it; maintain counts per value and answer a query), and the robotic version is
**BunchOfColors / SeqOfColors / ChainOfColors**. The single-item tasks are useful as a difficulty ladder and as
GRU-comparable controls, not as the SSP claim.

### 14.5 What the repo actually ships, and how our memory module attaches

Verified 2026-10-06 from `github.com/CognitiveAISystems/MIKASA-Robo` (README) and PyPI.

**Shipped.** Environments; trajectory datasets (22,500 PPO/motion-planning oracle trajectories, 6M+ transitions, in
RLDS and LeRobotDataset v3, on Hugging Face); a `baselines/` directory; an evaluation protocol with horizon splits;
and *published numbers* for PPO-MLP, PPO-LSTM, SAC and TD-MPC2 (online), DT, RATE, BC, CQL and Diffusion Policy
(offline), and Octo, OpenVLA, π0 and SpatialVLA (VLA). The repo's main branch has moved to **MIKASA-Robo-VLA: 90
tasks with language instructions**, `obs_mode="rgb"`, ManiSkill3 GPU backend; the **original 32-task RL
implementation is the `mikasa-robo-rl` branch** (`mikasa_robo_suite/rl/`, PyPI `mikasa-robo-suite==0.0.5`).

**Not shipped.** There is no plug-in memory slot. Baselines are fixed architectures; a new memory mechanism is ours
to write.

**Correction to section 14.1.** The "cheap configuration = joints (+TCP)" suggestion does not hold for the ranked
tasks: the remembered attribute in `RememberColor`/`RememberShape` and the set tasks is **visual**, and `joints`
mode carries only joint states and TCP pose. `state` mode carries it but includes the oracle fields (which leak the
answer). So those tasks require **RGB** — the compute-appropriate route is a single top-view image with a small CNN
and dense rewards, *not* the two-view VLA pipeline. The genuinely cheap vector-observation path is MIKASA-Base
(which wraps Memory Cards, Numpad, MiniGrid-Memory, POPGym, Object Permanence, Passive T-Maze and others under one
API) plus POPGym itself.

**How the VSA memory attaches.** Write the module once against **POPGym's `MemoryModel` interface**
(`memory_forward` / `initial_state`) — it exists precisely so memory mechanisms can be swapped, has 13 reference
implementations and published numbers, and its tasks are vector-observation and < 2 GPU-h per model. Then place the
identical module in MIKASA's PPO-LSTM slot in the `mikasa-robo-rl` branch. The comparison is then clean by
construction: same policy head, same PPO, same observations; only the memory differs.

Concrete module design (the SPA's declarative memory plus working memory):

1. **Item memory (codebook):** one random unit-modulus complex vector per discrete value (colour, shape) — for
   continuous quantities, the FHRR phase encoding `exp(i Omega x)`.
2. **Write:** `memory += slot_code_t * value_code_t` — superposition with a *slot code* for position, so order is
   recoverable (`ChainOfColors` needs this; an unbound bundle cannot preserve order).
3. **Read:** unbind with the query's slot code, then clean up by nearest neighbour in the codebook — the SPA's
   cleanup memory, and the mechanism that turns a noisy unbinding into a usable value.
4. **Policy:** `[o_t, read_t]` (or the raw bundle) into an MLP head producing the PPO action distribution and value.

Baselines to report at matched parameter count: PPO-MLP (no-memory floor), PPO-LSTM/GRU (the incumbent), plus a
transformer-style memory (FART) and an explicit "store everything, exact kNN under a matched byte budget"
baseline. Report **footprint (bytes, writes per step) as a first-class axis**, since the capability claim alone is
weak against a GRU.

Where the mechanism should show up: tasks whose item count exceeds a GRU's practical capacity (the 5/7/9 variants of
Bunch / Seq / Chain) and tasks where recall is by **cue** rather than by recency (Concentration, Memory Cards,
RememberColor). If it does not show up there, it will not show up anywhere.
