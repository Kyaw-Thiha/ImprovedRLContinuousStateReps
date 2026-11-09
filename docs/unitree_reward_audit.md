# Audit: do Unitree's mjlab velocity configs carry a harmful reward offset?

Written 2026-11-05. Read-only audit: nothing was launched on pods, nothing in the project tree or the running campaign was touched. Repos were cloned to `/tmp/unitree_audit/` (scratch, cleared on reboot). The only compute was a few hundred steps of zero-action rollouts on this machine's CPU (no GPU), using the existing `mjlab_centering/.venv` read-only.

Every number below is labelled **analytic** (derived from the term formulas and the command distribution), **measured** (a CPU zero-action rollout, see section 9 for exactly what was run), or **est.** (analytic with assumed state statistics, flagged where the assumption matters). Anything I could not determine is in section 10.

## 0. Sources and commits

| repo | commit | notes |
|---|---|---|
| `unitreerobotics/unitree_rl_mjlab` | `1425b15f73bd4095f0df53709d7c389c3eb9e790` (2026-04-13, "Fix the warnings during rough-terrain training.") | latest `main` at clone time, confirmed by `git ls-remote origin HEAD` |
| `mujocolab/mjlab` (our reference) | `8bb2bc9e187797689ec82c78c35019afd2fd0376` (2026-10-18), `pyproject.toml` says version 1.6.0 | what `mjlab_centering/pyproject.toml` pins |
| `mujocolab/mjlab` tag `v1.2.0` | `5af32e378dcb93c9e881ace83cc5a3f5d373fe60` | what Unitree pins (`setup.py:5`) |

Unitree-side paths below are relative to the Unitree repo root (`src/tasks/velocity/...`), mjlab-side paths relative to the mjlab repo root (`src/mjlab/tasks/velocity/...`). Line numbers are at the commits above.

Velocity tasks registered by Unitree: `Unitree-{Go2,G1,G1-23Dof,H1_2,A2,As2,R1,H2}-{Flat,Rough}` (`config/*/__init__.py`; the README lists only six of the eight). **All of them build on one shared base config** (`velocity_env_cfg.py`), and every `rl_cfg.py` is identical apart from `experiment_name`. The only per-robot reward differences are the `pose` std tables, the `foot_gait` offsets (4 feet vs 2) and one extra `self_collisions` term (weight -1.0) on the humanoids (`g1/env_cfgs.py:138-142`, same in `g1_23dof`, `h1_2`, `h2`, `r1`). So Go2 and G1 cover the family; A2/As2 are Go2-shaped and H1_2/H2/R1/G1-23Dof are G1-shaped.

## 1. The headline in four lines

1. **Unitree's reward has no additive constant at all.** Every positive term is `weight * kernel` with the kernel in [0, 1] (exp of a squared error, or a contact-match fraction); every other term is a penalty <= 0. No term returns a constant (no alive bonus). The strictly constant part of the reward is exactly 0 (analytic).
2. **Its total positive budget is smaller than mjlab's.** Sum of positive weights is 3.5 (track_lin 1.0 + track_ang 1.0 + pose 1.0 + foot_gait 0.5), i.e. at most **0.070 per step** at dt = 0.02. mjlab's Go1/G1 have 6.0 (2+2+1+1), i.e. at most 0.120 per step. Unitree's *ceiling* is below the c = 0.1 offset we tested.
3. **Per-step level, standing and trained, is about half of mjlab's** (Go2 ideal stand ~0.042 vs mjlab Go1 0.077; Go2 trained ~0.053 vs 0.10). So the "equivalent offset c" relative to mjlab is **negative, about -0.035 to -0.05**, which maps to the c = 0 row (19/20 uncentered walkers), not to c >= 0.1.
4. **Verdict: no, Unitree's configs do not sit in the harmful range** (with two caveats in section 8: their gamma is 0.99, not our 0.999, and early-training reward is net negative because of a -4 per-fall term; our table says nothing about negative levels).

## 2. Reward terms, per task

All terms are multiplied by `dt = 0.005 * 4 = 0.02` by mjlab's `RewardManager` (`scale_rewards_by_dt` defaults to True; the sim timestep and decimation are at `velocity_env_cfg.py:420-429`). The "per-step range" column already includes dt. Classification: **C** = constant (alive-style), **NC** = near-constant kernel (always positive, high when upright and roughly on-pose), **SD** = state-dependent term (tracking kernels, penalties, event terms).

### 2.1 Unitree-Go2-Flat (commit `1425b15`)

Base terms are `velocity_env_cfg.py:261-354`; Go2 specialisation is `config/go2/env_cfgs.py:88-113`.

| term | weight | functional form | per-step range (with dt) | class | where |
|---|---|---|---|---|---|
| `track_linear_velocity` | +1.0 | `exp(-(abs(c_xy - v_xy)^2 + 2 v_z^2) / 0.25)` | [0, 0.020] | SD (kernel; floor depends on command) | env_cfg:262-266, rewards.py:23-40 |
| `track_angular_velocity` | +1.0 | `exp(-((c_z - w_z)^2 + 0.05 abs(w_xy)^2) / 0.5)` | [0, 0.020] | SD (kernel) | env_cfg:267-271, rewards.py:43-60 |
| `pose` | +1.0 | `exp(-mean(err_j^2 / std_j^2))`, err = joint - default; std by command speed: standing (<0.1) hip/thigh/calf 0.05/0.1/0.15, walking and running 0.15/0.35/0.5 | [0, 0.020]; = 0.020 at the default pose | **NC when standing; drops to ~0.6-0.85 when walking** | env_cfg:277-289, go2/env_cfgs.py:88-102, rewards.py:326-407 |
| `foot_gait` | +0.5 | mean over feet of `1[is_stance(phase) == in_contact]`, phase clock period 0.6 s, stance threshold 0.56, zeroed when command <= 0.1 | [0, 0.010] | SD, but **a robot that stands still on all four feet earns ~0.56 of it** (see below) | env_cfg:304-315, rewards.py:188-212 |
| `body_orientation_l2` | -1.0 | `abs(g_xy)^2` of projected gravity, base_link | [-0.02, 0] (-0.02 at 90 deg), ~0 upright | SD penalty | env_cfg:272-276, rewards.py:63-84 |
| `body_ang_vel` | -0.05 | `abs(w_xy)^2`, base_link | <= 0 | SD penalty | env_cfg:290-294, rewards.py:109-118 |
| `angular_momentum` | -0.025 | `abs(L)^2`, whole body | <= 0 | SD penalty | env_cfg:295-299, rewards.py:121-131 |
| `is_terminated` | **-200** | 1 on a non-timeout termination step | **-4.0 on a termination step, else 0** | SD event penalty | env_cfg:300 |
| `joint_acc_l2` | -2.5e-7 | `sum(qdd^2)` | <= 0 | SD penalty | env_cfg:301 |
| `joint_pos_limits` | -10.0 | soft joint-limit excess | <= 0, ~0 nominal | SD penalty | env_cfg:302 |
| `action_rate_l2` | -0.05 | `sum((a_t - a_{t-1})^2)` | <= 0 | SD penalty | env_cfg:303 |
| `foot_clearance` | -1.0 | `sum_feet abs(z - 0.10) * abs(v_xy)`, active when command > 0.1 | <= 0 | SD penalty | env_cfg:316-325, rewards.py:163-185 |
| `foot_slip` | -0.25 | `sum_feet abs(v_xy)^2 * in_contact`, active when command > 0.1 | <= 0 | SD penalty | env_cfg:326-335, rewards.py:267-294 |
| `soft_landing` | -1e-3 | `sum_feet force at first contact`, active when command > 0.05 | <= 0 | SD penalty | env_cfg:336-344, rewards.py:297-323 |
| `stand_still` | -1.0 | `sum(joint - default)^2` only when command <= 0.1 | <= 0 | SD penalty | env_cfg:345-353, rewards.py:410-427 |

The `foot_gait` row deserves a note because it is the one place where Unitree has a near-floor term: with all feet down, `is_stance == in_contact` holds exactly when the clock says "stance", which is 56% of the cycle for each foot. Averaged over phase this is 0.56 (analytic), so a standing robot collects 0.56 x 0.5 x 0.02 = 0.0056/step of it (95% of the time, since 5% of envs are zero-command and get 0). A perfect gait gets 1.0. So about half of this term is a floor, but it is only 0.0056 of a ~0.04 total.

### 2.2 Unitree-G1-Flat (commit `1425b15`)

Same base terms and weights as 2.1 (`velocity_env_cfg.py:261-354`). Differences: `pose` std tables (`config/g1/env_cfgs.py:94-132`: standing 0.05 on every joint, walking/running per-joint 0.1-0.5), `foot_gait` offsets `[0.0, 0.5]` (the base value, two feet), orientation and angular-velocity terms use `torso_link` (`g1/env_cfgs.py:134-135`), and one added term:

| term | weight | form | per-step range | class | where |
|---|---|---|---|---|---|
| `self_collisions` | -1.0 | number of substeps with a self-contact force > 10 N | <= 0 (up to -0.08 for 4 substeps) | SD penalty | `g1/env_cfgs.py:138-142`, rewards.py:87-106 |

Because the weights are identical, the positive budget (0.070/step) is the same as Go2's. The penalty weights that matter for a humanoid are `joint_pos_limits` -10, `body_orientation_l2` -1.0 and `angular_momentum` -0.025.

### 2.3 mjlab reference (commit `8bb2bc9e`, for comparison)

Base terms `velocity_env_cfg.py:278-374`: `track_linear_velocity` +2.0, `track_angular_velocity` +2.0, `upright` +1.0 (kernel, positive), `pose` +1.0, `body_ang_vel` 0 (-0.05 on G1), `angular_momentum` 0 (-0.02 on G1), `dof_pos_limits` -1.0, `action_rate_l2` -0.1, `air_time` 0, `foot_clearance` -2.0, `foot_swing_height` -0.25, `foot_slip` -0.1, `soft_landing` -1e-5. Go1 overrides at `config/go1/env_cfgs.py:186-225` (flat removes the collision terms at :291-292), G1 at `config/g1/env_cfgs.py:147-161` (adds `self_collisions` -1.0). There is **no `is_terminated` term in mjlab**; the -P fall charge in our study comes from our wrapper (`wrapper.py:224`, P = 10 for Go1, 5.7 for G1). Positive weights sum to 6.0, ceiling 0.12/step.

The structural difference worth stating: mjlab has two large near-constant kernels (`upright` ~0.95 and `pose` ~0.8, together ~0.035/step when standing on Go1, ~0.025 on G1, measured above), Unitree has one (`pose`, 0.016-0.020/step) and replaces `upright` by a *penalty* (`body_orientation_l2`).

## 3. Terminations and episodes

| | Unitree Go2-Flat | Unitree G1-Flat | mjlab Go1-Flat | mjlab G1-Flat |
|---|---|---|---|---|
| time-out | 20 s = 1000 steps, truncation (`env_cfg:361`, `:430`) | same | same (`velocity_env_cfg.py:381`, `:455`) | same |
| fall | `fell_over`: base tilt > 70 deg (`env_cfg:362-365`) | same | same (`go1/env_cfgs.py:297-300`) | same |
| contact | `illegal_contact`: any non-foot collision geom touching the ground with force > 10 N (`go2/env_cfgs.py:110-113`) | none | none on flat (`go1/env_cfgs.py:295`) | none |
| terminal reward | `is_terminated` = -200 x dt = **-4.0** on the termination step | same | none | none |
| falls end the episode | yes (reset, `terminated`, value bootstrap cut) | yes | yes | yes |
| time-outs bootstrap | yes (rsl-rl adds `gamma * V` on `extras["time_outs"]`) | yes | yes | yes |

Order matters for the -4: in `ManagerBasedRlEnv.step` terminations are computed *before* rewards (mjlab `manager_based_rl_env.py:452-456`), and `is_terminated` reads `termination_manager.terminated`, so the -4 lands on the same step as the fall. Other episode properties: resampling of commands every 3-8 s, a random push every 5-6 s (`env_cfg:209-223`), 5% zero-command envs (`env_cfg:169`), heading-controlled commands for every env (`rel_heading_envs` defaults to 1.0, `velocity_command.py:255`), and a command curriculum that widens the ranges after `5000 * 24` env steps, i.e. iteration 5000 (`env_cfg:377-386`). Our 2000-2500 iteration runs would stay in stage 0 (x in [-0.5, 1], y in [-0.5, 0.5], yaw in [-1, 1]).

## 4. Per-step reward level (the part that matters)

What matters for the offset question is the near-constant part, not the total. I split each level into: the strictly constant part (an additive term independent of state), the near-constant part (kernels that stay high for any upright robot), and the state-dependent part.

**Definitions used.**
- *Strictly constant*: terms whose value does not depend on state. Unitree: none (analytic, from reading every function in `rewards.py`). mjlab: none.
- *Near-constant (NC)*: the part of the level that a robot earns without learning to track: the `pose` kernel at (or near) the default pose, plus `upright` in mjlab, plus the clock-matching floor of `foot_gait`. This is the closest analogue of an offset.
- *State-dependent (SD)*: the tracking kernels (they sit at their command-dependent floor when standing and climb when tracking), the gait improvement above the floor, and all penalties.

### 4.1 Standing robot

"Standing" = zero action = default joint pose, which is also how the study's standing floor (0.077 Go1, 0.0436 G1) was obtained.

Analytic ideal stand (a robot that holds the default pose perfectly, stage-0 commands), from a Monte Carlo of the command distribution (2M samples: zero out commands with norm <= 0.1, 5% zero-command envs, yaw from the heading controller `clip(0.5 * err, +-1)` with err ~ U(-pi, pi)):

| component | kernel mean | x weight x dt | per step |
|---|---|---|---|
| `track_linear_velocity` | 0.435 (stage 0); 0.174 (stage 1 / initial wide ranges) | x 1.0 x 0.02 | 0.0087 (0.0035) |
| `track_angular_velocity` | 0.458 | x 1.0 x 0.02 | 0.0092 |
| `pose` | 1.0 | x 1.0 x 0.02 | 0.0200 |
| `foot_gait` | 0.56 x P(command>0.1) = 0.532 | x 0.5 x 0.02 | 0.0053 |
| penalties | | | about -0.0009 (taken from the measured pre-fall penalty sum) |
| **total** | | | **~0.042 (stage 0), ~0.037 (stage 1)** |

Of that, near-constant = `pose` 0.0200 + gait floor 0.0053 = **0.025**; state-dependent = tracking floor 0.0179 + penalties -0.0009 = 0.017.

Measured (CPU zero-action rollouts, 300 steps; per-term numbers are mean per step, dt-scaled; "pre-fall" means only the steps before each env's first termination, so the -4 events and the post-reset spawn are excluded):

| task | envs x steps | total incl. terminations | total pre-fall | terminations | median first-fall step |
|---|---|---|---|---|---|
| mjlab Go1-Flat | 64 x 300 | **+0.0791** | n/a (21 terminations in 19,200 env-steps) | 21 | not computed |
| mjlab Go1-Flat | 16 x 200 | +0.0735 | | 1 | |
| mjlab G1-Flat | 32 x 300 | +0.0424 | +0.0417 | 115 | 72 |
| **Unitree Go2-Flat** | 64 x 300 | **-0.0487** | **+0.0334** | 409 | 38 |
| **Unitree G1-Flat** | 32 x 300 | **-0.0459** | **+0.0018** | 121 | 69 |

The mjlab rows reproduce the study's documented standing levels (Go1 0.077, G1 0.0436) to within 3-5%, which is the calibration check for the whole procedure. Per-term measured values for the Unitree rows (pre-fall, per step): Go2: `track_lin` +0.0031, `track_ang` +0.0099, `pose` +0.0164, `foot_gait` +0.0049, all penalties together about -0.0009; G1: `track_lin` +0.0033, `track_ang` +0.0073, `pose` +0.0098, `foot_gait` +0.0050, `angular_momentum` -0.0101, `joint_pos_limits` -0.0042, `body_ang_vel` -0.0039, `body_orientation_l2` -0.0038, others small. The -4 events: Go2 `is_terminated` averages -0.0852/step over the full run (409 terminations / 19,200 env-steps x 4.0), G1 -0.0504.

Two things the measured numbers show:

1. **The kernels match the analytic Monte Carlo** once the initial (wide, stage-1-like) command ranges are used: measured `track_lin` kernel 0.155 vs analytic 0.174, `track_ang` 0.493 vs 0.458, `foot_gait` 0.489/(0.5 x 0.02) vs 0.532. The `pose` kernel is 0.82 measured on Go2 vs 1.0 analytic, because the robot sags at zero action.
2. **Neither Unitree robot stands at zero action in this environment.** Go2 terminates (median step 38) through `illegal_contact`, and I logged which geoms: the rear calves, `RR_calf1_collision` and `RL_calf1_collision`, median contact force 60 N (a diagnostic run of 16 envs x 150 steps). G1 collapses (median step 69), but so does mjlab's own G1 (median step 72), so for the humanoid that is just the zero-action floor of the study's protocol. For Go2 I could not tell whether the rear-calf contact is a real property of the default pose and gains or an artefact of the compatibility shims needed to import Unitree's assets under our mjlab pin (section 9). So the *measured* Go2 number is a falling level; the *analytic* 0.042 is the level of a robot that actually stands.

### 4.2 Well-trained walking robot

There is no trained Unitree policy available offline and I did not train one, so **these are est. (analytic with assumed state statistics), not measured.** The assumption that matters is the tracking quality. I use our own trained mjlab Go1 as the anchor (REPORT_005: tracking fraction ~0.88 at J_task ~0.10, i.e. 83% of mjlab's 0.12 ceiling).

| component | low / mid / high kernel | x weight x dt | per step (low / mid / high) |
|---|---|---|---|
| `track_linear_velocity` | 0.80 / 0.88 / 0.95 | x 0.02 | 0.0160 / 0.0176 / 0.0190 |
| `track_angular_velocity` | 0.80 / 0.90 / 0.95 | x 0.02 | 0.0160 / 0.0180 / 0.0190 |
| `pose` (walking stds; joint RMS deviations 0.05-0.4 rad give `exp(-mean(err^2/std^2))` of 0.6-0.85 for Go2) | 0.60 / 0.70 / 0.85 | x 0.02 | 0.0120 / 0.0140 / 0.0170 |
| `foot_gait` (x 0.95 active share) | 0.76 / 0.86 / 0.92 | x 0.5 x 0.02 | 0.0076 / 0.0086 / 0.0092 |
| penalties (action_rate 0.0005-0.002, joint_acc 0.001-0.002, foot_clearance 0.002-0.004, the rest < 0.001) | | | -0.008 / -0.005 / -0.003 |
| **total** | | | **0.044 / 0.053 / 0.061** |

Near-constant part when walking: `pose` ~0.014 plus the gait kernel's floor 0.0053 = ~0.019; the rest is state-dependent. The pose kernel is lower when walking than when standing (0.014 vs 0.020), the opposite of what an alive bonus would do.

For G1 the same positive budget applies, so trained G1 is **est. 0.04-0.055**; the humanoid-specific penalties (angular momentum, joint limits at weight -10) are the uncertain part, and I did not estimate them tighter than that. For reference, our mjlab G1 centered runs reach J_task 0.088 at tracking 0.79 (REPORT_005 / REPORT_002).

### 4.3 Summary of levels

| quantity (per step) | mjlab Go1 | Unitree Go2 | mjlab G1 | Unitree G1 |
|---|---|---|---|---|
| positive-weight ceiling (analytic) | 0.120 | 0.070 | 0.120 | 0.070 |
| strictly constant part (analytic) | 0 | 0 | 0 | 0 |
| standing, ideal stable stand (analytic) | ~0.077 (measured 0.073-0.079) | ~0.042 | ~0.0436 (measured) | ~0.042 |
| standing, zero-action rollout, pre-fall (measured) | 0.079 (all steps) | 0.033 | 0.042 | **0.002** |
| near-constant part, standing (`upright` + `pose` for mjlab, measured; analytic for Unitree) | ~0.035 | ~0.025 | ~0.025 | ~0.025 |
| trained walking (est. for Unitree, measured for mjlab) | ~0.10 | ~0.053 (0.044-0.061) | ~0.088 | ~0.045 (0.04-0.055) |
| near-constant part, walking (est.) | ~0.03 | ~0.019 | | ~0.019 |
| state-dependent part, walking (est.) | ~0.07 | ~0.034 | | ~0.026 |

## 5. Comparison to mjlab and the "equivalent offset c"

Our dose-response adds a constant c to a reward whose own level is 0.077 (standing) to 0.10 (trained). So the right comparison is Unitree's level *minus* mjlab's level, in the same state. That difference is the extra constant Unitree would be carrying relative to the baseline that works at c = 0.

| robot, state | Unitree level | mjlab level | ratio | equivalent offset c = Unitree - mjlab | provenance |
|---|---|---|---|---|---|
| Go2 vs Go1, ideal stand | 0.042 | 0.077 | **0.55** | **-0.035** | analytic vs measured/doc |
| Go2 vs Go1, zero-action pre-fall | 0.033 | 0.079 | 0.42 | -0.046 | measured vs measured |
| Go2 vs Go1, trained | 0.053 (0.044-0.061) | 0.10 | **0.53** (0.44-0.61) | **-0.047** (-0.056 to -0.039) | est. vs measured |
| near-constant part only, standing (Go2 vs Go1) | ~0.025 | ~0.035 | 0.71 | -0.010 | analytic vs measured |
| G1 vs G1, ideal stand | 0.042 | 0.0436 | 0.96 | -0.002 | analytic vs measured |
| G1 vs G1, zero-action pre-fall | 0.002 | 0.042 | 0.04 | -0.040 | measured vs measured |
| G1 vs G1, trained | ~0.045 (0.04-0.055) | 0.088 | ~0.51 | ~-0.04 | est. vs measured |

**Worst case, stated explicitly.** Suppose one counted *every* positive weight of Unitree as if it were constant: that is 0.070/step, still below the 0.10 of our first harmful row, and mjlab's own reward already contains 0.12 of the same kind of kernels while walking at c = 0 (19/20 uncentered). On no definition does Unitree carry an extra +0.1.

**Mapping onto REPORT_005's table** (Go1, gamma .999, 20 seeds each):

| offset c | uncentered walks | centered walks | Unitree falls here? |
|---|---|---|---|
| 0 | 19/20 | 20/20 | **yes: every c_eq above is <= 0, i.e. at or below this row** |
| 0.1 | 12/20 | 20/20 | no |
| 0.3 | 2/20 | 18/20 | no |
| 1.0 | 0/20 | 19/20 | no |

The measured mjlab Go1 rows above (c = 0, standing level 0.077) are the row the table calls "0"; Unitree's level is about half of that, so on this axis it sits at or below the benign row.

## 6. PPO settings

| setting | Unitree Go2 / G1 (and all other velocity tasks) | mjlab Go1 / G1 (reference) | ours (study) |
|---|---|---|---|
| discount gamma | **0.99** (`go2/rl_cfg.py:37`, `g1/rl_cfg.py:37`) | 0.99 (`go1/rl_cfg.py:37`) | **0.999** main cells, 0.99 for G1 and some Go1 cells |
| GAE lambda | 0.95 (:38) | 0.95 | 0.95 (inherits) |
| rollout length | 24 steps/env (:44) | 24 | 24 |
| number of envs | README: `--env.scene.num-envs=4096`; the config default is 1 (`velocity_env_cfg.py:399`) | 4096 (mjlab README examples; config default is 1, `velocity_env_cfg.py:426`) | 4096 default, G1 pilots 2048 |
| minibatches x epochs | 4 x 5 (:33-34) | same | same |
| learning rate | 1e-3, adaptive KL schedule, desired_kl 0.01 (:35-36, :39) | same | same |
| clip_param / entropy / grad-norm | 0.2 / 0.01 / 1.0 (:31-32, :40) | 0.2 / 0.01 / 1.0 | same |
| **value-loss clipping** | **on** (`use_clipped_value_loss=True`, :30) | on | on by default; the study varies it (clip-off arms) |
| **reward normalization** | **none** | none | none (the study adds `reward_scale="std"` as a rival arm) |
| **return / value normalization** | **none** | none | none (rival arm `ret_norm="meanstd"`) |
| observation normalization | actor and critic `obs_normalization=True` (:16, :26) | same | same |
| advantage normalization | over the whole batch (`normalize_advantage_per_mini_batch=False`, a rsl-rl default; mjlab `rl/config.py:71`) | same | same |
| network / init std | 512-256-128 ELU, scalar std, init 1.0 (:14-22) | same | same |
| max iterations | 10001 (Go2, G1) | 10,000 (Go1), 30,000 (G1) | 2000-3000 |

Normalization check: I grepped both rsl-rl 5.0.1 and 5.5.1 for any extrinsic reward, return or value normalization; the only `reward_normalization` is the optional RND intrinsic-reward extension, which is off. **The "normalization might reduce the offset problem" caveat therefore does not apply: Unitree uses the same un-normalized, value-clipped PPO as mjlab and as our study's default.** Only observations are normalized.

## 7. Versions

| package | Unitree pins | ours | note |
|---|---|---|---|
| mjlab | `mjlab==1.2.0` (`setup.py:5`), commit `5af32e37` | `8bb2bc9e` (pyproject 1.6.0, dev) | 4 minor releases apart, plus unreleased commits |
| mujoco-warp | `==3.5.0` (`setup.py:6`) | `~=3.16.0` (local venv: 3.16.0) | |
| mujoco | via mjlab 1.2.0: `>=3.5.0` | `~=3.16.0` (3.16.0) | |
| rsl-rl-lib | not pinned directly; via mjlab 1.2.0: `==5.0.1` | `5.5.1` (the commit's pin; PyPI mjlab 1.6.0 would pull 5.4.2) | |
| torch | via mjlab 1.2.0: `>=2.7.0`, cu128 index | `>=2.14.0`; pods run 2.15+cu132 or 2.14.1+cu130; local venv 2.15.0+cu132 | |
| warp-lang | `>=1.12.0` | `>=1.15.0` (local 1.18.0) | |
| python | docs recommend 3.11; mjlab 1.2.0 needs `<3.14` | `>=3.12,<3.14` | 3.11 is out of our range |

mjlab 1.2.0 to our commit changes the pieces the Unitree package imports. Two breaking changes were found by running the import (section 9): `mjlab.utils.os.update_assets` was removed upstream (commit `9a8182988`, "Remove redundant spec.assets loading", #873), and `CollisionCfg` now **requires** `contype`, `conaffinity`, `condim` and `priority`, with new semantics (`disable_other_geoms=True` by default, dict-valued fields must cover every matched geom). The first breaks the import of every Unitree robot module; the second breaks G1 at import (`TypeError: CollisionCfg.__init__() missing 2 required positional arguments`) and silently changes collision behaviour for Go2.

## 8. Verdict

**Do Unitree's configs sit in the harmful range? No.**

- There is no constant or alive-style term (analytic). The positive part is four kernels with a ceiling of 0.070/step, 58% of mjlab's 0.120.
- The near-constant part is about 0.025/step standing and about 0.019 walking (analytic/est.), against ~0.035 for mjlab Go1 (~0.025 for mjlab G1). Equivalent offset c vs mjlab: **about -0.04, not +0.1.** Even if all positive weights were counted as constant (0.070), that is below our first harmful row and mjlab's own baseline already carries more.
- The big weights are on state-dependent terms: `is_terminated` -200 (-4 on a fall step) and the humanoid penalties at -10 (`joint_pos_limits`) and -1 (`body_orientation_l2`). These are large but vary with the state, so a critic can learn them; they are not an offset in the sense of REPORT_005.
- PPO has no normalization and uses value clipping, so there is no hidden mitigation: nothing in Unitree's pipeline would mask an offset if it were there, which makes the absence of one more convincing.

**Caveats (why the answer is "no" and not "no, with certainty").**

1. **Gamma.** Unitree trains at gamma 0.99; the dose-response table was measured at 0.999 on Go1. Our G1 result at gamma .99 (offset +1, uncentered 0/6) and the legacy Go1 gamma .99 runs (3 seeds) do not show a gamma dependence, but the c = 0.1/0.3 rows were never run at .99. The threshold at .99 could differ. It does not change the verdict because Unitree's c_eq is below 0 either way.
2. **Negative level early in training.** With `is_terminated` at -4 per fall, a policy that falls every ~40 steps has a *negative* mean reward (measured zero-action: Go2 -0.049, G1 -0.046; random initial policies fall at least as fast). Our dose-response only tested *positive* offsets. The fall term varies with state (it is a function of falling), so it is not a constant offset, but the sign and size of the early level is outside what REPORT_005 covers. In our continuing wrapper this term maps onto the existing fall charge: -4.0 is about 95x Go2's standing per-step reward (0.042), the same order as our P = 130x rule (Go1 10, G1 5.7).
3. **Reward shape.** The "c = 0 benign" mapping assumes Unitree's reward behaves like mjlab's at c = 0. That is untested: Unitree uses a gait clock, a tighter `pose` kernel, and a heavy termination term.

**Should we run the Unitree-Go2-Flat pilot (uncentered vs centered)? Not now.** The prediction from the audit is a null result (centered ~ uncentered, like c = 0, 20 seeds, equivalence margin met), so the pilot would most likely produce a negative control rather than a new effect, and it costs roughly 1-1.5 engineer-days plus a fidelity risk (below) while the study's open items (G1 claim set, SSP) compete for the same budget. It becomes worth doing if a reader asks for "does centering matter on a shipped Unitree config", in which case a 3-seed-per-arm pilot is the right size. If we run it, the informative cell is gamma 0.99 (Unitree's own), uncentered vs centered, because that is also the open caveat 1.

## 9. What was run, and how hard a pilot would be

### 9.1 Method for the measured numbers

Scripts: `/tmp/unitree_audit/{rollout.py,rollout_pre.py,diag3.py,mc.py}` (scratch; the essential logic is below). The existing local venv (`mjlab_centering/.venv`, mjlab `8bb2bc9`, torch 2.15, mujoco-warp 3.16) was used read-only, `CUDA_VISIBLE_DEVICES=""`, device `cpu`. Zero action every step. Unitree's task code was imported from the clone through `PYTHONPATH`, with these **in-process shims (nothing installed or edited)**:

1. `mjlab.utils.os.update_assets` re-injected with the mjlab v1.2.0 body (a 7-line directory glob into a dict).
2. `mjlab.utils.spec_config.CollisionCfg` wrapped to restore the v1.2.0 defaults (`contype=1, conaffinity=1, condim=3, priority=0`, add a catch-all `".*"` to dict-valued fields, `disable_other_geoms=False`).
3. `src.tasks` replaced by a stub package so the import does not walk the motion-tracking tasks.

With those the Go2 and G1 flat envs build and step. The mjlab-own rollouts need no shims and reproduce the study's standing levels. The Unitree rollouts therefore carry shim risk, which is why I give the analytic standing level as the primary number and the measured one as a cross-check.

### 9.2 Porting effort for a Go2 pilot with our continuing wrapper and centering patch

**Recommended route (A): vendor Unitree's `src/` onto our pinned stack.** Estimate **1-1.5 engineer-days, GPU cost about $1-3 for a 3-5 seed x 2 arm x 2500 iteration pilot** (25k iterations at our measured $0.063 per 1000; Go2 may be 1.2-1.5x Go1's cost because of the extra contact sensor history, which I did not measure).

| step | work | est. |
|---|---|---|
| 1 | vendor `src/assets` and `src/tasks/velocity` into a package under `mjlab_centering` (no edits to our code, new module only); port the two API breaks properly rather than with shims, then **verify Go2 stands at zero action** (the rear-calf contact found above must be explained: shim artefact or real) | 3-5 h |
| 2 | `env_cfg.build_env_cfg` / `build_agent_cfg`: add a `go2` branch (two ternaries today choose Go1 or G1); the agent config is identical to Go1's | 0.5 h |
| 3 | wrapper: set `penalty=0`, because the env already adds -4 on the fall step via `is_terminated`; otherwise the fall is charged twice. `fell` = `terminated` already covers both `fell_over` and `illegal_contact` (`fall_check_deg=None` path, `wrapper.py:211-214`) | 0.5 h |
| 4 | recalibrate the "walks" criterion: ours is `track_lin_frac >= 0.5` (floor 0.28 on Go1). Unitree's stage-0 floor is 0.435 analytic (0.17 once commands widen), so 0.5 no longer separates standing from walking; measure the Go2 standing floor and pre-register a new line | 2-3 h |
| 5 | unit tests (reward-term names present, wrapper reward = task + offset with no double penalty, fall reset-in-place on Go2) and a 20-iteration smoke on the pod | 2-3 h |

The centering patch itself (`CenteredPPO`) needs no change under route A because it runs on rsl-rl 5.5.1, same as now.

**Faithful route (B): rebuild Unitree's own pins** (mjlab 1.2.0, rsl-rl 5.0.1, torch 2.7 cu128, mujoco-warp 3.5.0). Estimate 2.5-3.5 days: a second Docker image and bootstrap, a port of our wrapper to the 1.2.0 `RslRlVecEnvWrapper`, and a port of `CenteredPPO` to rsl-rl 5.0.1, whose hooks differ (per-step `update_normalization` inside `process_env_step`, a different `__init__` and `update`; the hook names `process_env_step`, `compute_returns`, `update` and `construct_algorithm` do exist). I would not choose B for a first pilot: it buys fidelity to a version pair that is a year behind our whole campaign and changes everything at once.

**Blockers and risks, in order of severity**

1. **Collision semantics across versions** (section 7): the shimmed Go2 does not hold its default pose, so we cannot yet say whether route A reproduces Unitree's physics. Needs the step-1 check before any GPU time.
2. **`is_terminated` in the task reward**: `J_task` and the existing eval code include the -4 spikes, so Go2 J_task is not comparable with Go1's and the double-penalty trap above must be avoided.
3. **Walk threshold** (step 4).
4. **Version skew** in physics (mujoco-warp 3.5 vs 3.16): even after porting, results are "Unitree's reward on our stack", not "Unitree's training". For an offset audit that is acceptable, since the reward functions are the object under study.
5. **Custom runner**: `VelocityOnPolicyRunner` (wandb + ONNX export) is irrelevant because we use our own training loop. No blocker.
6. **CUDA/torch**: no new constraint under route A; the pods already run torch 2.14.1+cu130 / 2.15+cu132 with this mjlab. Under route B torch 2.7+cu128 would need its own image and warp cache.
7. **Python**: Unitree recommends 3.11; our package requires >=3.12. Irrelevant for route A.

## 10. Not determined (not guessed)

- **Trained-walker levels for Unitree are estimates.** No trained Unitree policy was available and none was trained. The kernel values at trained quality come from our mjlab Go1 as an anchor; the true Unitree walkers could sit a few percent above or below the 0.044-0.061 band. The qualitative result (ceiling 0.070, so below mjlab's level) does not depend on this.
- **Whether Go2 stands at zero action in Unitree's own stack.** In the shimmed environment it does not (rear calves hit the ground, median 38 steps to a termination). I did not isolate shim artefact from real behaviour.
- **Standing-floor and trained levels for the other Unitree tasks** (A2, As2, H1_2, H2, R1, G1-23Dof): reward weights are shared with Go2/G1 by reading, so the ceiling (0.070) is the same, but I measured nothing on them.
- **The dose-response at gamma 0.99** on Go1 (c = 0.1, 0.3): not in our data.
- **Negative offsets**: not tested anywhere in the study; the early-training Unitree level is negative because of the fall term.
- **Per-term trained-walker penalties for the humanoid** (angular momentum, joint limits at -10): wide uncertainty, which is why G1's trained estimate is a 0.04-0.055 band.
- **Cost per iteration on Go2**: not measured; the $1-3 pilot figure assumes Go1's $0.063/1000 iterations times 1.0-1.5.
