# Algorithms and Centering Implementation

## DQN (linear and deep)
- Direct implementation of the original reward-centering paper's Q-learning equations (9–10):
  `δ = R - R̄ + γ·max_a Q(s',a) - Q(s,a)`
  `Q += α·δ·∇Q`
  `R̄ += η·α·δ`
- This is the setting the original paper proves convergence for — most theoretically solid case, good as the core validation.
- "Linear" = single weight matrix, no hidden layer (`Q = W·φ(s)`). "Deep" = 2 hidden layers + output.

## A2C
- Legacy repo already has value-based centering logic for TD(0) in `network/rlnet/rules/td0centering.py` (Nengo-node implementation) — use as the starting point/reference, re-expressed as a batched PyTorch update, not Nengo.
- Should be re-framed using Bellman Error Centering's formulation (centering the TD-error / Bellman error) rather than the original paper's raw equations, per `01-theory-and-background.md`.

## PPO (RSL-RL)
- See `03-architecture.md` for the exact `compute_returns()` hook. Implementation sketch:

```python
class CenteredPPO(PPO):
    def __init__(self, *args, eta=1/64, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_bar = 0.0
        self.eta = eta

    def compute_returns(self, obs):
        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            centered_reward = st.rewards[step] - self.r_bar
            delta = centered_reward + next_is_not_terminal * self.gamma * next_values - st.values[step]
            self.r_bar += self.eta * self.learning_rate * delta.mean().item()  # value-based update
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)
```

(Sketch — verify against the exact RSL-RL version pinned in `pyproject.toml` before use; confirm `self.learning_rate` is the right step-size reference at the point `compute_returns` runs, since PPO's adaptive-KL schedule can change it mid-training.)

## SAC — deferred / optional
- Not in RSL-RL's actively maintained branch (only the unmaintained `algorithms` branch has SAC/DDPG/DSAC).
- Relevant precedent: Average-Reward Soft Actor-Critic (Adamczyk et al. 2025) — centers via a different mechanism (subtracting `Q(0,0)` from targets, adaptive reset-cost penalty for episodic tasks) than this project's TD-error-based `R̄` tracking. Related, not a drop-in implementation. Their own reported compute: ~3000 GPU-hours across their full multi-baseline, multi-environment study — a signal this algorithm family is not cheap, not a per-config estimate.
- If added: most naturally via Stable-Baselines3's SAC wrapped around mjlab's env directly, at Stage 2 only, not Stage 3.

## Plasticity monitoring (not a swept axis)
- Continuing tasks run for very long horizons (esp. Stage 3) — exactly where loss-of-plasticity effects (Alberta/Amii lineage) are known to occur.
- Not treated as a third experimental axis (each mitigation technique adds its own hyperparameters, too much scope). Instead: log a plasticity indicator (effective feature rank, or weight/gradient norm trends) during Stage 3 runs as a diagnostic. If it shows up, that's a finding to report or follow up on, not something being actively mitigated in this phase.
