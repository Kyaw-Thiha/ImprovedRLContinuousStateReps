import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.pos = 0
        self.full = False
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

    def push(self, obs, action: int, reward: float, next_obs, done: bool):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int):
        max_idx = self.capacity if self.full else self.pos
        idx = np.random.randint(0, max_idx, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos


class DQN:
    """Linear DQN with experience replay and a periodic target network.

    Q(s, a) = w[a] · φ(s)

    Raw states are stored in the replay buffer (state_dim floats); feature
    vectors φ(s) are recomputed from rep during update() so the buffer size
    is independent of representation dimensionality.
    Weight updates use the same L2-norm scaling as the A2C TD(0) rule so that
    learning rates are comparable across representations.
    """

    def __init__(
        self,
        rep,
        state_dim: int,
        n_actions: int,
        lr: float,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        target_update_freq: int,
        reward_center_mode: str = "none",
        reward_center_beta: float = 0.001,
        reward_center_eta: float = 1.0,
        reward_center_init: float = 0.0,
    ):
        self.rep = rep
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.reward_center_mode = reward_center_mode
        self.reward_center_beta = reward_center_beta
        self.reward_center_eta = reward_center_eta
        self.avg_reward = float(reward_center_init)

        obs_dim = rep.size_out
        self.w = np.zeros((n_actions, obs_dim))
        self.w_target = np.zeros((n_actions, obs_dim))
        self.buffer = ReplayBuffer(buffer_size, state_dim)
        self._update_count = 0

    def get_q_values(self, phi: np.ndarray) -> np.ndarray:
        return self.w @ phi

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states_b, act_b, rew_b, next_states_b, done_b = self.buffer.sample(self.batch_size)

        obs_b = np.stack([self.rep.map(s) for s in states_b])
        next_obs_b = np.stack([self.rep.map(s) for s in next_states_b])

        # Reward centering
        if self.reward_center_mode == "simple":
            centered_rew = rew_b - self.avg_reward
            self.avg_reward += self.reward_center_beta * (float(np.mean(rew_b)) - self.avg_reward)
        elif self.reward_center_mode == "value":
            centered_rew = rew_b - self.avg_reward
        else:
            centered_rew = rew_b

        # TD targets using the frozen target network
        next_q = next_obs_b @ self.w_target.T  # (batch, n_actions)
        max_next_q = np.max(next_q, axis=1)    # (batch,)
        targets = centered_rew + self.gamma * max_next_q * (~done_b)

        # Q-values for the actions that were taken
        current_q = np.sum(self.w[act_b] * obs_b, axis=1)  # (batch,)
        td_errors = targets - current_q

        # Weight update grouped by action — same L2-norm scaling as A2C TD(0)
        for a in range(self.n_actions):
            mask = act_b == a
            if not np.any(mask):
                continue
            phi_a = obs_b[mask]        # (n_a, dim)
            delta_a = td_errors[mask]  # (n_a,)
            norms = np.sum(phi_a ** 2, axis=1)
            scales = np.where(norms > 0, 1.0 / norms, 0.0)
            self.w[a] += self.lr * np.mean((delta_a * scales)[:, None] * phi_a, axis=0)

        # Value-centering avg_reward update (uses mean TD error as a proxy for value gradient)
        if self.reward_center_mode == "value":
            self.avg_reward += self.reward_center_eta * self.lr * float(np.mean(td_errors))

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.update_target()

    def update_target(self):
        np.copyto(self.w_target, self.w)
