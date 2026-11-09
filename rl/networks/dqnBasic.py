import math

import numpy as np
import torch
import torch.nn as nn

# Every DQN step touches torch for a handful of TINY tensor ops (a few thousand
# to tens of thousands of floats — get_q_values, update()'s matmuls/gathers).
# torch's default CPU intra-op thread pool is sized off the reported core
# count, and spinning up/tearing down that pool per call dominates when each
# op is only microseconds of real work — especially in containers, where
# os.cpu_count()/torch's default can reflect the HOST's full core count
# rather than the container's actual cgroup-limited vCPU allocation (a known
# Docker/cgroups gap on some cloud providers), causing far more threads than
# useful. Single-threaded execution is the standard fix for small-tensor CPU
# RL workloads. Set at import time (must happen before any parallel torch op
# runs in the process, per torch's own requirement for set_num_interop_threads).
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Torch forbids changing the inter-op pool after parallel work has started.
    # In mixed test/import processes this may have happened before importing DQN;
    # training workers import DQN in a fresh process and do set it successfully.
    pass


class ReplayBuffer:
    """CPU NumPy ring buffer. Stays on CPU; torch.from_numpy() at the boundary.

    When `sparse_k` is given, `obs`/`next_obs` store int64 active-tile-index
    arrays of length `sparse_k` instead of dense float32 `obs_dim`-length
    vectors (used for TileCoding — see DQN.__init__ below).
    """

    def __init__(self, capacity: int, obs_dim: int, sparse_k: int = None):
        self.capacity = capacity
        self.pos = 0
        self.full = False
        self.sparse = sparse_k is not None
        if self.sparse:
            self.obs = np.zeros((capacity, sparse_k), dtype=np.int64)
            self.next_obs = np.zeros((capacity, sparse_k), dtype=np.int64)
        else:
            self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
            self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

    def push(self, obs, action: int, reward: float, next_obs, done: bool):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = bool(done)
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


class QNetwork(nn.Module):
    """Q(s,a) network. hidden_dims=None/[] recovers the plain linear case
    (Q = w·phi(s), a single Linear layer); hidden_dims=[64,64] gives the
    "deep" variant (2 hidden layers + linear output, ReLU between layers,
    no activation on the output — Q-values are real-valued, not a
    distribution, so no softmax/sigmoid here).

    Sparse mode (rep.n_active set, e.g. TileCoding/OneHot representations)
    replaces the FIRST layer with an nn.EmbeddingBag(sparse=True) instead of
    an nn.Linear — see forward_sparse. A plain nn.Linear's backward pass
    densifies the gradient over the WHOLE (hidden_dim, obs_dim) weight
    matrix even though forward_sparse only ever reads n_active columns, and
    Adam.step() then walks every element of that dense gradient/moment
    buffer every update regardless of how few were actually touched.
    EmbeddingBag(sparse=True) keeps the gradient as an actual sparse tensor
    (only the touched rows), and DQN pairs it with torch.optim.SparseAdam so
    the optimizer step also only touches those rows. Benchmarked: 26-132x
    faster per update at obs_dim in [50625, 279841] with hidden_dims=[64,64]
    (obs_dim can reach 131072 for TileCoding, 279841 for one-hot at
    n_bins=23). Every layer after the first operates on hidden_dim, not
    obs_dim, and stays a normal dense nn.Linear on the regular Adam.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims=None, sparse: bool = False):
        super().__init__()
        dims = [obs_dim] + list(hidden_dims or []) + [n_actions]
        self.sparse = sparse

        if sparse:
            self.emb = nn.EmbeddingBag(obs_dim, dims[1], mode="sum", sparse=True)
            self.emb_bias = nn.Parameter(torch.zeros(dims[1]))
            with torch.no_grad():
                # Match nn.Linear(obs_dim, dims[1])'s default init exactly
                # (kaiming_uniform_, fan_in=obs_dim). EmbeddingBag's own
                # default init (normal_, std=1) targets NLP/RecSys pooling
                # use-cases, not this one — using it would make this a
                # behavior change instead of a pure performance fix.
                w = torch.empty(dims[1], obs_dim)
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                self.emb.weight.copy_(w.T)
                bound = 1 / math.sqrt(obs_dim)
                nn.init.uniform_(self.emb_bias, -bound, bound)
            tail_dims = dims[1:]
        else:
            tail_dims = dims

        # sparse=True: layers AFTER the embedding (empty for "linear", i.e.
        # hidden_dims=None). sparse=False: every layer, unchanged.
        self.layers = nn.ModuleList(
            nn.Linear(tail_dims[i], tail_dims[i + 1]) for i in range(len(tail_dims) - 1)
        )

    def forward_dense(self, phi: torch.Tensor) -> torch.Tensor:
        x = phi
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

    def forward_sparse(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (n_active,) int64 for a single state's active indices, or
        (batch, n_active) for a batch of them. Equivalent to forward_dense on
        the corresponding one-hot/multi-hot vector, without materializing it.
        """
        single = idx.dim() == 1
        idx2d = idx.unsqueeze(0) if single else idx
        x = self.emb(idx2d) + self.emb_bias
        for layer in self.layers:
            x = torch.relu(x)
            x = layer(x)
        return x.squeeze(0) if single else x


class DQN:
    """DQN with experience replay and a periodic target network. Supports
    both the plain linear architecture (Q = w·phi(s), hidden_dims=None) and
    a deep MLP (hidden_dims=[64,64], say) via QNetwork above.

    Encoded phi vectors are stored in the replay buffer (rep.size_out floats,
    or rep.n_active indices in sparse mode) to avoid re-encoding on every
    update call.

    Trained via standard autograd (loss.backward()) + torch.optim.Adam over
    all of QNetwork's parameters — NOT the hand-rolled per-action Adam this
    class used before hidden layers were introduced. That per-action design
    (separate step-counter/gradient-clip per action, frozen exactly when an
    action was absent from a batch) only made sense when every action had
    its own independent weight ROW in a single (n_actions, obs_dim) matrix;
    it doesn't extend to shared hidden-layer parameters, and per-parameter
    "exact freezing" was never something the underlying paper's update rule
    (Q += α·δ·∇Q — plain semi-gradient descent, no optimizer specified)
    actually required — it was an implementation nicety layered on top of an
    Adam-based training loop, and not one used by mainstream DQN
    implementations (which just backward()+step() the whole network every
    update). Unifying both linear and deep variants on the standard
    optimizer keeps "linear vs deep" a clean architecture-only comparison
    instead of also differing in optimizer mechanics. See runs/RUN_001.md
    for the full before/after discussion.
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
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        max_grad_norm: float = 10.0,
        hidden_dims=None,
        init_seed: int = 0,
    ):
        self.rep = rep
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.reward_center_mode = reward_center_mode
        self.reward_center_beta = reward_center_beta
        self.reward_center_eta = reward_center_eta
        self.avg_reward = float(reward_center_init)
        self.max_grad_norm = max_grad_norm

        obs_dim = rep.size_out
        self.sparse_k = getattr(rep, "n_active", None)
        self.sparse = self.sparse_k is not None

        # init_seed=0 reproduces the original fixed-weights behavior exactly (every search trial
        # was run this way). Pass the run's seed to get seed-dependent initial weights instead.
        torch.manual_seed(init_seed)
        self.q_net = QNetwork(obs_dim, n_actions, hidden_dims, sparse=self.sparse)
        self.q_target = QNetwork(obs_dim, n_actions, hidden_dims, sparse=self.sparse)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.requires_grad_(False)

        if self.sparse:
            # SparseAdam only accepts sparse gradients, and Adam only accepts
            # dense ones — the embedding's weight is the one parameter with a
            # sparse .grad (see QNetwork docstring), so it gets its own
            # optimizer; everything else (its bias, every later layer) trains
            # on the regular Adam exactly as before.
            self.optimizer = torch.optim.Adam(
                list(self.q_net.layers.parameters()) + [self.q_net.emb_bias],
                lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_eps,
            )
            self.optimizer_sparse = torch.optim.SparseAdam(
                self.q_net.emb.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_eps,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.q_net.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_eps,
            )
            self.optimizer_sparse = None

        self.buffer = ReplayBuffer(buffer_size, obs_dim, sparse_k=self.sparse_k)
        self._update_count = 0

    @property
    def lr(self) -> float:
        """Reads/writes the optimizer's actual learning rate (not a separate
        stored value) so that `dqn.lr = x` after construction — used e.g. by
        tests/test_dqn_update.py to rescale lr for a different phi norm, and
        by the "value" reward-centering formula — really changes the step
        size Adam applies, rather than silently going stale."""
        return self.optimizer.param_groups[0]["lr"]

    @lr.setter
    def lr(self, value: float):
        self.optimizer.param_groups[0]["lr"] = value
        if self.optimizer_sparse is not None:
            self.optimizer_sparse.param_groups[0]["lr"] = value

    def get_q_values(self, phi) -> torch.Tensor:
        """Return Q-values for a single phi. Accepts numpy or torch phi.

        In sparse mode phi is the active-index array (see QNetwork docstring);
        Q(s,a) = sum of w[a, idx] over active indices, exactly equal to w[a] @
        one_hot(idx) but without materializing the one-hot vector.
        """
        with torch.no_grad():
            if self.sparse:
                idx = torch.as_tensor(phi, dtype=torch.int64)
                return self.q_net.forward_sparse(idx)
            if not isinstance(phi, torch.Tensor):
                phi = torch.as_tensor(phi, dtype=torch.float32)
            return self.q_net.forward_dense(phi)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        # Convert NumPy buffer samples to torch tensors at the boundary.
        # torch.from_numpy on contiguous float32 is metadata-only (~us).
        obs_b_np, act_b_np, rew_b_np, next_obs_b_np, done_b_np = self.buffer.sample(self.batch_size)
        obs_b = torch.from_numpy(obs_b_np)
        next_obs_b = torch.from_numpy(next_obs_b_np)
        act_b = torch.from_numpy(act_b_np)
        rew_b = torch.from_numpy(rew_b_np)
        done_b = torch.from_numpy(done_b_np)

        # Reward centering
        if self.reward_center_mode == "simple":
            centered_rew = rew_b - self.avg_reward
            self.avg_reward += self.reward_center_beta * (float(rew_b.mean()) - self.avg_reward)
        elif self.reward_center_mode == "value":
            centered_rew = rew_b - self.avg_reward
        else:
            centered_rew = rew_b

        forward = self.q_net.forward_sparse if self.sparse else self.q_net.forward_dense
        forward_target = self.q_target.forward_sparse if self.sparse else self.q_target.forward_dense

        with torch.no_grad():
            next_q = forward_target(next_obs_b)             # (batch, n_actions)
            max_next_q = next_q.max(dim=1).values            # (batch,)
            targets = centered_rew + self.gamma * max_next_q * (~done_b)  # (batch,)

        q_values = forward(obs_b)                             # (batch, n_actions), grad-tracked
        current_q = q_values.gather(1, act_b.unsqueeze(1)).squeeze(1)
        td_errors = targets - current_q

        # MSE loss == 0.5*delta^2, matching the paper's semi-gradient TD update
        # (Q += α·δ·∇Q is exactly gradient descent on this loss).
        loss = 0.5 * (td_errors ** 2).mean()
        self.optimizer.zero_grad()
        if self.optimizer_sparse is not None:
            self.optimizer_sparse.zero_grad()
        loss.backward()
        # Global norm clip across all parameters (standard practice) — the old
        # per-action clip doesn't have a meaningful analogue once parameters
        # are shared across actions (hidden layers, or even just wanting a
        # single optimizer call over the whole network). Works fine with a
        # mix of dense and sparse .grad (verified directly).
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.optimizer_sparse is not None:
            self.optimizer_sparse.step()

        # Value-centering avg_reward update (uses mean TD error as a proxy for value gradient)
        if self.reward_center_mode == "value":
            self.avg_reward += self.reward_center_eta * self.lr * float(td_errors.mean().item())

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())
