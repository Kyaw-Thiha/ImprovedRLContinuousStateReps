import itertools
import os
import sys
import time
import warnings

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.networks.dqnBasic import DQN
from rl.representations.normal import NormalRep
from rl.representations.ssp import SSPRep
from rl.representations.tilecoding import TileCodingRep
from rl.representations.onehot import OneHotRepCP
from base_trial import BaseTrial

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def _run_eval_episodes(dqn, rep, encode, eval_env, obs_dim, n_episodes, max_steps,
                        normalize_state, state_scale, first_reset_seed=None, return_list=False):
    """Deterministic (no epsilon) rollouts on a held-out env — rl_zoo3's
    TrialEvalCallback pattern (see araffin.github.io/post/optuna/). Kept
    separate from the noisy training-reward trace so Optuna's pruner judges
    trials on the same signal rl-baselines3-zoo does, not on-policy exploration
    noise.

    first_reset_seed: seeds ONLY the first reset of this call (the env's stream then
    continues); None keeps the legacy unseeded behaviour. return_list=True returns
    (mean, per-episode returns) instead of just the mean (eval_v3 final-eval records).
    """
    returns = []
    for ep in range(n_episodes):
        if ep == 0 and first_reset_seed is not None:
            raw_obs = eval_env.reset(seed=int(first_reset_seed))[0]
        else:
            raw_obs = eval_env.reset()[0]
        state = rep.get_state(raw_obs[:obs_dim], eval_env)
        if normalize_state:
            state = state / state_scale
        phi = encode(state)
        ep_reward = 0.0
        for _ in range(max_steps):
            action = int(dqn.get_q_values(phi).argmax())
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
            state = rep.get_state(obs[:obs_dim], eval_env)
            if normalize_state:
                state = state / state_scale
            phi = encode(state)
        returns.append(ep_reward)
    if return_list:
        return float(np.mean(returns)), [float(r) for r in returns]
    return float(np.mean(returns))


def _probe_stats(dqn, rep, encode, env, probe_states, obs_dim, normalize_state, state_scale):
    """Mean max-Q and mean |Q(s,0)-Q(s,1)| over fixed probe states (no gradient, no RNG).

    Tile coding allocates IHT entries for tiles it has not seen; that would change which index
    later states get and so perturb training. The IHT is therefore rolled back to its pre-probe
    state (entries are only ever appended, so popping the newest ones restores it exactly).
    """
    import torch
    iht = getattr(rep, "iht", None)
    if iht is not None:
        n0, over0 = len(iht.dictionary), iht.overfullCount
    phis = []
    for s in probe_states:
        state = rep.get_state(s[:obs_dim], env)
        if normalize_state:
            state = state / state_scale
        phis.append(np.asarray(encode(state)))
    if iht is not None:
        while len(iht.dictionary) > n0:
            iht.dictionary.popitem()
        iht.overfullCount = over0
    batch = np.stack(phis)
    with torch.no_grad():
        if dqn.sparse:
            q = dqn.q_net.forward_sparse(torch.as_tensor(batch, dtype=torch.int64))
        else:
            q = dqn.q_net.forward_dense(torch.as_tensor(batch, dtype=torch.float32))
    q = q.numpy()
    return float(q.max(axis=1).mean()), float(np.abs(q[:, 0] - q[:, 1]).mean())


class DQNTrial(BaseTrial):
    def params(self):
        super().params()

        ## Representation
        self.param("Method for representing the state", rep_="Normal")
        self.param("Discretization of the representation", n_bins=100)
        self.param("Number of tilings for tile coding", num_tilings=8)
        self.param("Tiles per dimension for tile coding", tiles_per_dim=None)
        self.param("IHT size for tile coding", iht_size=4096)
        self.param("State indices for tile coding", tile_state_indices=None)
        self.param("Normalize state", normalize_state=False)
        self.param("Length scale", length_scale=1.0)
        self.param("Number of rotations", n_rotates=5)
        self.param("Number of scales (HexSSP)", n_scales=5)

        ## DQN Parameters
        self.param(
            "Network architecture: 'linear' (Q=w.phi(s), no hidden layers) or "
            "'deep' (2 hidden layers of 64 units, ReLU, ~SB3's DQN default net_arch=[64,64])",
            architecture="linear",
        )
        self.param("Epsilon for epsilon-greedy", eps=0.1)
        self.param("Dynamic Epsilon", dynamic_epsilon=False)
        self.param("Learning rate", lr=0.001)
        self.param("Discount factor", state_dis=0.99)
        self.param("Number of trials with learning", learnTrials=None)
        self.param("Replay buffer size", buffer_size=10000)
        self.param("Mini-batch size", batch_size=64)
        self.param("Target network update frequency (gradient steps)", target_update_freq=100)
        self.param("Steps before first gradient update", learning_starts=200)
        self.param("Env steps between gradient updates", train_freq=4)

        ## Reward Centering
        self.param("Reward centering mode", reward_center_mode="none")
        self.param("Reward centering beta", reward_center_beta=0.001)
        self.param("Reward centering eta", reward_center_eta=1.0)
        self.param("Initial avg reward estimate", reward_center_init=0.0)

        ## Optuna search evaluation (see rl_zoo3's TrialEvalCallback pattern)
        self.param("Episodes between deterministic eval checkpoints (0 disables)", eval_freq=0)
        self.param("Deterministic eval episodes per checkpoint", n_eval_episodes=10)
        self.param("Greedy eval episodes of the FINAL policy after training (0 disables)", final_eval_episodes=0)
        self.param("Initial-weight seed: 0 = legacy fixed init, <0 = use the run seed", init_seed=0)

        ## Adam Optimizer
        self.param("Adam beta1", adam_beta1=0.9)
        self.param("Adam beta2", adam_beta2=0.999)
        self.param("Adam epsilon", adam_eps=1e-8)
        self.param("Max gradient norm", max_grad_norm=10.0)

        ## eval_v3 hooks (all default off; with defaults the run is unchanged)
        self.param("Seed the envs from the run seed (first reset only) so a run is reproducible", seed_env=False)
        self.param("Save the final policy to this path ('' disables)", save_policy_path="")
        self.param("Probe-state .npz (key 'states'); logs Q statistics at every greedy checkpoint ('' disables)",
                   probe_path="")
        self.param("Extra in-memory greedy verify episodes before saving (needs save_policy_path)",
                   verify_episodes=0)

    def evaluate(self, param):
        total_start = time.time()
        build_start = time.time()

        trials = param.trials
        steps = param.steps
        learnTrials = param.learnTrials if param.learnTrials is not None else trials

        env = gym.make(param.env)
        env._max_episode_steps = steps
        obs_dim = len(env.observation_space.high)

        low = env.observation_space.low.copy()
        low[1] = -10
        low[3] = -10
        high = env.observation_space.high.copy()
        high[1] = 10
        high[3] = 10

        if param.normalize_state:
            self.state_scale = np.array([4.8, 10.0, 0.418, 10.0])
            domain_bounds_ = np.array([[-1, -1, -1, -1], [1, 1, 1, 1]]).T
        else:
            domain_bounds_ = np.array([low, high]).T

        if param.rep_ in ("HexSSP", "PlaceSSP"):
            rep = SSPRep(
                domain_dim=obs_dim,
                length_scale=param.length_scale,
                n_rotates=param.n_rotates,
                n_scales=getattr(param, "n_scales", 5),
            )
        elif param.rep_ == "Normal":
            rep = NormalRep(env)
            rep.upper = high
            rep.lower = low
            rep.ranges = rep.upper - rep.lower
        elif param.rep_ == "TileCoding":
            tile_low  = np.array([-1., -1., -1., -1.]) if param.normalize_state else low
            tile_high = np.array([ 1.,  1.,  1.,  1.]) if param.normalize_state else high
            rep = TileCodingRep(
                env,
                num_tilings=param.num_tilings,
                tiles_per_dim=param.tiles_per_dim,
                iht_size=param.iht_size,
                bounds_low=tile_low,
                bounds_high=tile_high,
                state_indices=param.tile_state_indices,
            )
        elif param.rep_ == "Discrete":
            rep = OneHotRepCP(
                (param.n_bins, param.n_bins, param.n_bins, param.n_bins),
                bounds_low=low,
                bounds_high=high,
            )
        else:
            raise ValueError(f"Unknown representation: {param.rep_}")

        n_actions = env.action_space.n
        obs_dim_rep = rep.size_out

        eval_env = None
        if param.eval_freq > 0:
            eval_env = gym.make(param.env)
            eval_env._max_episode_steps = steps

        # TileCodingRep exposes map_indices() (sparse active-tile-index encode);
        # other representations only have the dense map(). DQN (dqnBasic.py)
        # auto-detects sparsity via rep.n_active, so using whichever encode
        # method is available here is enough to route TileCoding onto the
        # cheap sparse update path without touching other representations or
        # the legacy A2C/Nengo pipeline (trial_cartpole_nengo.py), which still calls
        # rep.map() directly.
        encode = rep.map_indices if hasattr(rep, "map_indices") else rep.map

        if param.architecture == "linear":
            hidden_dims = None
        elif param.architecture == "deep":
            hidden_dims = [64, 64]
        else:
            raise ValueError(f"Unknown architecture: {param.architecture!r} (expected 'linear' or 'deep')")

        dqn = DQN(
            rep=rep,
            state_dim=obs_dim,
            n_actions=n_actions,
            lr=param.lr,
            gamma=param.state_dis,
            buffer_size=param.buffer_size,
            batch_size=param.batch_size,
            target_update_freq=param.target_update_freq,
            reward_center_mode=param.reward_center_mode,
            reward_center_beta=param.reward_center_beta,
            reward_center_eta=param.reward_center_eta,
            reward_center_init=param.reward_center_init,
            adam_beta1=param.adam_beta1,
            adam_beta2=param.adam_beta2,
            adam_eps=param.adam_eps,
            max_grad_norm=param.max_grad_norm,
            hidden_dims=hidden_dims,
            # 0 (default) = legacy fixed init, used by every search trial. A negative value
            # means "follow this run's seed" (used by the reeval and final eval).
            init_seed=(int(param.seed) if param.init_seed < 0 else int(param.init_seed)),
        )

        build_end = time.time()

        eps = param.eps
        Ep_rewards = []
        last_eval_return = float("nan")
        rdata = {}
        sdata = {}
        vdata = {}  # stores max Q-value per step
        global_step = 0
        eval_history = []
        eval_time = 0.0
        n_updates = 0

        self.data_dir = os.path.join(os.path.dirname(__file__), param.data_dir, param.data_filename)

        eval_seeded = False
        probe_states = None
        probe_history = []
        if param.probe_path:
            probe_states = np.load(param.probe_path)["states"].astype(np.float32)
        if param.seed_env:
            env.action_space.seed(int(param.seed))

        trials_start = time.time()
        for trial in tqdm(range(trials)):
            trial_str = f"trial{trial}"

            rs = []
            cps, cvs, pas, pvs = [], [], [], []
            qs = []  # max Q-value per step

            # seed_env: only the very first reset is seeded; gymnasium's stream then continues.
            raw_obs = env.reset(seed=int(param.seed))[0] if (param.seed_env and trial == 0) else env.reset()[0]
            state = rep.get_state(raw_obs[:obs_dim], env)
            if param.normalize_state:
                state = state / self.state_scale
            phi = encode(state)

            ep_exploring = trial < learnTrials

            for step in range(steps):
                q_values = dqn.get_q_values(phi)

                if global_step < param.learning_starts:
                    action = env.action_space.sample()
                elif ep_exploring and np.random.random() < eps:
                    action = env.action_space.sample()
                else:
                    action = int(q_values.argmax())

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                next_state = rep.get_state(obs[:obs_dim], env)
                if param.normalize_state:
                    next_state = next_state / self.state_scale
                next_phi = encode(next_state)

                dqn.push(phi, action, reward, next_phi, done)

                if (global_step >= param.learning_starts
                        and global_step % param.train_freq == 0
                        and trial < learnTrials):
                    dqn.update()
                    n_updates += 1

                rs.append(reward)
                cps.append(obs[0])
                cvs.append(obs[1])
                pas.append(obs[2])
                pvs.append(obs[3])
                qs.append(q_values.max().item())

                phi = next_phi
                global_step += 1

                if done:
                    break

            rdata[trial_str] = rs
            vdata[trial_str] = qs
            for label, data in zip(["cp", "cv", "pa", "pv"], [cps, cvs, pas, pvs]):
                sdata[f"{label}-{trial_str}"] = data

            Ep_rewards.append(float(np.sum(rs)))

            if eval_env is not None and (trial + 1) % param.eval_freq == 0:
                _eval_t0 = time.time()
                _first_seed = (int(param.seed) + 1_000_000) if (param.seed_env and not eval_seeded) else None
                eval_seeded = True
                last_eval_return = _run_eval_episodes(
                    dqn, rep, encode, eval_env, obs_dim, param.n_eval_episodes, steps,
                    param.normalize_state, self.state_scale if param.normalize_state else None,
                    first_reset_seed=_first_seed,
                )
                eval_time += time.time() - _eval_t0
                eval_history.append(float(last_eval_return))
                if probe_states is not None:
                    _probe_t0 = time.time()
                    _mq, _gap = _probe_stats(
                        dqn, rep, encode, env, probe_states, obs_dim, param.normalize_state,
                        self.state_scale if param.normalize_state else None,
                    )
                    probe_history.append({
                        "checkpoint": len(eval_history),
                        "episode": trial + 1,
                        "mean_max_q": _mq,
                        "mean_action_gap": _gap,
                        "avg_reward": float(dqn.avg_reward),
                    })
                    eval_time += time.time() - _probe_t0
                _cb = getattr(self, "_pruning_callback", None)
                if _cb is not None:
                    _cb(last_eval_return)  # raises optuna.exceptions.TrialPruned() internally if pruned

            if param.dynamic_epsilon and trial >= 20:
                if np.mean(Ep_rewards[trial - 10 : trial]) > np.mean(
                    Ep_rewards[trial - 20 : trial - 10]
                ) + np.std(Ep_rewards[trial - 20 : trial - 10]):
                    eps = np.clip(eps - 0.001, 0.0, 1.0)

        # Optional longer greedy evaluation of the FINAL policy (added 2026-10-04; 0 = off).
        # Used by the final-eval script: 10 episodes per checkpoint is noisy, this is cheap
        # next to training. Counted in eval_time, not train_time.
        final_eval_reward = float("nan")
        final_eval_returns = None
        if eval_env is not None and param.final_eval_episodes > 0:
            _eval_t0 = time.time()
            _first_seed = (int(param.seed) + 1_000_000) if (param.seed_env and not eval_seeded) else None
            eval_seeded = True
            final_eval_reward, final_eval_returns = _run_eval_episodes(
                dqn, rep, encode, eval_env, obs_dim, param.final_eval_episodes, steps,
                param.normalize_state, self.state_scale if param.normalize_state else None,
                first_reset_seed=_first_seed, return_list=True,
            )
            eval_time += time.time() - _eval_t0

        # eval_v3: extra greedy episodes on a FRESH env, then the policy file. The save comes
        # last on purpose: every evaluation above (and this one) allocates tile-coding IHT
        # entries, and the saved IHT must contain all of them for an exact reload.
        verify_returns = None
        verify_env_seed = None
        policy_saved = None
        if param.save_policy_path:
            _eval_t0 = time.time()
            _scale = self.state_scale if param.normalize_state else None
            if param.verify_episodes > 0:
                verify_env_seed = int(param.seed) + 2_000_000
                _venv = gym.make(param.env)
                verify_returns = []
                try:
                    for _ep in range(int(param.verify_episodes)):
                        raw_obs = _venv.reset(seed=verify_env_seed)[0] if _ep == 0 else _venv.reset()[0]
                        state = rep.get_state(raw_obs[:obs_dim], _venv)
                        if param.normalize_state:
                            state = state / _scale
                        phi = encode(state)
                        ep_reward = 0.0
                        for _ in range(steps):
                            action = int(dqn.get_q_values(phi).argmax())
                            obs, reward, terminated, truncated, _ = _venv.step(action)
                            ep_reward += reward
                            if terminated or truncated:
                                break
                            state = rep.get_state(obs[:obs_dim], _venv)
                            if param.normalize_state:
                                state = state / _scale
                            phi = encode(state)
                        verify_returns.append(float(ep_reward))
                finally:
                    _venv.close()
            from rl.policy_io import save_policy
            policy_saved = save_policy(
                param.save_policy_path, dqn, rep, dict(vars(param)), int(param.seed), obs_dim=obs_dim,
                normalize_state=bool(param.normalize_state), state_scale=_scale,
                extra={"verify_env_seed": verify_env_seed, "verify_max_steps": int(steps)},
            )
            eval_time += time.time() - _eval_t0

        trials_end = time.time()
        total_end = time.time()

        os.makedirs(self.data_dir, exist_ok=True)

        rdf = pd.DataFrame({k: pd.Series(v) for k, v in rdata.items()})
        rdf.to_csv(os.path.join(self.data_dir, "rewards.csv"))

        sdf = pd.DataFrame({k: pd.Series(v) for k, v in sdata.items()})
        sdf.to_csv(os.path.join(self.data_dir, "states.csv"))

        vdf = pd.DataFrame({k: pd.Series(v) for k, v in vdata.items()})
        vdf.to_csv(os.path.join(self.data_dir, "values.csv"))

        env.close()
        if eval_env is not None:
            eval_env.close()

        reward_rolling_mean = rdf.sum(axis=0).rolling(100).mean()
        # Pandas 2.x: rolling-mean Series is indexed by string labels ('trial0', 'trial1', ...),
        # so positional access via .iloc[] is required (the original `[learnTrials - 1]`
        # used label-based lookup and raised KeyError on the modern pandas).
        learn_idx = min(learnTrials - 1, len(reward_rolling_mean) - 1)
        terminal_reward_learning = reward_rolling_mean.iloc[learn_idx]
        terminal_reward = reward_rolling_mean.iloc[-1]
        episodes_to_learn = next(
            itertools.chain(iter(i for i, v in enumerate(reward_rolling_mean) if v > 495.0), [-1])
        )
        if episodes_to_learn == -1:
            episodes_to_learn = np.nan

        return {
            "dimensionality": obs_dim_rep,
            "terminal_reward_learning": terminal_reward_learning,
            "terminal_reward": terminal_reward,
            # Last deterministic eval-checkpoint's mean return (see
            # _run_eval_episodes) — NaN unless param.eval_freq > 0. This, not
            # terminal_reward, is what the Optuna search scores/prunes on.
            "terminal_eval_reward": last_eval_return,
            "episodes_to_learn": episodes_to_learn,
            "trial_ID": param.data_filename,
            "build_time": build_end - build_start,
            "total_time": total_end - total_start,
            "avg_trial_time": (trials_end - trials_start) / trials,
            # --- extra per-run metadata (added 2026-10-04; additive, nothing above changed) ---
            # Every greedy-eval checkpoint mean in order (NaN-free; empty if eval_freq == 0).
            # terminal_eval_reward above is just eval_history[-1].
            "eval_history": eval_history,
            "eval_time": eval_time,
            "train_time": (trials_end - trials_start) - eval_time,
            "env_steps": global_step,
            "n_updates": n_updates,
            "n_params": int(sum(p.numel() for p in dqn.q_net.parameters())),
            "n_active": getattr(rep, "n_active", None),
            "final_eval_reward": final_eval_reward,
            # --- eval_v3 hooks (None / empty unless the matching param was set) ---
            "final_eval_returns": final_eval_returns,
            "probe_history": probe_history,
            "avg_reward_final": float(dqn.avg_reward),
            "verify_returns": verify_returns,
            "verify_env_seed": verify_env_seed,
            "verify_max_steps": int(steps) if verify_returns is not None else None,
            "policy_saved": policy_saved,
        }
