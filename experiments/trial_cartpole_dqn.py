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
import rl as net
from rl.networks.dqnBasic import DQN
from base_trial import BaseTrial

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


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

        ## DQN Parameters
        self.param("Epsilon for epsilon-greedy", eps=0.1)
        self.param("Dynamic Epsilon", dynamic_epsilon=False)
        self.param("Learning rate", lr=0.001)
        self.param("Discount factor", state_dis=0.99)
        self.param("Number of trials with learning", learnTrials=None)
        self.param("Replay buffer size", buffer_size=10000)
        self.param("Mini-batch size", batch_size=64)
        self.param("Target network update frequency (gradient steps)", target_update_freq=100)
        self.param("Steps before first gradient update", learning_starts=500)
        self.param("Env steps between gradient updates", train_freq=1)

        ## Reward Centering
        self.param("Reward centering mode", reward_center_mode="none")
        self.param("Reward centering beta", reward_center_beta=0.001)
        self.param("Reward centering eta", reward_center_eta=1.0)
        self.param("Initial avg reward estimate", reward_center_init=0.0)

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
            rep = net.representations.SSPRep(
                obs_dim,
                length_scale=param.length_scale,
                n_rotates=param.n_rotates,
                domain_bounds=domain_bounds_,
            )
        elif param.rep_ == "Normal":
            rep = net.representations.NormalRep(env)
            rep.upper = high
            rep.lower = low
            rep.ranges = rep.upper - rep.lower
        elif param.rep_ == "TileCoding":
            rep = net.representations.TileCodingRep(
                env,
                num_tilings=param.num_tilings,
                tiles_per_dim=param.tiles_per_dim,
                iht_size=param.iht_size,
                bounds_low=low,
                bounds_high=high,
                state_indices=param.tile_state_indices,
            )
        elif param.rep_ == "Discrete":
            rep = net.representations.OneHotRepCP(
                (param.n_bins, param.n_bins, param.n_bins, param.n_bins)
            )
        else:
            raise ValueError(f"Unknown representation: {param.rep_}")

        n_actions = env.action_space.n
        obs_dim_rep = rep.size_out

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
        )

        build_end = time.time()

        eps = param.eps
        Ep_rewards = []
        rdata = {}
        sdata = {}
        vdata = {}  # stores max Q-value per step
        global_step = 0

        self.data_dir = os.path.join(os.path.dirname(__file__), param.data_dir, param.data_filename)

        trials_start = time.time()
        for trial in tqdm(range(trials)):
            trial_str = f"trial{trial}"

            rs = []
            cps, cvs, pas, pvs = [], [], [], []
            qs = []  # max Q-value per step

            raw_obs = env.reset()[0]
            state = rep.get_state(raw_obs[:obs_dim], env)
            if param.normalize_state:
                state = state / self.state_scale
            phi = rep.map(state)

            ep_exploring = trial < learnTrials

            for step in range(steps):
                q_values = dqn.get_q_values(phi)

                if ep_exploring and np.random.random() < eps:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(q_values))

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                next_state = rep.get_state(obs[:obs_dim], env)
                if param.normalize_state:
                    next_state = next_state / self.state_scale
                next_phi = rep.map(next_state)

                dqn.push(state, action, reward, next_state, done)

                if global_step >= param.learning_starts and global_step % param.train_freq == 0:
                    dqn.update()

                rs.append(reward)
                cps.append(obs[0])
                cvs.append(obs[1])
                pas.append(obs[2])
                pvs.append(obs[3])
                qs.append(float(np.max(q_values)))

                phi = next_phi
                global_step += 1

                if done:
                    break

            rdata[trial_str] = rs
            vdata[trial_str] = qs
            for label, data in zip(["cp", "cv", "pa", "pv"], [cps, cvs, pas, pvs]):
                sdata[f"{label}-{trial_str}"] = data

            Ep_rewards.append(float(np.sum(rs)))

            _cb = getattr(self, "_pruning_callback", None)
            if _cb is not None:
                _rolling = float(
                    np.mean(Ep_rewards[-100:]) if len(Ep_rewards) >= 100 else np.mean(Ep_rewards)
                )
                _cb(trial, _rolling)

            if param.dynamic_epsilon:
                if np.mean(Ep_rewards[trial - 10 : trial]) > np.mean(
                    Ep_rewards[trial - 20 : trial - 10]
                ) + np.std(Ep_rewards[trial - 20 : trial - 10]):
                    eps = np.clip(eps - 0.001, 0.0, 1.0)

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

        reward_rolling_mean = rdf.sum(axis=0).rolling(100).mean()
        terminal_reward_learning = reward_rolling_mean[learnTrials - 1]
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
            "episodes_to_learn": episodes_to_learn,
            "trial_ID": param.data_filename,
            "build_time": build_end - build_start,
            "total_time": total_end - total_start,
            "avg_trial_time": (trials_end - trials_start) / trials,
        }
