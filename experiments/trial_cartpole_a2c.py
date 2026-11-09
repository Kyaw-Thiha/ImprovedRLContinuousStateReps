"""CartPole training loop for the NumPy port of the legacy A2C (no Nengo, no pytry, no torch).

Port of `ACTrial.evaluate` in `trial_cartpole_nengo.py` for the configuration the A2C grid used
(TD(0) rule, epsilon-greedy behaviour policy over linear action preferences, rho forced to 1, dynamic
epsilon, no neuron layer). The learning rule lives in `rl/networks/acLinear.py`.

Legacy loop semantics kept on purpose (see tests/test_a2c_port.py, which checks them against the real
legacy rule code):
* episode start = `ac.reset(phi)` (no update); every env step = `ac.step(phi', action, reward)`;
* after a terminal/truncated step, `n_done` (=1) extra `ac.step(phi', action, 0)` calls on the same state;
* the action is drawn from the epsilon-greedy distribution through its CDF with one uniform draw
  (what `np.random.choice(n, p=...)` does);
* dynamic epsilon: after episode t >= 20, eps -= 0.001 when mean(E[t-10:t]) > mean(E[t-20:t-10]) + std(E[t-20:t-10]).

Differences from the legacy trial, all intentional: the first env reset can be seeded from the run seed
(`seed_env`), the exploration RNG is a local `numpy.random.Generator`, and only the metrics needed for the
port gate are returned.
"""

import os
import sys
import time

import gymnasium as gym
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.networks.acLinear import LinearActorCritic  # noqa: E402
from rl.representations.onehot import OneHotRepCP  # noqa: E402
from rl.representations.ssp_legacy import LegacySSPRep  # noqa: E402
from rl.representations.tilecoding import TileCodingRep  # noqa: E402

# Constants of the legacy grid (scripts/run_hparam_search_a2c_nengo.py BASE_PARAMS); searched values
# (lr, eps, representation knobs, centering parameters) come from the winner yaml.
A2C_BASE_PARAMS = dict(
    trials=1000,
    steps=500,
    env="CartPole-v1",
    n_done=1,
    learnTrials=1000,
    dynamic_epsilon=True,
    act_dis=0.9,
    state_dis=0.99,
    reward_center_beta=0.001,
    reward_center_eta=1.0,
    reward_center_init=0.0,
)

WINNER_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "hparam_search", "best")


def load_winner_yaml(path: str):
    """Legacy A2C winner yaml -> (full trial params = A2C_BASE_PARAMS + winner, old best_value). Read only."""
    with open(path) as f:
        blob = yaml.safe_load(f)
    params = dict(A2C_BASE_PARAMS)
    params.update(blob["params"])
    return params, float(blob.get("best_value", float("nan")))


def load_legacy_winner(representation: str, centering: str, directory: str = WINNER_DIR):
    """Same, by cell, from `directory` (default: the git-ignored originals in outputs/hparam_search/best;
    tracked copies for pods live in conf/a2c_winners)."""
    return load_winner_yaml(os.path.join(directory, f"{representation}_{centering}.yaml"))


def _bounds(env):
    low = env.observation_space.low.copy()
    high = env.observation_space.high.copy()
    low[1], low[3] = -10, -10
    high[1], high[3] = 10, 10
    return low, high


def build_rep(params: dict, env):
    """Returns (rep, encode, sparse). `encode(get_state(obs))` gives phi."""
    rep_name = params["rep_"]
    low, high = _bounds(env)
    if rep_name in ("HexSSP", "PlaceSSP"):
        rep = LegacySSPRep(
            len(env.observation_space.high), length_scale=params["length_scale"], n_rotates=int(params["n_rotates"])
        )
        return rep, rep.map, False
    if rep_name == "TileCoding":
        tpd = params.get("tiles_per_dim")
        if isinstance(tpd, (int, float)):
            tpd = [int(tpd)] * 4
        rep = TileCodingRep(
            env,
            num_tilings=int(params["num_tilings"]),
            tiles_per_dim=tpd,
            iht_size=int(params["iht_size"]),
            bounds_low=low,
            bounds_high=high,
            state_indices=params.get("tile_state_indices"),
        )
        # unique: a hash collision makes the dense one-hot have a single 1 there
        return rep, (lambda s: np.unique(rep.map_indices(s))), True
    if rep_name == "Discrete":
        n = int(params["n_bins"])
        rep = OneHotRepCP((n, n, n, n), bounds_low=low, bounds_high=high)
        return rep, rep.map_indices, True
    raise ValueError(f"unsupported rep_ {rep_name!r}")


def build_agent(params: dict, rep, sparse: bool, n_actions: int = 2) -> LinearActorCritic:
    env_dt = params.get("env_dt", 0.001)
    lr = params["lr"] / (env_dt / 0.001) if env_dt is not None else params["lr"]  # legacy lr rescale (1.0 at 0.001)
    return LinearActorCritic(
        rep.size_out,
        n_actions=n_actions,
        lr=lr,
        act_dis=params["act_dis"],
        state_dis=params["state_dis"],
        centering=params.get("reward_center_mode", "none"),
        beta=params["reward_center_beta"],
        eta=params["reward_center_eta"],
        avg_reward_init=params["reward_center_init"],
        sparse=sparse,
    )


def select_action(prefs, eps: float, u: float) -> int:
    """Epsilon-greedy over preferences via one CDF draw; identical to np.random.choice(n, p=probs)."""
    n = len(prefs)
    probs = np.full(n, eps / n, dtype=float)
    probs[np.argmax(prefs)] += 1.0 - eps
    cdf = probs.cumsum()
    cdf /= cdf[-1]
    return int(cdf.searchsorted(u, side="right"))


def greedy_episodes(ac, rep, encode, env, obs_dim, n_episodes, steps, first_reset_seed=None):
    """Greedy (argmax of the actor preferences, no epsilon, no learning) rollouts; same structure as the DQN
    trial's `_run_eval_episodes`. first_reset_seed seeds ONLY the first reset of this call."""
    returns = []
    for ep in range(n_episodes):
        raw = env.reset(seed=int(first_reset_seed))[0] if (ep == 0 and first_reset_seed is not None) else env.reset()[0]
        phi = encode(rep.get_state(raw[:obs_dim], env))
        total = 0.0
        for _ in range(steps):
            obs, reward, terminated, truncated, _ = env.step(int(np.argmax(ac.preferences(phi))))
            total += reward
            if terminated or truncated:
                break
            phi = encode(rep.get_state(obs[:obs_dim], env))
        returns.append(float(total))
    return returns


def probe_stats(ac, rep, encode, env, probe_states, obs_dim) -> dict:
    """Diagnostics on fixed probe states (no learning, no RNG): mean state value V, mean max preference, mean
    |pref(s,0) - pref(s,1)| (action gap), plus the centering average reward R-bar. Tile coding allocates IHT
    entries for unseen tiles; they are rolled back so probing cannot change later training."""
    iht = getattr(rep, "iht", None)
    if iht is not None:
        n0, over0 = len(iht.dictionary), iht.overfullCount
    phis = [encode(rep.get_state(s[:obs_dim], env)) for s in probe_states]
    if iht is not None:
        while len(iht.dictionary) > n0:
            iht.dictionary.popitem()
        iht.overfullCount = over0
    prefs = np.stack([ac.preferences(p) for p in phis])
    vals = np.array([ac.value(p) for p in phis])
    return {
        "mean_value": float(vals.mean()),
        "mean_max_q": float(prefs.max(axis=1).mean()),
        "mean_action_gap": float(np.abs(prefs[:, 0] - prefs[:, 1]).mean()),
        "avg_reward": float(ac.avg_reward),
    }


def train_a2c(
    params: dict,
    seed: int,
    rng=None,
    seed_env: bool = True,
    env=None,
    eval_every: int = 0,
    n_eval_episodes: int = 10,
    final_eval_episodes: int = 0,
    probe_states=None,
    save_policy_path: str = None,
    verify_episodes: int = 0,
) -> dict:
    """Train one run. `rng` needs a `.random()` method (default: numpy Generator(seed)).

    Returns per-episode returns, the old online metric, and the trained agent/rep. With every eval option at
    its default the run is identical to the legacy loop (the gate path). Options, all mirroring the DQN trial:
      eval_every          greedy checkpoint (n_eval_episodes episodes on a separate env) after every
                          `eval_every`-th episode; first checkpoint env seeded with seed + 1_000_000
      final_eval_episodes a longer greedy eval of the final policy (0 = off)
      probe_states        (n, 4) array: log probe_stats() at every checkpoint
      save_policy_path    write the final policy (rl/policy_io_a2c.py); if verify_episodes > 0 first run that
                          many greedy episodes on a fresh env seeded seed + 2_000_000 and report them so the
                          caller can check the reloaded file replays them exactly.  Saved LAST.
    """
    t0 = time.time()
    trials, steps, n_done = int(params["trials"]), int(params["steps"]), int(params["n_done"])
    learn_trials = params.get("learnTrials")
    learn_trials = trials if learn_trials is None else int(learn_trials)
    rng = np.random.default_rng(seed) if rng is None else rng

    env = gym.make(params["env"]) if env is None else env
    obs_dim = len(env.observation_space.high)
    rep, encode, sparse = build_rep(params, env)
    ac = build_agent(params, rep, sparse, n_actions=int(env.action_space.n))
    get_state = rep.get_state
    eps = float(params["eps"])
    dynamic = bool(params.get("dynamic_epsilon", False))
    build_time = time.time() - t0

    eval_env = gym.make(params["env"]) if (eval_every or final_eval_episodes) else None
    eval_seeded = False
    eval_time = 0.0
    eval_history, probe_history = [], []

    returns = []
    env_steps = 0
    for t in range(trials):
        obs = env.reset(seed=int(seed))[0] if (seed_env and t == 0) else env.reset()[0]
        phi = encode(get_state(obs[:obs_dim], env))
        prefs = ac.reset(phi)
        total = 0.0
        for _ in range(steps):
            action = select_action(prefs, eps if t < learn_trials else 0.0, rng.random())
            obs, reward, terminated, truncated, _ = env.step(action)
            env_steps += 1
            phi = encode(get_state(obs[:obs_dim], env))
            prefs = ac.step(phi, action, reward)
            total += reward
            if terminated or truncated:
                for _ in range(n_done):
                    ac.step(phi, action, 0.0)
                break
        returns.append(total)

        if eval_every and (t + 1) % eval_every == 0:
            te = time.time()
            first = (int(seed) + 1_000_000) if (seed_env and not eval_seeded) else None
            eval_seeded = True
            rets = greedy_episodes(ac, rep, encode, eval_env, obs_dim, n_eval_episodes, steps, first)
            eval_history.append(float(np.mean(rets)))
            if probe_states is not None:
                probe_history.append({"checkpoint": len(eval_history), "episode": t + 1,
                                      **probe_stats(ac, rep, encode, env, probe_states, obs_dim)})
            eval_time += time.time() - te

        if dynamic and t >= 20:
            if np.mean(returns[t - 10 : t]) > np.mean(returns[t - 20 : t - 10]) + np.std(returns[t - 20 : t - 10]):
                eps = float(np.clip(eps - 0.001, 0.0, 1.0))

    out = dict(
        returns=returns,
        terminal_reward=float(np.mean(returns[-100:])),  # the old online metric (rolling mean of last 100 episodes)
        final_eps=eps,
        env_steps=env_steps,
        ac=ac,
        rep=rep,
        dimensionality=int(rep.size_out),
        n_params=int(ac.W.shape[0] - 1) * int(ac.W.shape[1]),
        n_active=getattr(rep, "n_active", None),
        build_time=build_time,
        avg_reward_final=float(ac.avg_reward),
    )

    if final_eval_episodes > 0:
        te = time.time()
        first = (int(seed) + 1_000_000) if (seed_env and not eval_seeded) else None
        eval_seeded = True
        rets = greedy_episodes(ac, rep, encode, eval_env, obs_dim, final_eval_episodes, steps, first)
        out["final_eval_returns"] = rets
        out["final_eval_reward"] = float(np.mean(rets))
        eval_time += time.time() - te

    if save_policy_path:
        te = time.time()
        verify_returns, verify_env_seed = None, None
        if verify_episodes > 0:
            verify_env_seed = int(seed) + 2_000_000
            venv = gym.make(params["env"])
            try:
                verify_returns = greedy_episodes(ac, rep, encode, venv, obs_dim, int(verify_episodes), steps,
                                                 verify_env_seed)
            finally:
                venv.close()
        from rl.policy_io_a2c import save_a2c_policy

        out["policy_saved"] = save_a2c_policy(
            save_policy_path, ac, rep, dict(params), int(seed), obs_dim=obs_dim,
            extra={"verify_env_seed": verify_env_seed, "verify_max_steps": int(steps)})
        out["verify_returns"], out["verify_env_seed"], out["verify_max_steps"] = verify_returns, verify_env_seed, steps
        eval_time += time.time() - te

    env.close()
    if eval_env is not None:
        eval_env.close()
    out["eval_time"] = eval_time
    out["total_time"] = time.time() - t0
    out["train_time"] = out["total_time"] - eval_time - build_time
    if eval_history:
        out["eval_history"] = eval_history
        out["probe_history"] = probe_history
    return out
