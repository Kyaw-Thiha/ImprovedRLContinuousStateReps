"""Save / reload a trained DQN policy so it can be re-evaluated later under perturbations.

Why this exists: before eval_v3 no trained policy was ever persisted, so every robustness /
deployment question would need a retrain. The file written here is self-contained: the
representation (SSP phase matrix, tile-coding IHT dictionary, one-hot bin bounds), the network
weights and the exact preprocessing are all stored, so a fresh process can rebuild the policy
and reproduce the training-time greedy behaviour bit for bit.

Design constraints:
  * The file is a `torch.save` of plain python types and tensors ONLY (no pickled classes), and is
    always loaded with `weights_only=True`, so loading a policy can never execute arbitrary code.
  * The tile-coding IHT is stateful (every unseen tile gets the next free index, in order of first
    visit), so the complete dictionary and its overfull counter are stored. The caller must save
    AFTER all evaluation that should be reproducible, because evaluation also allocates tiles.
  * `Policy.clone()` shares the read-only network / SSP matrix but deep-copies the IHT, so a rollout
    under perturbation (which may visit tiles the training never saw) cannot contaminate the saved
    policy or a sibling rollout.
  * The training hot loop in experiments/trial_cartpole_dqn.py is deliberately NOT routed through
    this module; tests prove that `Policy.encode/act` reproduce it exactly.
"""

import copy
import datetime
import hashlib
import os
import sys
import types

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rl.networks.dqnBasic import QNetwork
from rl.representations.onehot import OneHotRepCP
from rl.representations.ssp import SSPRep
from rl.representations.tilecoding import TileCodingRep

FORMAT_VERSION = 1


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _plain(v):
    """Recursively convert numpy / tuple / odd objects into plain JSON-like python types."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        return str(v)  # strip str subclasses (e.g. torch.__version__) that weights_only rejects
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return str(v)


def _tensor(a, dtype=None):
    """Contiguous owning CPU tensor from an array-like, preserving numpy dtype unless given."""
    arr = np.ascontiguousarray(np.asarray(a) if dtype is None else np.asarray(a, dtype=dtype))
    return torch.from_numpy(arr.copy())


def _phase_sha256(phase_matrix: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(phase_matrix, dtype=np.float64).tobytes()).hexdigest()


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _network_spec(dqn) -> dict:
    net = dqn.q_net
    sparse = bool(net.sparse)
    if sparse:
        dim = int(net.emb.num_embeddings)
        if len(net.layers) > 0:
            hidden = [int(net.emb.embedding_dim)] + [int(l.out_features) for l in net.layers[:-1]]
        else:
            hidden = None
    else:
        dim = int(net.layers[0].in_features)
        hidden = [int(l.out_features) for l in net.layers[:-1]] or None
    return {
        "hidden_dims": hidden,
        "sparse": sparse,
        "obs_dim": dim,
        "n_actions": int(dqn.n_actions),
        "state_dict": {k: v.detach().cpu().contiguous().clone() for k, v in net.state_dict().items()},
    }


def _rep_kind_of(rep) -> str:
    """'ssp' | 'tile' | 'discrete' by class NAME plus duck-typed attributes, never by class identity.

    Why not isinstance: the same rl.representations module can legitimately exist twice in one process
    (tests that purge sys.modules['rl*'] and re-import, a worker importing the trial via two sys.path
    entries). An object built from one copy then fails isinstance against the other copy, although it is
    exactly the same code. Names/attributes are stable across copies.
    """
    name = type(rep).__name__
    if name == "SSPRep" or (hasattr(rep, "_encoder") and hasattr(rep._encoder, "phase_matrix")):
        return "ssp"
    if name == "TileCodingRep" or (hasattr(rep, "iht") and hasattr(rep, "num_tilings")):
        return "tile"
    if name == "OneHotRepCP" or (hasattr(rep, "bounds_low") and hasattr(rep, "factors")
                                 and hasattr(rep, "map_indices")):
        return "discrete"
    raise ValueError(f"Unsupported representation for policy_io: {name}")


def _rep_spec(rep, params: dict, obs_dim: int) -> dict:
    kind = _rep_kind_of(rep)
    if kind == "ssp":
        enc = rep._encoder
        phase = np.asarray(enc.phase_matrix, dtype=np.float64)
        return {
            "kind": "ssp",
            "domain_dim": int(enc.domain_dim),
            "length_scale": float(np.asarray(enc.length_scale).flat[0]),
            "n_rotates": int(getattr(rep, "n_rotates", params.get("n_rotates", 5))),
            "n_scales": int(getattr(rep, "n_scales", params.get("n_scales", 5) or 5)),
            "kernel": str(getattr(rep, "kernel", "hypergeom")),
            "phase_matrix": _tensor(phase),
            "phase_sha256": _phase_sha256(phase),
            "size_out": int(rep.size_out),
        }
    if kind == "tile":
        items = list(rep.iht.dictionary.items())
        klen = 1 + len(rep.state_indices)
        keys = np.asarray([list(k) for k, _ in items], dtype=np.int64).reshape(len(items), klen)
        vals = np.asarray([v for _, v in items], dtype=np.int64)
        return {
            "kind": "tile",
            "obs_dim": int(obs_dim),
            "num_tilings": int(rep.num_tilings),
            "tiles_per_dim": _tensor(rep.tiles_per_dim, np.float64),
            "iht_size": int(rep.iht.size),
            "lower": _tensor(rep.lower, np.float64),
            "upper": _tensor(rep.upper, np.float64),
            "state_indices": _tensor(rep.state_indices, np.int64),
            "iht_keys": _tensor(keys),
            "iht_values": _tensor(vals),
            "iht_overfull": int(rep.iht.overfullCount),
            "size_out": int(rep.size_out),
        }
    if kind == "discrete":
        if rep.bounds_low is None or rep.bounds_high is None:
            raise ValueError("OneHotRepCP without explicit bounds cannot be saved (it would need an env)")
        return {
            "kind": "discrete",
            "ranges": [int(r) for r in rep.ranges],
            # dtype preserved on purpose: (state - low) / window is float32 vs float64 arithmetic and a
            # different dtype can move a value across a bin edge.
            "bounds_low": _tensor(rep.bounds_low),
            "bounds_high": _tensor(rep.bounds_high),
            "size_out": int(rep.size_out),
        }
    raise ValueError(f"Unsupported representation for policy_io: {type(rep).__name__}")  # pragma: no cover


def _provenance() -> dict:
    import gymnasium
    return {
        "torch": str(torch.__version__),
        "gymnasium": str(gymnasium.__version__),
        "numpy": str(np.__version__),
        "python": sys.version.split()[0],
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------------------------
def save_policy(path, dqn, rep, params: dict, seed: int, obs_dim: int = 4, normalize_state: bool = False,
                state_scale=None, extra: dict = None) -> dict:
    """Write the final policy atomically (tmp file + rename). Returns {"path","bytes","sha256"}."""
    blob = {
        "format_version": FORMAT_VERSION,
        "rep": _rep_spec(rep, params or {}, int(obs_dim)),
        "network": _network_spec(dqn),
        "preprocessing": {
            "obs_dim": int(obs_dim),
            "normalize_state": bool(normalize_state),
            "state_scale": None if state_scale is None else _tensor(state_scale, np.float64),
        },
        "avg_reward": float(dqn.avg_reward),
        "params": _plain(params or {}),
        "seed": int(seed),
        "extra": _plain(extra or {}),
        "provenance": _plain(_provenance()),
    }
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        torch.save(blob, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"path": path, "bytes": os.path.getsize(path), "sha256": _file_sha256(path)}


# ---------------------------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------------------------
def _build_rep(spec: dict):
    kind = spec["kind"]
    if kind == "ssp":
        rep = SSPRep(
            domain_dim=int(spec["domain_dim"]),
            length_scale=float(spec["length_scale"]),
            n_rotates=int(spec["n_rotates"]),
            n_scales=int(spec["n_scales"]),
            kernel=spec["kernel"],
        )
        saved = spec["phase_matrix"].numpy().astype(np.float64)
        if _phase_sha256(saved) != spec["phase_sha256"]:
            raise ValueError("SSP phase matrix in the file does not match its checksum (corrupted file)")
        built = np.asarray(rep._encoder.phase_matrix, dtype=np.float64)
        rebuilt_equal = built.shape == saved.shape and np.array_equal(built, saved)
        # The saved matrix is the ground truth: a different scipy/numpy could regenerate a slightly
        # different rotation, and the network was trained on the saved one.
        if not rebuilt_equal:
            if built.shape != saved.shape:
                raise ValueError(f"SSP rebuild shape {built.shape} != saved {saved.shape}")
            rep._encoder.phase_matrix = saved
        rep._rebuilt_equal = bool(rebuilt_equal)
        return rep
    if kind == "tile":
        obs_dim = int(spec["obs_dim"])
        sidx = spec["state_indices"].numpy().astype(int)
        lower_full = np.zeros(obs_dim)
        upper_full = np.zeros(obs_dim)
        lower_full[sidx] = spec["lower"].numpy()
        upper_full[sidx] = spec["upper"].numpy()
        shim = types.SimpleNamespace(
            observation_space=types.SimpleNamespace(high=np.zeros(obs_dim), low=np.zeros(obs_dim)))
        rep = TileCodingRep(
            shim,
            num_tilings=int(spec["num_tilings"]),
            tiles_per_dim=spec["tiles_per_dim"].numpy().copy(),
            iht_size=int(spec["iht_size"]),
            bounds_low=lower_full,
            bounds_high=upper_full,
            state_indices=sidx,
        )
        keys = spec["iht_keys"].tolist()
        vals = spec["iht_values"].tolist()
        rep.iht.dictionary = {tuple(k): int(v) for k, v in zip(keys, vals)}
        rep.iht.overfullCount = int(spec["iht_overfull"])
        return rep
    if kind == "discrete":
        return OneHotRepCP(
            tuple(int(r) for r in spec["ranges"]),
            bounds_low=spec["bounds_low"].numpy().copy(),
            bounds_high=spec["bounds_high"].numpy().copy(),
        )
    raise ValueError(f"unknown rep kind {kind!r}")


def load_policy(path) -> "Policy":
    """Rebuild a Policy. `weights_only=True`: the file can only contain tensors and plain types."""
    blob = torch.load(os.fspath(path), map_location="cpu", weights_only=True)
    if blob.get("algo") == "a2c_linear":  # A2C port policies (rl/policy_io_a2c.py); DQN files carry no `algo`
        from rl.policy_io_a2c import policy_from_blob
        return policy_from_blob(blob)
    if blob.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported policy format_version {blob.get('format_version')!r}")
    return Policy(blob)


class Policy:
    """Greedy policy over a raw 4-dim CartPole observation. See module docstring."""

    def __init__(self, blob: dict = None, *, _clone_of: "Policy" = None):
        if _clone_of is not None:
            self.__dict__.update(_clone_of.__dict__)
            return
        pre = blob["preprocessing"]
        self.obs_dim = int(pre["obs_dim"])
        self.normalize_state = bool(pre["normalize_state"])
        self.state_scale = None if pre["state_scale"] is None else pre["state_scale"].numpy().astype(np.float64)
        rep_spec = blob["rep"]
        self.rep_kind = rep_spec["kind"]
        self.rep = _build_rep(rep_spec)
        self.dim = int(self.rep.size_out)

        spec = blob["network"]
        self.hidden_dims = None if spec["hidden_dims"] is None else [int(h) for h in spec["hidden_dims"]]
        self.sparse = bool(spec["sparse"])
        self.n_actions = int(spec["n_actions"])
        if int(spec["obs_dim"]) != self.dim:
            raise ValueError(f"network input {spec['obs_dim']} != representation size {self.dim}")
        # QNetwork.__init__ draws from the global torch RNG; fork it so loading a policy never
        # perturbs the caller's random stream.
        with torch.random.fork_rng(devices=[]):
            self.net = QNetwork(self.dim, self.n_actions, self.hidden_dims, sparse=self.sparse)
        self.net.load_state_dict({k: v for k, v in spec["state_dict"].items()})
        self.net.eval()
        self.net.requires_grad_(False)
        self.n_params = int(sum(p.numel() for p in self.net.parameters()))

        self.meta = {
            "format_version": int(blob["format_version"]),
            "params": blob.get("params", {}),
            "seed": int(blob.get("seed", -1)),
            "avg_reward": float(blob.get("avg_reward", 0.0)),
            "provenance": blob.get("provenance", {}),
            "extra": blob.get("extra", {}),
        }
        if self.rep_kind == "ssp":
            self.meta["ssp_rebuilt_equal"] = bool(getattr(self.rep, "_rebuilt_equal", True))

    # --- encoding ---------------------------------------------------------------------------
    def _state(self, obs):
        state = self.rep.get_state(np.asarray(obs)[: self.obs_dim], None)
        if self.normalize_state:
            state = state / self.state_scale
        return state

    def _map(self, state):
        if self.rep_kind == "ssp":
            return self.rep.map(state)
        phi = self.rep.map_indices(state)
        if self.rep_kind == "discrete":
            idx = int(phi[0])
            if not 0 <= idx < self.dim:
                # Only reachable for observations far outside anything training can produce
                # (e.g. heavy observation noise): clip each bin so the lookup stays defined.
                # In-range inputs never take this branch, so training equivalence is untouched.
                hi = np.asarray(self.rep.ranges, dtype=np.int64) - 1
                phi = self.rep.map_indices(np.clip(np.asarray(state), 0, hi))
        return phi

    def encode(self, obs):
        """Raw env observation -> phi, exactly the trial loop's get_state / normalize / encode."""
        return self._map(self._state(obs))

    # --- Q-values ---------------------------------------------------------------------------
    def q_from_phi(self, phi) -> np.ndarray:
        with torch.no_grad():
            if self.sparse:
                out = self.net.forward_sparse(torch.as_tensor(phi, dtype=torch.int64))
            else:
                out = self.net.forward_dense(torch.as_tensor(phi, dtype=torch.float32))
        return out.detach().cpu().numpy().astype(np.float32, copy=True)

    def q_values(self, obs) -> np.ndarray:
        return self.q_from_phi(self.encode(obs))

    def q_values_batch(self, obs_batch) -> np.ndarray:
        obs_batch = np.asarray(obs_batch)
        if len(obs_batch) == 0:
            return np.zeros((0, self.n_actions), dtype=np.float32)
        phis = np.stack([np.asarray(self.encode(o)) for o in obs_batch])
        with torch.no_grad():
            if self.sparse:
                out = self.net.forward_sparse(torch.as_tensor(phis, dtype=torch.int64))
            else:
                out = self.net.forward_dense(torch.as_tensor(phis, dtype=torch.float32))
        return out.detach().cpu().numpy().astype(np.float32, copy=True)

    def act(self, obs) -> int:
        # np.argmax returns the first (lowest-index) maximum, matching the training-time eval.
        return int(np.argmax(self.q_values(obs)))

    # --- lifecycle --------------------------------------------------------------------------
    def clone(self) -> "Policy":
        """Cheap copy for one rollout: shares the read-only network and SSP matrix, deep-copies
        the only mutable encoder state (the tile-coding IHT)."""
        other = Policy(_clone_of=self)
        if self.rep_kind == "tile":
            rep = copy.copy(self.rep)
            # type(...) of the rep's own IHT: tiles()/hashcoords() compare `type(m) == IHT` against the
            # IHT of the module the rep class came from, so the clone must use that very class.
            iht = type(self.rep.iht)(self.rep.iht.size)
            iht.dictionary = dict(self.rep.iht.dictionary)
            iht.overfullCount = self.rep.iht.overfullCount
            rep.iht = iht
            other.rep = rep
        return other

    def reset(self) -> None:
        return None


# ---------------------------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------------------------
def greedy_returns(act_fn, env_seed: int, n_episodes: int, max_steps: int = 500, env_id: str = "CartPole-v1"):
    """C3 rollout semantics: one env, reset(seed=env_seed) on the first episode only, greedy actions."""
    env = gym.make(env_id)
    returns = []
    try:
        for ep in range(n_episodes):
            obs = env.reset(seed=int(env_seed))[0] if ep == 0 else env.reset()[0]
            total = 0.0
            for _ in range(max_steps):
                obs, reward, terminated, truncated, _ = env.step(act_fn(obs))
                total += reward
                if terminated or truncated:
                    break
            returns.append(float(total))
    finally:
        env.close()
    return returns


def verify_saved_policy(path, env_seed: int, n_episodes: int, expected_returns: list,
                        max_steps: int = 500) -> dict:
    """Reload `path` and replay the greedy episodes; `ok` only if every return matches exactly."""
    policy = load_policy(path)
    returns = greedy_returns(policy.act, env_seed, n_episodes, max_steps)
    exp = [float(x) for x in expected_returns]
    if len(exp) != len(returns):
        return {"ok": False, "max_abs_diff": float("inf"), "returns": returns}
    diff = max((abs(a - b) for a, b in zip(returns, exp)), default=0.0)
    return {"ok": bool(diff == 0.0), "max_abs_diff": float(diff), "returns": returns}
