"""Save / reload a trained linear actor-critic (A2C port) policy for eval_v3.

Same contract as `rl/policy_io.py` (which handles DQN): a `torch.save` of plain python types and tensors
only, always loaded with `weights_only=True`, written atomically, self-contained (representation + weights).
`rl.policy_io.load_policy` dispatches here when the file's `algo` is "a2c_linear", so every eval script that
calls `load_policy` (robustness, deployment, verification) works on A2C policies unchanged.

`LinearACPolicy` deliberately exposes the SAME surface as the DQN `Policy` (`act`, `encode`, `q_from_phi`,
`q_values(_batch)`, `clone`, `reset`, and for the deployment code `rep_kind`, `rep`, `dim`, `sparse`,
`hidden_dims`, `net`, `n_params`, `n_actions`), so no eval script needs an A2C branch:
  * `net` is a one-tensor torch module `weight` of shape (n_actions, dim): the ACTOR rows of W. The critic
    (value) row is only needed for learning, so it is stored in the file (`value_row`, for diagnostics) but is
    NOT part of the deployed policy and not counted in its size.
  * the "Q-values" returned are the action PREFERENCES; the policy is argmax of them.
  * `net.sparse` / `net.emb.embedding_dim` / `net.layers` mimic the DQN QNetwork only as far as
    `run_deployment.network_macs` reads them.
"""

import copy
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rl.policy_io import (  # noqa: E402
    _build_rep as _build_rep_dqn,
    _file_sha256,
    _phase_sha256,
    _plain,
    _provenance,
    _rep_kind_of,
    _rep_spec as _rep_spec_dqn,
    _tensor,
)
from rl.representations.ssp_legacy import LegacySSPRep  # noqa: E402

FORMAT_VERSION = 1
ALGO = "a2c_linear"


# ---------------------------------------------------------------------------------------------
# representation (de)serialisation: tile / discrete reuse the DQN code, SSP is the legacy encoder
# ---------------------------------------------------------------------------------------------
def rep_spec(rep, params: dict, obs_dim: int) -> dict:
    if _rep_kind_of(rep) == "ssp":
        enc = rep._encoder
        phase = np.asarray(enc.phase_matrix, dtype=np.float64)
        return {
            "kind": "ssp_legacy",
            "domain_dim": int(enc.domain_dim),
            "length_scale": float(np.asarray(enc.length_scale).flat[0]),
            "n_rotates": int(rep.n_rotates),
            "n_scales": int(rep.n_scales),
            "scale_min": float(rep.scale_min),
            "scale_max": float(rep.scale_max),
            "phase_matrix": _tensor(phase),
            "phase_sha256": _phase_sha256(phase),
            "size_out": int(rep.size_out),
        }
    return _rep_spec_dqn(rep, params, obs_dim)


def build_rep(spec: dict):
    if spec["kind"] == "ssp_legacy":
        saved = spec["phase_matrix"].numpy().astype(np.float64)
        if _phase_sha256(saved) != spec["phase_sha256"]:
            raise ValueError("SSP phase matrix in the file does not match its checksum (corrupted file)")
        # The saved matrix is the ground truth (no rebuild from n_rotates/n_scales).
        return LegacySSPRep(
            int(spec["domain_dim"]), float(spec["length_scale"]), int(spec["n_rotates"]), int(spec["n_scales"]),
            float(spec["scale_min"]), float(spec["scale_max"]), phase_matrix=saved,
        )
    return _build_rep_dqn(spec)


# ---------------------------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------------------------
def save_a2c_policy(path, ac, rep, params: dict, seed: int, obs_dim: int = 4, extra: dict = None) -> dict:
    """Write the final policy atomically. Returns {"path", "bytes", "sha256"}. Call AFTER all evaluation that
    should be reproducible: evaluation allocates tile-coding IHT entries and the saved IHT must contain them."""
    W = np.asarray(ac.W, dtype=np.float64)
    blob = {
        "format_version": FORMAT_VERSION,
        "algo": ALGO,
        "rep": rep_spec(rep, params or {}, int(obs_dim)),
        "weights": {"actor": _tensor(W[1:]), "value_row": _tensor(W[0])},
        "agent": {
            "n_actions": int(ac.n_actions),
            "lr": float(ac.lr),
            "act_dis": float(ac.act_dis),
            "state_dis": float(ac.state_dis),
            "centering": str(ac.centering),
            "avg_reward": float(ac.avg_reward),
            "sparse": bool(ac.sparse),
        },
        "preprocessing": {"obs_dim": int(obs_dim)},
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
# policy object
# ---------------------------------------------------------------------------------------------
class ActorHead(torch.nn.Module):
    """The deployed policy parameters: ONE weight matrix (n_actions, dim). Mimics just enough of the DQN
    QNetwork (`sparse`, `emb.embedding_dim`, `layers`) for `run_deployment.network_macs`."""

    def __init__(self, weight: torch.Tensor, sparse: bool):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        n_actions, dim = int(weight.shape[0]), int(weight.shape[1])
        self.sparse = bool(sparse)
        self.emb = types.SimpleNamespace(embedding_dim=n_actions, num_embeddings=dim)
        self.layers = [] if sparse else [types.SimpleNamespace(in_features=dim, out_features=n_actions)]


class LinearACPolicy:
    """Greedy policy (argmax of linear action preferences) over a raw 4-dim CartPole observation."""

    def __init__(self, blob: dict = None, *, _clone_of: "LinearACPolicy" = None):
        if _clone_of is not None:
            self.__dict__.update(_clone_of.__dict__)
            return
        pre = blob["preprocessing"]
        self.obs_dim = int(pre["obs_dim"])
        self.normalize_state = False
        self.state_scale = None
        spec = blob["rep"]
        self.rep = build_rep(spec)
        self.rep_kind = "ssp" if spec["kind"] == "ssp_legacy" else spec["kind"]
        self.dim = int(self.rep.size_out)
        ag = blob["agent"]
        self.n_actions = int(ag["n_actions"])
        self.hidden_dims = None
        self.sparse = self.rep_kind != "ssp"
        weight = blob["weights"]["actor"].to(torch.float64)
        if tuple(weight.shape) != (self.n_actions, self.dim):
            raise ValueError(f"actor weights {tuple(weight.shape)} != ({self.n_actions}, {self.dim})")
        self.net = ActorHead(weight, self.sparse)
        self.net.eval()
        self.n_params = int(weight.numel())
        self.meta = {
            "format_version": int(blob["format_version"]),
            "algo": blob["algo"],
            "params": blob.get("params", {}),
            "seed": int(blob.get("seed", -1)),
            "avg_reward": float(ag["avg_reward"]),
            "agent": ag,
            "provenance": blob.get("provenance", {}),
            "extra": blob.get("extra", {}),
        }

    # --- encoding ---------------------------------------------------------------------------
    def _map(self, state):
        if self.rep_kind == "ssp":
            return self.rep.map(state)
        phi = self.rep.map_indices(state)
        if self.rep_kind == "discrete":
            idx = int(phi[0])
            if not 0 <= idx < self.dim:
                hi = np.asarray(self.rep.ranges, dtype=np.int64) - 1
                phi = self.rep.map_indices(np.clip(np.asarray(state), 0, hi))
            return phi
        return np.unique(phi)  # tile: a hash collision makes the dense one-hot have a single 1 there

    def encode(self, obs):
        """Raw env observation -> phi, exactly the training loop's get_state / encode."""
        state = self.rep.get_state(np.asarray(obs)[: self.obs_dim], None)
        return self._map(state)

    # --- preferences ------------------------------------------------------------------------
    def q_from_phi(self, phi) -> np.ndarray:
        W = self.net.weight.detach().numpy()
        if self.sparse:
            return W[:, phi].sum(axis=1)
        return W @ phi

    def q_values(self, obs) -> np.ndarray:
        return self.q_from_phi(self.encode(obs))

    def q_values_batch(self, obs_batch) -> np.ndarray:
        obs_batch = np.asarray(obs_batch)
        if len(obs_batch) == 0:
            return np.zeros((0, self.n_actions))
        return np.stack([self.q_values(o) for o in obs_batch])

    def act(self, obs) -> int:
        # np.argmax returns the first (lowest-index) maximum, matching the training-time greedy eval.
        return int(np.argmax(self.q_values(obs)))

    # --- lifecycle --------------------------------------------------------------------------
    def clone(self) -> "LinearACPolicy":
        """Cheap copy for one rollout: shares the read-only weights / SSP matrix, copies the only mutable
        encoder state (the tile-coding IHT)."""
        other = LinearACPolicy(_clone_of=self)
        if self.rep_kind == "tile":
            rep = copy.copy(self.rep)
            iht = type(self.rep.iht)(self.rep.iht.size)
            iht.dictionary = dict(self.rep.iht.dictionary)
            iht.overfullCount = self.rep.iht.overfullCount
            rep.iht = iht
            other.rep = rep
        return other

    def reset(self) -> None:
        return None


def policy_from_blob(blob: dict) -> LinearACPolicy:
    if blob.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported a2c policy format_version {blob.get('format_version')!r}")
    return LinearACPolicy(blob)
