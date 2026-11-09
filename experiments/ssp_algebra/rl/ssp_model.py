"""POPGym-compatible RLlib ``ModelV2`` that wraps ``rl_memory.SSPMemory`` (handoff 11 section 3.2 / decision 5).

Runs in the *reference venv* (``/tmp/popgym_ref``: ray 2.59, popgym 1.0.7, torch CPU), through POPGym's own PPO config.
It subclasses POPGym's ``BaseModel`` (the class every ``popgym.baselines.ray_models`` model derives from), so the
observation feature map, the actor/critic heads, the ``t_starts`` / ``seq_lens`` handling and the RLlib recurrent-state
protocol are POPGym's own.  Only ``forward_memory`` (the "core") is new.

Design (one documented default, no per-task tuning)
---------------------------------------------------
* **Tokens.**  The first component of POPGym's observation tuple is a ``Discrete(n)`` (RepeatPrevious: the card suit,
  n = 4).  RLlib one-hot flattens it, so the token is ``argmax(obs_flat[..., :n])`` and indexes ``SSPMemory.codes``.
* **State.**  The streaming bundle ``M`` (``dim`` complex = ``2*dim`` floats; 256 floats at dim = 128 = the GRU's
  ``hidden_size`` 256) plus POPGym's own time counter.  It is RLlib state, so it is **zero at every episode start**
  (RLlib begins a new sequence from ``get_initial_state()``) and the chunk-to-chunk state is carried by RLlib.
* **Write.**  ``M_t = lam * M_{t-1} + code(tok_t) * pos(phase(t))`` with ``phase(t) = w_time . abs_time_feats(t)``
  (exactly ``SSPMemory.write``).  Over a whole ``[B, T]`` chunk this is a causal weighted cumulative sum, computed in
  one shot with a ``T x T`` decay matrix instead of a Python loop (equal to the step loop; tested).
* **Read.**  ``addr = addr_head([h_t, abs_time_feats(t)])`` with ``h_t = hid_proj(z_t)`` (the policy's preprocessed
  features: this model has no recurrent policy state, so the "hidden state" is the feed-forward feature).  Scores
  are ``SSPMemory.read`` cleanup scores of ``M_t`` unbound at ``addr``.
  **Default address init = "now with a slope error"**: the ``abs_time_feats`` ramp weight is set to the write ramp
  (``phase`` scale) times ``(1 + 0.03 N(0,1))`` and the bias to 0 (the ``slope_only_cfg`` recipe: offset exactly 0), so
  the head starts by reading the item it just wrote and must *learn* the lag.  ``rl_memory.SSPMemory`` itself
  initialises the head to the constant address 0; that is replaced here (documented deviation, see REPORT.md).
* **Output.**  ``features_t = LeakyReLU(Linear([z_t, scores_t / tau]))``, size ``hidden_size`` (256), then POPGym's own
  actor / critic.  ``tau = 0.1`` is the aux-loss temperature.
* **Aux loss (flag ``aux_coeff``, default 0 = off).**  The label-free recall loss of ``SSPMemory.aux_loss``
  (cross-entropy of ``scores / tau`` against the token just observed, read at the module's own address), summed over
  valid timesteps, added in ``custom_loss`` as ``aux_coeff * aux``.  Costs no extra samples.
* **``ablate_memory`` flag.**  Zeroes the read-out (scores), turning the model into a feed-forward policy; used for
  the "memory zeroed at evaluation" ablation and the no-op contract test.
"""

import math
import os
import sys
from typing import List, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from ray.rllib.utils.typing import ModelConfigDict, TensorType

_SSP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SSP_DIR not in sys.path:
    sys.path.insert(0, _SSP_DIR)

import feats  # noqa: E402
import rl_memory  # noqa: E402
import ssp_ops as S  # noqa: E402

from popgym.baselines.ray_models.base_model import BaseModel  # noqa: E402

TAU = 0.1


class SSPModel(BaseModel):
    MODEL_CONFIG = {
        "ssp_dim": 128,            # complex dims; carried state = 2 * ssp_dim floats
        "ssp_hidden": 32,          # width of the hidden feature fed to the address head
        "ssp_seed": 0,
        "ssp_lam_init": 0.99,
        "ssp_length_scale": 1.0,
        "ssp_addr_eps": 0.03,      # slope error of the address init (recipes.DEFAULT_SLOPE_EPS)
        "ssp_addr_bias": 0.0,      # initial read offset in index units (0 = now); only the DIAGNOSTIC arm sets it (-4)
        "n_items": None,           # None: read from the first Discrete component of the observation space
        "aux_coeff": 0.0,          # >0 switches the label-free recall auxiliary on
        "ablate_memory": False,    # zero the read-out (feed-forward policy)
    }

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, **custom_model_kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        cfg = self.cfg
        n_items = cfg["n_items"]
        if n_items is None:
            first = getattr(obs_space, "original_space", obs_space)       # RLlib passes the flattened Box
            if isinstance(first, gym.spaces.Tuple):
                first = first.spaces[0]
            assert isinstance(first, gym.spaces.Discrete), "SSPModel needs a discrete first observation component"
            n_items = int(first.n)
        self.n_items = int(n_items)
        self.aux_coeff = float(cfg["aux_coeff"])
        self.ablate_memory = bool(cfg["ablate_memory"])
        dim, hid = int(cfg["ssp_dim"]), int(cfg["ssp_hidden"])
        self.dim = dim

        self.mem = rl_memory.SSPMemory(self.n_items, dim=dim, hidden=hid, seed=int(cfg["ssp_seed"]),
                                       lam_init=float(cfg["ssp_lam_init"]),
                                       length_scale=float(cfg["ssp_length_scale"]))
        self._init_address_head(float(cfg["ssp_addr_eps"]), int(cfg["ssp_seed"]), float(cfg["ssp_addr_bias"]))
        self.hid_proj = nn.Linear(cfg["preprocessor_output_size"], hid)
        self.out_proj = nn.Linear(cfg["preprocessor_output_size"] + self.n_items, cfg["hidden_size"])
        self.act = nn.LeakyReLU()
        self._tok = None
        self._aux = None
        self._aux_value = 0.0

    # ------------------------------------------------------------------------------------------------
    def _init_address_head(self, eps, seed, bias=0.0):
        """Address = t * (1 + eps N(0,1)) at init: the write ramp's slope with a small error, offset exactly 0."""
        g = torch.Generator().manual_seed(seed + 3)
        hid = self.mem.hidden
        with torch.no_grad():
            self.mem.addr_head.weight.zero_()
            self.mem.addr_head.bias.fill_(bias)
            slope = float(self.mem.w_time[0]) * (1.0 + eps * float(torch.randn(1, generator=g)))
            self.mem.addr_head.weight[0, hid] = slope      # column ``hid`` is feature 0 of abs_time_feats = t / 32

    def initial_state(self) -> List[TensorType]:
        return [torch.zeros(1, 2 * self.dim)]

    def state_floats(self) -> int:
        return 2 * self.dim

    # ------------------------------------------------------------------------------------------------
    def forward(self, input_dict, state, seq_lens):
        obs, _, _, _ = self.preprocess_obs(input_dict["obs_flat"], seq_lens.long())
        self._tok = obs[..., : self.n_items].argmax(-1)          # [B, T] long
        return super().forward(input_dict, state, seq_lens)

    def read_address(self, h, tf, t):
        """Read phase (index units, [B,T]).  Hook: the untied head of ``SSPMemory.read_from``; subclasses override."""
        return self.mem.addr_head(torch.cat([h, tf], -1))[..., 0]

    def aux_term(self, scores, tok, seq_lens):
        """Label-free recall loss (``SSPMemory.aux_loss``): the token just observed, read at the module's own address."""
        B, T = tok.shape
        valid = (torch.arange(T, device=scores.device).reshape(1, -1) < seq_lens.reshape(-1, 1)).float()
        ce = nn.functional.cross_entropy((scores / TAU).reshape(B * T, -1), tok.reshape(-1), reduction="none")
        return (ce * valid.reshape(-1)).sum() / valid.sum().clamp_min(1.0)

    def address_summary(self):
        """Scalars logged every PPO iteration (the address-bias trajectory)."""
        w = self.mem.addr_head.weight.detach()
        return {"addr_bias": float(self.mem.addr_head.bias.detach()[0]), "ramp_w": float(w[0, self.mem.hidden]),
                "w_time0": float(self.mem.w_time.detach()[0]), "lam": float(torch.sigmoid(self.mem.lam_raw.detach()))}

    def _write_bundle(self, z, tok, state_vec, t_starts, seq_lens):
        """The streaming write, as a pure function of tensors.  Returns M [B,T,D] complex, tf [B,T,TIME_DIM], t [B,T].

        Split out of ``memory_core`` so a subclass can replace only the *read* (see ``SSPTiedLagBankModel``).  The
        ops, their order and the dtypes are unchanged from the original single-function form.
        """
        m = self.mem
        B, T, _ = z.shape
        D = self.dim
        t = t_starts.reshape(-1, 1).float() + torch.arange(T, device=z.device).float().reshape(1, -1)   # [B,T]
        tf = feats.abs_time_feats(t).to(z.dtype)                                                     # [B,T,13]
        phase = tf @ m.w_time                                                                       # [B,T]
        u = m.codes[tok] * S.pos(phase, m.omega)                                                    # [B,T,D] complex
        log_lam = nn.functional.logsigmoid(m.lam_raw)
        idx = torch.arange(T, device=z.device).float()
        diff = idx.reshape(-1, 1) - idx.reshape(1, -1)                                              # t - s
        L = torch.where(diff >= 0, torch.exp(diff.clamp_min(0) * log_lam), torch.zeros_like(diff))
        M0 = torch.complex(state_vec[..., :D], state_vec[..., D:])                                  # [B,D]
        decay0 = torch.exp((idx + 1.0) * log_lam)                                                   # lam^(t+1)
        M = torch.einsum("ts,bsd->btd", L.to(u.dtype), u) + decay0.reshape(1, T, 1) * M0.unsqueeze(1)
        return M, tf, t

    def memory_core(self, z, tok, state_vec, t_starts, seq_lens):
        """Pure function of tensors (no RLlib state).  Returns scores [B,T,n], M [B,T,D] complex, addr [B,T],
        and the new state vector [B,2D] (the bundle after the last *valid* step of each row)."""
        m = self.mem
        B, T, _ = z.shape
        M, tf, t = self._write_bundle(z, tok, state_vec, t_starts, seq_lens)
        h = self.hid_proj(z)
        addr = self.read_address(h, tf, t)                                                          # [B,T]
        r = M * S.pos(addr, m.omega).conj()
        scores = S.cleanup_scores(r, m.codes)                                                       # [B,T,n]
        last = (seq_lens.long() - 1).clamp(0, T - 1)
        Mlast = M[torch.arange(B, device=z.device), last]
        new_state = torch.cat([Mlast.real, Mlast.imag], -1)
        return scores, M, addr, new_state

    def forward_memory(self, z: TensorType, state: List[TensorType], t_starts: TensorType,
                       seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        B, T, _ = z.shape
        tok = self._tok[:, :T]
        scores, _, _, new_state = self.memory_core(z, tok, state[0].reshape(B, -1), t_starts, seq_lens)
        if self.aux_coeff > 0 and torch.is_grad_enabled():
            self._aux = self.aux_term(scores, tok, seq_lens)
        else:
            self._aux = None
        read = torch.zeros_like(scores) if self.ablate_memory else scores / TAU
        out = self.act(self.out_proj(torch.cat([z, read], -1)))
        return out, [new_state.reshape(B, 1, 2 * self.dim)]

    # ------------------------------------------------------------------------------------------------
    def custom_loss(self, policy_loss, loss_inputs):
        if self._aux is None or self.aux_coeff <= 0:
            return policy_loss
        self._aux_value = float(self._aux.detach())
        extra = self.aux_coeff * self._aux
        if isinstance(policy_loss, (list, tuple)):
            return [l + extra for l in policy_loss]
        return policy_loss + extra

    def metrics(self):
        return {"ssp_aux_loss": self._aux_value, "ssp_lam": float(torch.sigmoid(self.mem.lam_raw).detach())}

    def core_params(self):
        """Parameters of the memory path (not the shared POPGym feature map / actor / critic)."""
        mods = [self.mem, self.hid_proj, self.out_proj]
        return sum(p.numel() for mod in mods for p in mod.parameters())
