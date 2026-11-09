"""Shared helpers for the POPGym-native RL track (reference venv /tmp/popgym_ref only).

``popgym_config`` is ``ref_ppo/run_ref.py::build_config`` (POPGym's own ``baselines/ppo.py`` config dict, ported to
ray 2.59 names) plus the per-arm ``custom_model`` / ``custom_model_config`` entries.  Every arm gets *identical* PPO
hyperparameters, preprocessor, actor, critic, ``hidden_size`` and batch layout; only ``custom_model`` (and the SSP
model's own ``MODEL_CONFIG`` keys) differ.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SSP_DIR = os.path.abspath(os.path.join(HERE, ".."))
REF_DIR = os.path.join(SSP_DIR, "ref_ppo")
for p in (SSP_DIR, REF_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

import run_ref  # noqa: E402  (ref_ppo/run_ref.py; imports ray, popgym)
from popgym.baselines.ray_models.ray_fwp import FastWeightProgrammer, DeepFastWeightProgrammer  # noqa: E402
from popgym.baselines.ray_models.ray_gru import GRU  # noqa: E402
from popgym.baselines.ray_models.ray_lstm import LSTM  # noqa: E402
from popgym.baselines.ray_models.ray_linear_attention import LinearAttention  # noqa: E402
from ssp_model import SSPModel  # noqa: E402
from ssp_tied_model import SSPTiedModel, SSPTiedLagBankModel, SSPScaledUntiedModel  # noqa: E402

ARMS = {
    "gru": (GRU, {}),
    "fwp": (FastWeightProgrammer, {}),
    "ssp": (SSPModel, {"aux_coeff": 0.0}),
    "ssp_aux": (SSPModel, {"aux_coeff": 0.1}),
    # tied relative addressing round (REPORT_TIED.md); lag = LAG_SCALE * head output, lag_init 0 = "read now", no task constant
    "tied": (SSPTiedModel, {"aux_coeff": 0.0}),
    "tied_aux": (SSPTiedModel, {"aux_coeff": 0.1}),
    "tied_lag4": (SSPTiedModel, {"aux_coeff": 0.0, "lag_init": 4.0}),          # DIAGNOSTIC control: starts at the task's lag
    "untied_scaled": (SSPScaledUntiedModel, {"aux_coeff": 0.0}),               # ablation: untied head, same lag parameterisation
    # lag bank (handoffs/11 step 4): read at every lag 1..8, learned softmax selection -- selection instead of search,
    # so the dead zone does not apply.  One default bank for every task; the aux is the lag-aware form of the parent's.
    "tied_bank": (SSPTiedLagBankModel, {"aux_coeff": 0.0}),
    "tied_bank_aux": (SSPTiedLagBankModel, {"aux_coeff": 0.1}),
    # DIAGNOSTIC only (hand-set lag 4 = RepeatPrevious's k): checks the RLlib plumbing, never enters the comparison
    "ssp_oracle4": (SSPModel, {"aux_coeff": 0.0, "ssp_addr_bias": -4.0}),
}


def popgym_config(env_id, arm, workers, envs_per_worker, bptt_cutoff=1024, num_minibatch=8, seed=None):
    """POPGym's PPO config dict (as in run_ref.build_config) with the arm's model plugged in."""
    model_cls, extra = ARMS[arm]
    cfg = run_ref.build_config(env_id, model_cls, bptt_cutoff, workers, envs_per_worker, num_minibatch, None)
    cfg["model"]["custom_model_config"].update(extra)
    # API drift that run_ref.py did not catch: ray 2.59's PPOConfig silently ignores the old key ``sgd_minibatch_size``
    # (update_from_dict sets an unused attribute, ``minibatch_size`` stays at its default 128 timesteps).  POPGym's
    # ppo.py means 8 * bptt_cutoff = 8192 timesteps per SGD minibatch (4 minibatches x 30 epochs per iteration).
    cfg["minibatch_size"] = cfg.pop("sgd_minibatch_size")
    if seed is not None:
        cfg["seed"] = int(seed)
        if model_cls is SSPModel:
            cfg["model"]["custom_model_config"]["ssp_seed"] = int(seed)
    return cfg


def n_params(module, skip_prefixes=()):
    return sum(p.numel() for n, p in module.named_parameters() if not n.startswith(skip_prefixes))
