"""
Unit tests for the pure helpers of scripts/run_deployment.py (no trained policies needed).

Covered: int8 per-row quantization error bound, fp16 round-trip helper, size accounting on a hand-built
QNetwork (dense and sparse first layer), MAC counting (linear vs [64,64]), latency percentile helper,
quantized state_dict does not mutate its input, and the paired greedy rollout on a dummy policy.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, ".."))

import run_deployment as rd  # noqa: E402
from rl.networks.dqnBasic import QNetwork  # noqa: E402


def test_int8_per_row_roundtrip_error_bound():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(7, 13)).astype(np.float32) * rng.uniform(0.01, 5.0, size=(7, 1)).astype(np.float32)
    q, scale = rd.quantize_int8_per_row(w, out_axis=0)
    assert q.dtype == np.int8 and scale.shape == (7, 1)
    assert np.abs(q).max() <= 127
    err = np.abs(rd.dequantize_int8(q, scale) - w)
    # every element is within half a quantization step of its own ROW's scale
    assert np.all(err <= scale / 2 + 1e-6 * np.abs(w).max())
    # the largest element of each row is represented (almost) exactly
    assert np.allclose(np.abs(rd.dequantize_int8(q, scale)).max(axis=1), np.abs(w).max(axis=1), rtol=1e-5)


def test_int8_per_column_for_embedding_table_and_zero_rows():
    w = np.zeros((5, 3), dtype=np.float32)
    w[:, 1] = [1.0, -2.0, 0.5, 0.0, 2.0]
    q, scale = rd.quantize_int8_per_row(w, out_axis=1)
    assert scale.shape == (1, 3)
    deq = rd.dequantize_int8(q, scale)
    assert np.all(deq[:, 0] == 0) and np.all(deq[:, 2] == 0)  # all-zero outputs stay zero (no NaN)
    assert np.allclose(deq[:, 1], w[:, 1], atol=2.0 / 127 / 2 + 1e-6)


def test_fp16_roundtrip():
    x = np.array([0.0, 1.0, -2.5, 0.5, 65504.0], dtype=np.float64)
    assert np.array_equal(rd.fp16_roundtrip(x), x)  # exactly representable
    assert rd.fp16_roundtrip(x).dtype == np.float64  # dtype preserved (SSP phase matrix is float64)
    rng = np.random.default_rng(1)
    y = rng.uniform(0.1, 100.0, size=1000).astype(np.float32)
    rel = np.abs(rd.fp16_roundtrip(y) - y) / y
    assert rel.max() <= 2.0 ** -11 + 1e-9
    assert not np.array_equal(rd.fp16_roundtrip(y), y)


def test_quantize_state_dict_does_not_mutate_input():
    torch.manual_seed(0)
    net = QNetwork(obs_dim=6, n_actions=2, hidden_dims=[8])
    before = {k: v.clone() for k, v in net.state_dict().items()}
    for mode in ("fp16", "int8"):
        out = rd.quantize_state_dict(net.state_dict(), mode)
        assert set(out) == set(before)
        for k, v in net.state_dict().items():
            assert torch.equal(v, before[k])
        # biases are untouched by int8
        if mode == "int8":
            for k in out:
                if k.endswith("bias"):
                    assert torch.equal(out[k], before[k])
        assert any(not torch.equal(out[k], before[k]) for k in out)
    net.load_state_dict(rd.quantize_state_dict(net.state_dict(), "int8"))  # loadable


def test_network_size_dense_tiny():
    net = QNetwork(obs_dim=3, n_actions=2, hidden_dims=[4])  # Linear(3,4), Linear(4,2)
    s = rd.network_size(net)
    assert s["n_params"] == 3 * 4 + 4 + 4 * 2 + 2 == 26
    assert s["fp32"] == 26 * 4 and s["fp16"] == 26 * 2
    # int8: matrices 12 + 8 bytes, one fp32 scale per output unit (4 + 2), biases fp32 (4 + 2)
    assert s["int8"] == (12 + 8) + 4 * (4 + 2) + 4 * (4 + 2) == 68
    assert s["tensors"]["layers.0.weight"]["shape"] == [4, 3]
    assert s["tensors"]["layers.0.bias"]["kind"] == "bias"


def test_network_size_sparse_counts_embedding_table():
    net = QNetwork(obs_dim=100, n_actions=2, hidden_dims=[8], sparse=True)
    s = rd.network_size(net)
    emb = s["tensors"]["emb.weight"]
    assert emb["shape"] == [100, 8] and emb["fp32"] == 100 * 8 * 4
    assert emb["int8"] == 100 * 8 + 4 * 8  # per output COLUMN scale
    assert s["tensors"]["emb_bias"]["n"] == 8
    assert s["n_params"] == 100 * 8 + 8 + 8 * 2 + 2


def test_network_macs_linear_vs_deep():
    lin = QNetwork(obs_dim=100, n_actions=2, hidden_dims=None)
    assert rd.network_macs(lin)["total"] == 200
    deep = QNetwork(obs_dim=100, n_actions=2, hidden_dims=[64, 64])
    assert rd.network_macs(deep)["total"] == 100 * 64 + 64 * 64 + 64 * 2 == 10624
    sparse_deep = QNetwork(obs_dim=5000, n_actions=2, hidden_dims=[64, 64], sparse=True)
    m = rd.network_macs(sparse_deep, n_active=8)
    assert m["first_layer_sparse_accumulates"] == 8 * 64
    assert m["total"] == 8 * 64 + 64 * 64 + 64 * 2
    sparse_lin = QNetwork(obs_dim=5000, n_actions=2, hidden_dims=None, sparse=True)
    assert rd.network_macs(sparse_lin, n_active=1)["total"] == 2
    with pytest.raises(AssertionError):
        rd.network_macs(sparse_lin)


def test_latency_stats_synthetic():
    ns = list(range(1000, 101000, 1000))  # 1..100 us
    s = rd.latency_stats(ns)
    assert s["n"] == 100
    assert s["median_us"] == pytest.approx(50.5)
    assert s["p95_us"] == pytest.approx(np.percentile(np.arange(1, 101), 95))
    assert s["p99_us"] == pytest.approx(np.percentile(np.arange(1, 101), 99))
    assert s["median_us"] <= s["p95_us"] <= s["p99_us"]
    one = rd.latency_stats([2500])
    assert one["median_us"] == one["p99_us"] == pytest.approx(2.5)


class _DummyPolicy:
    """Always pushes the cart right; counts clones so the per-episode clone contract is checkable."""

    n_clones = 0

    def act(self, obs):
        return 1

    def clone(self):
        _DummyPolicy.n_clones += 1
        return _DummyPolicy()

    def reset(self):
        pass


def test_greedy_returns_paired_and_one_clone_per_episode():
    _DummyPolicy.n_clones = 0
    base = _DummyPolicy()
    a = rd.greedy_returns(base.clone, env_seed=123, n_episodes=12)
    assert _DummyPolicy.n_clones == 12
    b = rd.greedy_returns(base.clone, env_seed=123, n_episodes=12)
    assert a == b and len(a) == 12 and all(0 < r <= 500 for r in a)
    c = rd.greedy_returns(base.clone, env_seed=124, n_episodes=12)
    assert c != a  # different stream


# ---------------------------------------------------------------------------------------------
# Real saved policies of every representation kind, built by hand (no training), pushed through
# save_policy -> load_policy -> size_report / operation_counts / make_variant / quant_worker / time_calls.
# ---------------------------------------------------------------------------------------------
import types  # noqa: E402

from rl.policy_io import load_policy, save_policy  # noqa: E402
from rl.representations.onehot import OneHotRepCP  # noqa: E402
from rl.representations.ssp import SSPRep  # noqa: E402
from rl.representations.tilecoding import TileCodingRep  # noqa: E402

_LOW = np.array([-2.4, -3.0, -0.21, -3.5])
_HIGH = np.array([2.4, 3.0, 0.21, 3.5])


def _states(n=16, seed=0):
    return np.random.default_rng(seed).uniform(_LOW * 0.5, _HIGH * 0.5, size=(n, 4)).astype(np.float32)


def _save_load(tmp_path, rep, net, params=None, name="p.pt"):
    dqn = types.SimpleNamespace(q_net=net, n_actions=2, avg_reward=1.0)
    path = str(tmp_path / name)
    save_policy(path, dqn, rep, params or {}, seed=1)
    return path, load_policy(path)


def _tile_policy(tmp_path, hidden=None):
    torch.manual_seed(0)
    shim = types.SimpleNamespace(observation_space=types.SimpleNamespace(high=_HIGH.copy(), low=_LOW.copy()))
    rep = TileCodingRep(shim, num_tilings=4, tiles_per_dim=(4, 4, 4, 4), iht_size=256,
                        bounds_low=_LOW, bounds_high=_HIGH)
    for s in _states(20):
        rep.map_indices(s)  # allocate some IHT entries, as training/eval does
    net = QNetwork(rep.size_out, 2, hidden, sparse=True)
    return _save_load(tmp_path, rep, net)


def _discrete_policy(tmp_path, hidden=None):
    torch.manual_seed(0)
    rep = OneHotRepCP((3, 3, 3, 3), bounds_low=_LOW.copy(), bounds_high=_HIGH.copy())
    net = QNetwork(rep.size_out, 2, hidden, sparse=True)
    return _save_load(tmp_path, rep, net)


def _ssp_policy(tmp_path, hidden=None):
    torch.manual_seed(0)
    rep = SSPRep(domain_dim=4, length_scale=0.5, n_rotates=2)
    net = QNetwork(rep.size_out, 2, hidden, sparse=False)
    return _save_load(tmp_path, rep, net, params={"n_rotates": 2})


@pytest.mark.parametrize("builder,kind", [(_tile_policy, "tile"), (_discrete_policy, "discrete"),
                                          (_ssp_policy, "ssp")])
@pytest.mark.parametrize("hidden", [None, [8]])
def test_size_report_every_rep_kind(tmp_path, builder, kind, hidden):
    _, pol = builder(tmp_path, hidden)
    assert pol.rep_kind == kind
    r = rd.size_report(pol)
    for p in rd.PRECISIONS:
        assert r["total"][p] == r["network"][p] + r["encoder"][p]
        assert r["encoder"][p] == sum(it[p] for it in r["encoder"]["items"].values())
    assert r["network"]["fp32"] == 4 * r["n_params"] == 4 * pol.n_params
    assert r["dim"] == pol.dim and r["n_active"] >= 1
    assert r["total"]["fp16"] <= r["total"]["fp32"] and r["total"]["int8"] <= r["total"]["fp32"]
    json.dumps(r)  # the report must be JSON serialisable as is
    if kind == "tile":
        n = len(pol.rep.iht.dictionary)
        assert n > 0
        assert r["encoder"]["iht_n_entries"] == n and r["encoder"]["items"]["iht_dictionary"]["n_entries"] == n
        assert r["encoder"]["iht_size"] == 256
        dense = r["encoder"]["iht_dense_alternative_bytes"]
        assert dense == 256 * 6 * 4 and dense > r["encoder"]["items"]["iht_dictionary"]["fp32"]
        # the dense alternative is reported separately and never part of `total`
        alt = r["total_with_dense_iht_alternative"]
        assert alt["fp32"] == r["network"]["fp32"] + r["encoder"]["items"]["bounds_and_grid"]["fp32"] + dense
        assert alt["fp32"] > r["total"]["fp32"]
        assert r["n_active"] == 4
    elif kind == "ssp":
        pm = r["encoder"]["items"]["phase_matrix"]
        assert pm["shape"] == [pol.dim, 4] and pm["stored_dtype"] == "float64"
        assert pm["fp32"] == 4 * pol.dim * 4 and pm["fp16"] == 2 * pol.dim * 4
        assert "total_with_dense_iht_alternative" not in r
        assert r["n_active"] == pol.dim
    else:
        assert r["encoder"]["items"]["bin_edges"]["n"] == 4 * (3 + 1)
        assert r["n_active"] == 1 and pol.dim == 81


@pytest.mark.parametrize("builder", [_tile_policy, _discrete_policy, _ssp_policy])
def test_operation_counts_every_rep_kind(tmp_path, builder):
    _, pol = builder(tmp_path, [8])
    oc = rd.operation_counts(pol)
    assert oc["network_macs"] == oc["network"]["total"] > 0
    assert oc["total_ops"] == oc["encoder_total_ops"] + oc["network_macs"]
    if pol.rep_kind == "ssp":
        assert oc["encoder"]["phase_matmul_macs"] == (pol.dim // 2) * 4
        assert oc["encoder_total_ops"] == (pol.dim // 2) * 4 + pol.dim
        assert oc["network"]["first_layer_sparse_accumulates"] == 0
    elif pol.rep_kind == "tile":
        assert oc["encoder"]["hash_lookups"] == 4
        assert oc["network"]["first_layer_sparse_accumulates"] == 4 * 8
    else:
        assert oc["network"]["first_layer_sparse_accumulates"] == 1 * 8
    json.dumps(oc)


@pytest.mark.parametrize("builder", [_tile_policy, _discrete_policy, _ssp_policy])
def test_variants_do_not_mutate_shared_state_and_change_function(tmp_path, builder):
    _, pol = builder(tmp_path, [8])
    states = _states(32, seed=3)
    ref = pol.clone().q_values_batch(states)
    net_before = {k: v.clone() for k, v in pol.net.state_dict().items()}
    pm_before = np.array(pol.rep._encoder.phase_matrix) if pol.rep_kind == "ssp" else None
    variants = ["fp32", "fp16_weights", "int8_weights"] + (["fp16_encoder"] if pol.rep_kind == "ssp" else [])
    for v in variants:
        var = rd.make_variant(pol, v)
        if v == "fp32":
            assert np.array_equal(var.q_values_batch(states), ref)
        else:
            assert var.net is not pol.net or v == "fp16_encoder"
            q = var.clone().q_values_batch(states)  # clone() must keep the override
            assert q.shape == ref.shape and np.all(np.isfinite(q))
            assert np.abs(q - ref).max() < 1.0  # a mild perturbation, not a different function
            if v != "fp16_encoder":
                assert not np.array_equal(q, ref)
    # the original (and its shared net / phase matrix) is untouched
    for k, v in pol.net.state_dict().items():
        assert torch.equal(v, net_before[k])
    assert np.array_equal(pol.clone().q_values_batch(states), ref)
    if pm_before is not None:
        assert np.array_equal(np.asarray(pol.rep._encoder.phase_matrix), pm_before)
        assert pm_before.dtype == np.float64
    if pol.rep_kind != "ssp":
        with pytest.raises(ValueError):
            rd.make_variant(pol, "fp16_encoder")


@pytest.mark.parametrize("builder", [_tile_policy, _discrete_policy, _ssp_policy])
def test_quant_worker_and_latency_on_real_policies(tmp_path, builder):
    path, pol = builder(tmp_path, [8])
    probe = tmp_path / "probe.npz"
    np.savez(probe, states=_states(256, seed=5), noise_sigma=np.ones(4))
    job = {"seed_idx": 0, "seed": 3000, "path": path, "n_episodes": 2, "probe_path": str(probe)}
    rec = rd.quant_worker(job)
    assert "error" not in rec, rec.get("error")
    expected = {"fp32", "fp16_weights", "int8_weights"} | ({"fp16_encoder"} if pol.rep_kind == "ssp" else set())
    assert set(rec["variants"]) == expected
    for v, e in rec["variants"].items():
        assert len(e["returns"]) == 2 and e["mean_return"] > 0
    assert rec["variants"]["fp32"]["returns"] == rd.greedy_returns(pol.clone, 3000 + rd.QUANT_ENV_SEED_OFFSET, 2)
    agg = rd.aggregate_quant([rec, {**rec, "seed_idx": 1, "seed": 3001}])
    assert agg["fp32"]["retained"] == pytest.approx(1.0) and agg["fp32"]["n_seeds"] == 2
    json.dumps(agg)
    json.dumps(rec)

    t = rd.time_calls(pol, np.load(probe)["states"], n_calls=20, warmup=3)
    assert set(t) == {"encode", "q_from_phi", "act"} and t["act"]["n"] == 20
    assert t["act"]["median_us"] > 0
    json.dumps(t)
