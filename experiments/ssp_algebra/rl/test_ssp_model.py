"""Contract tests for rl/ssp_model.py.  Run in the reference venv:

    cd experiments/ssp_algebra/rl && OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python test_ssp_model.py

(pytest is not installed in that venv; every ``test_*`` function is also pytest-collectable.)
"""

import sys
import time

import gymnasium as gym
import numpy as np
import torch

import common  # noqa: F401  (sets sys.path)
from common import ARMS, popgym_config
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.sample_batch import SampleBatch

import feats
import rl_memory
from ssp_model import SSPModel, TAU

OBS = gym.spaces.Tuple((gym.spaces.Discrete(4), gym.spaces.Discrete(4), gym.spaces.Discrete(2)))   # RepeatPrevious
ACT = gym.spaces.Discrete(4)


def make(arm="ssp", **over):
    cfg = popgym_config("popgym-RepeatPreviousEasy-v0", arm, workers=1, envs_per_worker=1)
    cfg["model"]["custom_model_config"].update(over)
    pre = get_preprocessor(OBS)(OBS)
    m = cfg["model"]["custom_model"](pre.observation_space, ACT, 4, cfg["model"], "m",
                                     **cfg["model"]["custom_model_config"])
    return m


def onehot_obs(tok, prev=None):
    """tok: (B,T) long -> flat obs [B*T, 10] exactly as RLlib's one-hot flattening of the Tuple."""
    B, T = tok.shape
    prev = torch.zeros_like(tok) if prev is None else prev
    o = torch.zeros(B, T, 10)
    o.scatter_(2, tok.unsqueeze(-1), 1.0)
    o.scatter_(2, (4 + prev).unsqueeze(-1), 1.0)
    o[:, :, 8] = 1.0
    return o.reshape(B * T, 10)


def inp(tok):
    o = onehot_obs(tok)
    return {"obs": o, "obs_flat": o}


def init_state(m, B):
    return [s.unsqueeze(0).repeat(B, *([1] * s.dim())).clone() for s in m.get_initial_state()]


def run(m, tok, state=None, seq_lens=None):
    B, T = tok.shape
    state = init_state(m, B) if state is None else state
    seq_lens = torch.full((B,), T, dtype=torch.int32) if seq_lens is None else seq_lens
    logits, st = m(inp(tok), state, seq_lens)
    return logits.reshape(B, T, -1), st


def toks(B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 4, (B, T), generator=g)


# --- 1. shapes and state size ------------------------------------------------------------------------
def test_forward_shapes_and_state_size():
    m = make()
    gru = make("gru")
    B, T = 3, 7
    logits, st = run(m, toks(B, T))
    assert logits.shape == (B, T, 4)
    assert m.value_function().shape == (B * T,)
    assert [tuple(s.shape) for s in st] == [(B, 1, 256), (B, 1)]
    assert st[1].flatten().tolist() == [T] * B                       # POPGym's own time counter
    # state floats match the POPGym GRU at hidden_size 256 (state excluding the time counter)
    n_ssp = sum(int(np.prod(s.shape)) for s in m.initial_state())
    n_gru = sum(int(np.prod(s.shape)) for s in gru.initial_state())
    assert n_ssp == n_gru == 256 == m.state_floats()


# --- 2. state init / reset ---------------------------------------------------------------------------
def test_state_init_zero_and_reset_equals_fresh():
    m = make()
    init = m.get_initial_state()
    assert all(float(s.abs().sum()) == 0.0 for s in init)
    A, Bk = toks(2, 6, 1), toks(2, 6, 2)
    la, sta = run(m, A)
    assert float(sta[0].abs().sum()) > 0.0                            # state is non-trivial after an episode
    # RLlib starts a new episode from get_initial_state(): the result must equal running it from scratch
    lb_fresh, _ = run(m, Bk)
    lb_reset, _ = run(m, Bk, state=init_state(m, 2))
    assert torch.allclose(lb_fresh, lb_reset, atol=1e-6)
    # carrying A's state instead (no reset) leaks A into B
    lb_leak, _ = run(m, Bk, state=sta)
    assert not torch.allclose(lb_fresh, lb_leak, atol=1e-4)


def test_chunked_equals_whole_sequence():
    """RLlib splits long episodes into max_seq_len chunks and carries the state: chunking must not change outputs."""
    m = make()
    tk = toks(2, 10, 3)
    whole, _ = run(m, tk)
    l1, st = run(m, tk[:, :4])
    l2, _ = run(m, tk[:, 4:], state=st)
    assert torch.allclose(whole[:, :4], l1, atol=1e-5)
    assert torch.allclose(whole[:, 4:], l2, atol=1e-5), float((whole[:, 4:] - l2).abs().max())


def test_padding_does_not_change_state():
    """Rows shorter than T (seq_lens) must return the state after their last valid step, not after the padding."""
    m = make()
    tk = toks(2, 8, 4)
    short = torch.tensor([5, 8], dtype=torch.int32)
    _, st_pad = run(m, tk, seq_lens=short)
    _, st_ref = run(m, tk[:1, :5])
    assert torch.allclose(st_pad[0][0], st_ref[0][0], atol=1e-5)
    assert st_pad[1].flatten().tolist() == [5, 8]


# --- 3. equals the stepwise rl_memory.SSPMemory --------------------------------------------------------
def test_vectorised_core_equals_stepwise_ssp_memory():
    m = make()
    B, T = 3, 12
    tk = toks(B, T, 5)
    with torch.no_grad():
        m.mem.addr_head.bias.fill_(-2.0)                               # a non-trivial read address
        m.mem.addr_head.weight[:, :m.mem.hidden].normal_(0, 0.05)
    z = torch.randn(B, T, 128)
    state0 = torch.zeros(B, 256)
    with torch.no_grad():
        scores, M, addr, _ = m.memory_core(z, tk, state0, torch.zeros(B, dtype=torch.long),
                                           torch.full((B,), T))
        ref = rl_memory.SSPMemory(4, dim=128, hidden=32, seed=0)
        ref.load_state_dict({k: v for k, v in m.mem.state_dict().items() if k != "mem"}, strict=False)
        ref.reset(B)
        h = m.hid_proj(z)
        for t in range(T):
            tt = torch.full((B,), float(t))
            ref.write(tk[:, t], tt)
            s_ref, a_ref = ref.read_from(h[:, t], tt)
            assert torch.allclose(scores[:, t], s_ref, atol=1e-4), t
            assert torch.allclose(addr[:, t], a_ref, atol=1e-4), t
        assert torch.allclose(M[:, -1], ref.mem, atol=1e-4)


def test_aux_loss_equals_ssp_memory_aux_loss():
    m = make("ssp_aux")
    B, T = 3, 9
    tk = toks(B, T, 6)
    st = init_state(m, B)
    logits, _ = m(inp(tk), st, torch.full((B,), T, dtype=torch.int32))
    aux = float(m._aux)
    with torch.no_grad():
        z = m.pre(m.feature_map(onehot_obs(tk).reshape(B, T, 10)))
        h = m.hid_proj(z)
        ref = rl_memory.SSPMemory(4, dim=128, hidden=32, seed=0)
        ref.load_state_dict({k: v for k, v in m.mem.state_dict().items() if k != "mem"}, strict=False)
        ref.reset(B)
        vals = []
        for t in range(T):
            tt = torch.full((B,), float(t))
            ref.write(tk[:, t], tt)
            vals.append(float(ref.aux_loss(tk[:, t], h[:, t], tt)))
    assert abs(aux - float(np.mean(vals))) < 1e-4, (aux, np.mean(vals))


# --- 4. gradients reach the address parameters -----------------------------------------------------
def test_gradient_reaches_address_parameters_from_policy_loss():
    m = make()
    tk = toks(4, 10, 7)
    logits, _ = run(m, tk)
    loss = logits.pow(2).mean() + m.value_function().pow(2).mean()
    loss.backward()
    for name in ("addr_head.weight", "addr_head.bias"):
        g = dict(m.mem.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0.0, name
    for name in ("w_time", "lam_raw"):
        g = dict(m.mem.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0.0, name


def test_aux_loss_flag_off_and_on_through_custom_loss():
    off, on = make("ssp"), make("ssp_aux")
    tk = toks(3, 8, 8)
    run(off, tk)
    run(on, tk)
    base = [torch.tensor(1.0, requires_grad=True)]
    assert off.custom_loss(base, {}) is base                           # off: policy loss untouched
    out = on.custom_loss(base, {})
    assert float(out[0]) > 1.0 and isinstance(out, list)
    on.zero_grad()
    out[0].backward()
    g = on.mem.addr_head.weight.grad
    assert g is not None and float(g.abs().sum()) > 0.0                # aux gradient reaches the address head
    assert "ssp_aux_loss" in on.metrics()


# --- 5. no-op behaviour / memory is used ------------------------------------------------------------------
def test_ablated_memory_is_a_feedforward_policy():
    m = make(ablate_memory=True)
    a = toks(2, 6, 9)
    b = a.clone()
    b[:, :-1] = (b[:, :-1] + 1) % 4                                    # change every earlier token, keep the current
    la, _ = run(m, a)
    lb, _ = run(m, b)
    assert torch.allclose(la[:, -1], lb[:, -1], atol=1e-6)             # output depends on the current obs only
    on = make()
    on.load_state_dict(m.state_dict(), strict=False)
    on.ablate_memory = False
    lon_a, _ = run(on, a)
    lon_b, _ = run(on, b)
    assert not torch.allclose(lon_a[:, -1], lon_b[:, -1], atol=1e-6)   # with memory the history is visible


def test_memory_recalls_lagged_token_at_the_right_address():
    """With the head set to lag k the read-out argmax is the token k steps back (what the policy has to learn)."""
    m = make()
    B, T, k = 4, 24, 4
    tk = toks(B, T, 10)
    with torch.no_grad():
        m.mem.addr_head.weight.zero_()
        m.mem.addr_head.weight[0, m.mem.hidden] = float(m.mem.w_time[0])      # address = t
        m.mem.addr_head.bias.fill_(-float(k))
        z = torch.randn(B, T, 128)
        scores, _, _, _ = m.memory_core(z, tk, torch.zeros(B, 256), torch.zeros(B, dtype=torch.long),
                                        torch.full((B,), T))
    acc = (scores[:, k:].argmax(-1) == tk[:, :-k]).float().mean()
    assert float(acc) > 0.95, float(acc)


# --- 6. per-step cost ---------------------------------------------------------------------------------------
def _step_time(m, t_start, reps=40):
    B = 8
    st = init_state(m, B)
    st[-1] = torch.full((B, 1), t_start, dtype=torch.long)
    obs = inp(toks(B, 1, 11))
    sl = torch.ones(B, dtype=torch.int32)
    with torch.no_grad():
        m(obs, [s.clone() for s in st], sl)
        t0 = time.perf_counter()
        for _ in range(reps):
            m(obs, [s.clone() for s in st], sl)
    return (time.perf_counter() - t0) / reps


def test_per_step_cost_is_constant_and_bounded():
    ssp, gru = make(), make("gru")
    early, late = _step_time(ssp, 0), _step_time(ssp, 5000)
    g = _step_time(gru, 0)
    print(f"per-step forward (B=8, T=1): ssp t=0 {early*1e3:.2f} ms, ssp t=5000 {late*1e3:.2f} ms, gru {g*1e3:.2f} ms")
    assert late < 2.0 * early + 1e-3                                   # no growth with the episode position
    assert early < 20e-3                                               # absolute bound on a laptop-class core


def test_state_does_not_grow():
    m = make()
    st = init_state(m, 2)
    for t in range(30):
        _, st = m(inp(toks(2, 1, t)), st, torch.ones(2, dtype=torch.int32))
        assert tuple(st[0].shape) == (2, 1, 256)


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for k, f in fns:
        try:
            f()
            print("PASS", k)
        except Exception as e:  # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print("FAIL", k, repr(e))
    print(f"{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
