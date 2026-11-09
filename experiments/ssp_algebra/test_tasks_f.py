"""Checks for the delay-with-distractors task (RUN_012 step 2; `tasks_f.DelayDistractTask`).

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_tasks_f.py

The task is only meaningful if the distractors are *answerable-never* and the alphabet stays fixed across splits (the
content family crashed twice by changing the alphabet, RUN_012 section 4).  What is checked:

  * determinism: the same rng gives identical tensors;
  * no masked target ever points at a distractor step, and every content step after `lag` content steps is masked;
  * the alphabet is identical in every split (``obs < n_items``, ``out_dim == n_items``, feature width constant);
  * a constant predictor is at chance, so the task has no leak (RUN_007's `index` leak is the precedent for checking);
  * a perfect "look back `lag` content steps" oracle answers every masked step, so the task is *solvable* and the
    mask is not vacuous.
"""

import numpy as np
import torch

import feats
import harness as H
import tasks_f


def _task(ep_len=32, p_dist=0.25):
    return tasks_f.DelayDistractTask(n_items=8, ep_len=ep_len, train_lags=(1, 2, 3, 4), p_dist=p_dist)


def test_determinism():
    task = _task()
    a = task.batch(8, task.ep_len, rng=np.random.default_rng(0))
    b = task.batch(8, task.ep_len, rng=np.random.default_rng(0))
    for k in ("obs", "target", "mask", "is_dist", "t"):
        assert torch.equal(a[k], b[k]), f"{k} is not rng-deterministic"
    print("  determinism: identical tensors for the same rng")


def test_no_answer_is_a_distractor_and_the_oracle_solves_it():
    """Two properties at once: the target is never drawn from a distractor step, and an oracle can answer."""
    task = _task()
    for p_dist in (0.25, 0.5):
        b = task.batch(16, task.ep_len, rng=np.random.default_rng(1), p_dist=p_dist)
        obs, mask, dist, lag = b["obs"], b["mask"], b["is_dist"], b["lag"]
        assert mask.any(), "the mask must not be vacuous"
        # an oracle that returns the symbol at the lag-th *content* step back must match the stored target everywhere
        ok = 0
        for bb in range(obs.shape[0]):
            content = []
            for t in range(obs.shape[1]):
                if not bool(dist[bb, t]):
                    l = int(lag[bb])
                    if len(content) >= l:
                        if bool(mask[bb, t]):
                            assert int(obs[bb, content[-l]]) == int(b["target"][bb, t]), "oracle disagrees with target"
                            ok += 1
                    content.append(t)
        assert ok == int(mask.sum()), f"oracle answered {ok} of {int(mask.sum())}"
        # no target may be taken from a distractor step: the answer positions are content by construction above, so
        # re-check the property directly by counting masked steps whose own symbol is a distractor
        assert not bool((mask & dist).any()), "no masked step may itself be a distractor"
        print(f"  p_dist={p_dist}: mask {int(mask.sum())} steps, oracle solves all, no target on a distractor")


def test_alphabet_is_fixed_across_splits():
    task = _task()
    for name, spec in H.splits_for("delay_distract", task).items():
        b = H.make_batch(task, spec, 16, np.random.default_rng(0))
        assert int(b["obs"].max()) < task.out_dim, f"{name}: obs outside the alphabet"
        if b["mask"].any():
            assert int(b["target"][b["mask"]].max()) < task.out_dim, f"{name}: target outside the alphabet"
        assert task.features(b).shape[-1] == task.in_dim, f"{name}: feature width changed"
        assert task.out_dim == task.n_items
        print(f"  {name:11s} T={b['obs'].shape[1]:3d} obs<{task.out_dim} target<{task.out_dim} feat={task.in_dim}")


def test_a_constant_predictor_is_at_chance():
    """No leak: the most frequent class must not beat chance by much (1/8 = 0.125 for 8 items)."""
    task = _task()
    b = task.batch(64, task.ep_len, rng=np.random.default_rng(2))
    tgt = b["target"][b["mask"]]
    counts = torch.bincount(tgt, minlength=task.out_dim).float()
    best = float(counts.max() / counts.sum())
    print(f"  best constant predictor: {best:.3f} (chance {1 / task.out_dim:.3f})")
    assert best < 0.30, f"a constant predictor scores {best:.3f}: the task leaks"

    # and a real no-memory arm (mlp) must sit near chance too
    import arms as A
    sp = {"T": task.ep_len}
    rng = np.random.default_rng(0)
    arm, _ = A.build_arm("mlp", task, budget=256, seed=0, hidden=32)
    accs = []
    with torch.no_grad():
        for _ in range(10):
            bb = H.make_batch(task, sp, 64, rng)
            _, acc = H.loss_and_acc(arm, task, bb, task.features(bb))
            accs.append(acc)
    print(f"  untrained mlp floor: {np.mean(accs):.3f}")
    assert np.mean(accs) < 0.30, "the no-memory floor is too high: the task leaks"


# --- the lag-free variant (handoffs/11 step 2b) ------------------------------------------------------
def _nolag(ep_len=32):
    return tasks_f.DelayNoLagTask(n_items=8, ep_len=ep_len, train_lags=(1, 2, 3, 4))


def test_delay_nolag_is_the_same_task_with_the_lag_hidden():
    """The observations, targets and mask are bit-identical to `delay_distract` at `p_dist=0`; only the *features*
    differ.  That is what makes it a fair mirror of `RepeatPrevious`: same task, no lag given."""
    no, yes = _nolag(), _task(p_dist=0.0)
    b_no = no.batch(8, 32, rng=np.random.default_rng(0))
    b_yes = yes.batch(8, 32, rng=np.random.default_rng(0))
    for k in ("obs", "target", "mask", "is_dist", "t"):
        assert torch.equal(b_no[k], b_yes[k]), f"{k} differs; the two variants must be the same task"
    # identical targets and mask prove the same *drawn* lag was used, since both depend on it
    assert float(b_no["lag"].abs().max()) == 0.0, "the drawn lag must be hidden from the arm (see the channel test)"

    f0 = no.features(b_no)
    mutated = {**b_no, "query": torch.full_like(b_no["query"], 0.75), "lag": b_no["lag"] + 1}
    assert torch.equal(f0, no.features(mutated)), "the lag-free features must not depend on the lag at all"
    # control: the ordinary variant's features DO move with the lag feature, so the test above can fail
    assert not torch.equal(yes.features(b_yes),
                           yes.features({**b_yes, "query": torch.full_like(b_yes["query"], 0.75)}))

    assert float(b_no["query"].abs().max()) == 0.0, "the lag feature must be zeroed, not merely ignored"
    assert no.in_dim == 8 + 1 + feats.TIME_DIM, no.in_dim
    assert yes.in_dim == 8 + 2 + feats.TIME_DIM, yes.in_dim
    print(f"  identical task; lag absent from {no.in_dim} features (lag variant has {yes.in_dim})")


def test_delay_nolag_alphabet_and_width_are_fixed_across_splits():
    task = _nolag()
    for name, spec in H.splits_for("delay_nolag", task).items():
        b = H.make_batch(task, spec, 16, np.random.default_rng(0))
        assert int(b["obs"].max()) < task.out_dim, f"{name}: obs outside the alphabet"
        if b["mask"].any():
            assert int(b["target"][b["mask"]].max()) < task.out_dim, f"{name}: target outside the alphabet"
        assert task.features(b).shape[-1] == task.in_dim, f"{name}: feature width changed"
        assert float(b["query"].abs().max()) == 0.0, f"{name}: the lag leaked back in"
        assert task.out_dim == task.n_items
        print(f"  {name:11s} T={b['obs'].shape[1]:3d} feat={task.in_dim} lag-hidden")


def test_delay_nolag_gives_the_arm_no_lag_channel():
    """Ask the *arm*, not the task: the read address must not move with the (now hidden) lag.

    Zeroing the feature column is not enough -- `SSPIndexLearned._read_addr` for the delay family reads
    ``batch["lag"]`` directly, so a variant that only blanked ``query`` would leave the lag fully visible and every
    arm would score ~1.000 while appearing to solve a harder problem (this happened on the first run of
    `jobs/p17_nolag.json`, which is why the test exists).
    """
    task = _nolag()
    b = H.make_batch(task, {"T": 32}, 8, np.random.default_rng(0))
    assert float(b["lag"].abs().max()) == 0.0, "the arm's own lag channel is not hidden"
    assert float(b["query"].abs().max()) == 0.0

    arm, _ = H.build_any("ssp_index_learned", task, 256, 0,
                         {"rung": 1, "init": "affine", "affine_eps": 0.03, "affine_offset_zero": True})
    with torch.no_grad():
        addr = arm._read_addr(b)
    assert float(addr.abs().max()) == 0.0, "the read address still receives the lag"

    # and the same for the fast-weight arm, which reads the lag through its own line.  Its `read_abs` form returns the
    # wanted item's absolute time, so a hidden lag means the address must be exactly `t` (the lag contributes nothing);
    # the historical lag-only form must return exactly zero.
    fwp, _ = H.build_any("fwp_delta", task, 256, 0, {"rule": "delta", "tied": True, "read_abs": True,
                                                     "beta_init": 4.0, "lr": 0.02, "hidden": 32, "lam_init": 0.99})
    with torch.no_grad():
        assert torch.equal(fwp._addr(b), b["t"].float()), "fwp's read address still carries the lag"
    fwp2, _ = H.build_any("fwp_delta", task, 256, 0, {"rule": "delta", "tied": True, "read_abs": False,
                                                      "beta_init": 4.0, "lr": 0.02, "hidden": 32, "lam_init": 0.99})
    with torch.no_grad():
        assert float(fwp2._addr(b).abs().max()) == 0.0, "fwp's lag-only address is not zero"
    print("  no lag channel: batch['lag'], batch['query'] and both arms' read addresses carry no lag")



def test_delay_nolag_has_no_leak_and_the_mask_is_not_vacuous():
    """The task is solvable under the true lag (checked on the lag-carrying variant, which is otherwise identical),
    but that lag is not recoverable from the input, so a no-memory arm must sit at chance."""
    ref = _task(p_dist=0.0)
    b = ref.batch(32, ref.ep_len, rng=np.random.default_rng(3))
    obs, mask, dist, lag = b["obs"], b["mask"], b["is_dist"], b["lag"]
    assert mask.any(), "the mask must not be vacuous"
    ok = 0
    for bb in range(obs.shape[0]):
        content = []
        for t in range(obs.shape[1]):
            if not bool(dist[bb, t]):
                l = int(lag[bb])
                if len(content) >= l:
                    if bool(mask[bb, t]):
                        assert int(obs[bb, content[-l]]) == int(b["target"][bb, t])
                        ok += 1
                content.append(t)
    assert ok == int(mask.sum())

    task = _nolag()
    import arms as A
    rng = np.random.default_rng(0)
    arm, _ = A.build_arm("mlp", task, budget=256, seed=0, hidden=32)
    accs = []
    with torch.no_grad():
        for _ in range(10):
            bb = H.make_batch(task, {"T": task.ep_len}, 64, rng)
            _, acc = H.loss_and_acc(arm, task, bb, task.features(bb))
            accs.append(acc)
    print(f"  untrained mlp floor: {np.mean(accs):.3f} (chance {1 / task.out_dim:.3f})")
    assert np.mean(accs) < 0.30, "the no-memory floor is too high: the lag leaked into the features"


def test_shapes_match_the_delay_interface():
    task = _task()
    b = H.make_batch(task, {"T": task.ep_len}, 8, np.random.default_rng(0))
    for k in ("obs", "t", "target", "mask", "query", "lag", "is_dist"):
        assert k in b, f"missing {k}"
    assert b["obs"].shape == b["t"].shape == b["target"].shape == b["mask"].shape == b["is_dist"].shape
    assert b["lag"].shape == (8,)
    assert b["loss"] == "ce"
    print(f"  interface keys present; shapes {tuple(b['obs'].shape)}")


if __name__ == "__main__":
    for fn in (test_determinism, test_no_answer_is_a_distractor_and_the_oracle_solves_it,
               test_alphabet_is_fixed_across_splits, test_a_constant_predictor_is_at_chance,
               test_shapes_match_the_delay_interface,
               test_delay_nolag_is_the_same_task_with_the_lag_hidden,
               test_delay_nolag_alphabet_and_width_are_fixed_across_splits,
               test_delay_nolag_gives_the_arm_no_lag_channel,
               test_delay_nolag_has_no_leak_and_the_mask_is_not_vacuous):
        print(f"\n{fn.__name__}")
        fn()
    print("\ntest_tasks_f: all checks passed")
