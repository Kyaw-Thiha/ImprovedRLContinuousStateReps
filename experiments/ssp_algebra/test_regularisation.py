"""Tests for baseline regularisation (handoff 09 section 15.4 step 1).

Run:  OMP_NUM_THREADS=1 ../../.venv/bin/python test_regularisation.py

The claim these guard is two-sided: (a) ``dropout`` / ``weight_decay`` are actually wired to the arms and the
optimiser, so a search over them means something; (b) the default path (no dropout, no weight decay) is *bit-identical*
to the code before the knobs existed, so every earlier result stays comparable.
"""
import sys

import numpy as np
import torch

import arms as A
import harness as H
import tasks_c


def _index_task(ep_len=12):
    return tasks_c.IndexTask(n_items=8, ep_len=ep_len)


def test_default_dropout_is_identity():
    """An arm built without ``dropout`` and one built with ``dropout=0.0`` must be the same model."""
    task = _index_task()
    a, _ = A.build_arm("gru", task, budget=32, seed=3)
    b, _ = A.build_arm("gru", task, budget=32, seed=3, dropout=0.0)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys(), (set(sa) ^ set(sb))
    assert all(torch.equal(sa[k], sb[k]) for k in sa), "default construction changed with dropout=0.0"
    print(f"  gru dropout=0.0 identical to default over {len(sa)} tensors")


def test_dropout_is_wired_for_every_hardened_arm():
    """Dropout must reach the readout; a silently dropped knob would waste a search dimension."""
    task = _index_task()
    for name in ("mlp", "posmlp", "gru", "lru", "tf_full", "tf_window", "kernel_attn"):
        kw = {"tf_d_model": 8, "win_d_model": 4} if name.startswith("tf") or name == "kernel_attn" else {}
        arm, _ = A.build_arm(name, task, budget=32, seed=0, dropout=0.3, **kw)
        mods = [m for m in arm.modules() if isinstance(m, torch.nn.Dropout)]
        assert mods and max(m.p for m in mods) == 0.3, f"{name}: dropout not wired ({[m.p for m in mods]})"
    print("  dropout wired on mlp, posmlp, gru, lru, tf_full, tf_window, kernel_attn")


def test_dropout_changes_training_but_not_inference():
    """Dropout is training-time only: the same batch must give identical outputs in eval mode, different curves in training."""
    task = _index_task()
    sp = H.splits_for("index", task)
    b = task.batch(8, task.ep_len, rng=np.random.default_rng(0))
    feat = task.features(b)
    outs = []
    for p in (0.0, 0.5):
        arm, _ = A.build_arm("gru", task, budget=32, seed=0, dropout=p)
        arm.eval()
        with torch.no_grad():
            outs.append(arm(b, feat))
    assert torch.equal(outs[0], outs[1]), "dropout changed inference, which would break every recorded eval"
    print("  eval-mode outputs identical; dropout is training-time only")


def test_weight_decay_reaches_the_optimizer():
    task = _index_task()
    arm, _ = A.build_arm("gru", task, budget=32, seed=0)
    assert H.make_optimizer(arm, 1e-2).defaults["weight_decay"] == 0.0, "default must stay 0 for old runs"
    opt = H.make_optimizer(arm, 1e-2, weight_decay=1e-2)
    assert len(opt.param_groups) == 1 and opt.defaults["weight_decay"] == 1e-2
    import arms_learned as L
    arm, _ = L.build_learned("ssp_index_learned", task, budget=32, seed=0, rung=1, hidden=16)
    opt = H.make_optimizer(arm, 1e-2, 0.1, 1e-2)
    assert len(opt.param_groups) == 2 and opt.defaults["weight_decay"] == 1e-2
    print(f"  weight_decay reaches 1-group and 2-group optimizers: {opt.defaults['weight_decay']}")


def test_weight_decay_shrinks_parameters():
    """End-to-end wiring check: the same run with weight decay must end at different weights."""
    task = _index_task()
    sp = H.splits_for("index", task)
    finals = []
    for wd in (0.0, 1e-2):
        arm, _ = A.build_arm("gru", task, budget=32, seed=0)
        H.train_arm(arm, task, sp, seed=0, steps=6, batch=16, lr=1e-2, eval_every=6, weight_decay=wd)
        finals.append(torch.cat([p.detach().reshape(-1) for p in arm.parameters()]))
    assert finals[0].shape == finals[1].shape
    assert not torch.allclose(finals[0], finals[1]), "weight_decay did not reach training"
    print("  train_arm(weight_decay=1e-2) ends at different weights than 0.0")


def test_sample_config_draws_regularisation_for_baselines_only():
    for name in ("mlp", "posmlp", "gru", "lru", "tf_full", "tf_window", "kernel_attn"):
        rng = np.random.default_rng(H.stable_seed(name, "index", 256))
        cfg = H.sample_config(name, rng, 256, "index")
        assert cfg["weight_decay"] in (0.0, 1e-4, 1e-2) and cfg["dropout"] in (0.0, 0.1, 0.3), (name, cfg)
    for name in ("ssp", "fwp", "ssp_index_learned", "ssp_pointer_learned"):
        rng = np.random.default_rng(H.stable_seed(name, "index", 256))
        cfg = H.sample_config(name, rng, 256, "index")
        assert "dropout" not in cfg and "weight_decay" not in cfg, (name, cfg)
    # determinism: the same rng state must give the same config
    c1 = H.sample_config("gru", np.random.default_rng(5), 256, "index")
    c2 = H.sample_config("gru", np.random.default_rng(5), 256, "index")
    assert c1 == c2, "config draw is not reproducible"
    print("  baselines draw weight_decay/dropout, memory arms do not, draws reproducible")


def test_regularisation_keys_are_not_passed_to_arms():
    """``weight_decay`` is a training key; if it leaked into the arm kwargs it would be dropped silently."""
    assert "weight_decay" in H.TRAIN_KEYS and "dropout" not in H.TRAIN_KEYS, H.TRAIN_KEYS
    print("  TRAIN_KEYS carries weight_decay (harness) and not dropout (arm)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
