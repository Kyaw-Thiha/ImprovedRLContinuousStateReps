"""Tests for the continuous-address recipe rule (handoff 09 section 15.4 step 2).

Run:  OMP_NUM_THREADS=1 ../../.venv/bin/python test_recipes.py

These pin the *rule* (so a later edit cannot silently change the constants a frozen run used) and, more importantly,
that the rule's output is consumable: every key of ``arm_cfg`` is either a harness training key or a real
``SSPIndexLearned`` argument, and the schedule fits inside the number of updates.
"""
import inspect
import sys

import numpy as np

import arms_learned as L
import harness as H
import recipes as RC
import tasks_c


def test_pinned_values_for_the_two_tuning_variants():
    """The two conditions RUN_009 already probed must come out of the rule, not out of a hand-set table."""
    s16 = RC.continuous_address_cfg(RC.address_range("index", ep_len=16), a=0.5, b=24.0)
    assert (s16["R"], s16["arm_cfg"]["ls_start"], s16["arm_cfg"]["width_steps"]) == (14, 7.0, 336), s16
    assert s16["steps"] == 672
    s32 = RC.continuous_address_cfg(RC.address_range("index", ep_len=32), a=1.0, b=47.0)
    assert (s32["R"], s32["arm_cfg"]["ls_start"], s32["arm_cfg"]["width_steps"]) == (30, 30.0, 1410), s32
    assert s32["steps"] == 2820
    print(f"  T=16 -> start {s16['arm_cfg']['ls_start']} width {s16['arm_cfg']['width_steps']} steps {s16['steps']}; "
          f"T=32 -> start {s32['arm_cfg']['ls_start']} width {s32['arm_cfg']['width_steps']} steps {s32['steps']}")


def test_every_knob_of_the_recipe_is_fixed_by_the_rule():
    """A recipe fixes *every* hyperparameter (docs/11 4.7): nothing may be left to the search draw."""
    cfg = RC.continuous_address_cfg(14)["arm_cfg"]
    for k in ("rung", "init", "length_scale", "ls_start", "ls_mode", "width_steps", "lr", "lam_init", "temp"):
        assert k in cfg, f"the rule leaves {k} unspecified, so a search draw would set it"
    # the knobs the learned-arm sampler draws (lam_init, temp) must be overridden by the rule, not inherited
    drawn = {"lam_init", "temp", "length_scale"}
    assert drawn <= set(cfg), drawn
    print(f"  rule fixes {sorted(cfg)}")


def test_arm_cfg_keys_are_consumable():
    """Every key must be a training key or an arm argument: an unknown key makes ``build_learned`` raise."""
    allowed = set(H.TRAIN_KEYS) | set(inspect.signature(L.SSPIndexLearned.__init__).parameters)
    for R in (1, 14, 30, 62):
        cfg = RC.continuous_address_cfg(R)["arm_cfg"]
        bad = set(cfg) - allowed
        assert not bad, f"R={R}: un-consumable arm_cfg keys {bad}"
    print("  arm_cfg keys are all TRAIN_KEYS or SSPIndexLearned arguments")


def test_schedule_fits_inside_the_run():
    for a in RC.A_GRID:
        for b in RC.B_GRID:
            for R in (2, 14, 30, 62):
                s = RC.continuous_address_cfg(R, a=a, b=b)
                assert 0 < s["arm_cfg"]["width_steps"] <= s["steps"], (R, a, b, s)
    print("  width_steps <= steps for every grid point and R")


def test_rule_scales_with_the_range():
    """The rule's whole purpose: bigger address range -> wider start and longer schedule, no retuning."""
    starts = [RC.continuous_address_cfg(R)["arm_cfg"]["ls_start"] for R in (14, 24, 30, 48, 62)]
    widths = [RC.continuous_address_cfg(R)["arm_cfg"]["width_steps"] for R in (14, 24, 30, 48, 62)]
    assert starts == sorted(starts) and widths == sorted(widths), (starts, widths)
    assert all(w2 > w1 for w1, w2 in zip(widths, widths[1:])), widths
    print(f"  ls_start {starts}; width_steps {widths}")


def test_address_range_definitions():
    assert RC.address_range("index", ep_len=16) == 14
    assert RC.address_range("index", ep_len=4) == 2
    assert RC.address_range("delay", max_lag=8) == 8
    for bad in (("index", {}), ("delay", {}), ("pointer", {"ep_len": 16})):
        try:
            RC.address_range(bad[0], **bad[1])
        except ValueError:
            continue
        raise AssertionError(f"address_range({bad[0]}, {bad[1]}) should raise")
    print("  index R=T-2, delay R=max_lag, everything else raises")


def test_rule_is_reproducible_and_has_no_hidden_state():
    a = RC.continuous_address_cfg(14, a=0.5, b=24.0)
    b = RC.continuous_address_cfg(14, a=0.5, b=24.0)
    assert a == b and a["arm_cfg"] is not b["arm_cfg"], "the rule must return equal values, not shared state"
    print("  repeated calls agree, no shared state")


def test_recipe_delay_path_reads_at_t_minus_lag():
    """Handoff 09 step 2 requires the delay path to be verified before the study relies on it."""
    task = tasks_c.DelayTask(n_items=8, ep_len=16, train_lags=(1, 2, 3, 4))
    assert task.name == "delay"
    sp = H.splits_for("delay", task)
    spec = RC.continuous_address_cfg(RC.address_range("delay", max_lag=4), a=1.0, b=24.0)
    cfg = {k: v for k, v in spec["arm_cfg"].items()}
    arm, info = L.build_learned("ssp_index_learned", task, budget=256, seed=0, hidden=32, **cfg)
    b = task.batch(8, task.ep_len, rng=np.random.default_rng(0))
    addr = arm._read_addr(b)
    lag = b["lag"].float()[:, None].expand_as(b["t"].float()) * b["mask"].float()
    assert addr.shape == b["t"].shape and np.allclose(addr.numpy(), lag.numpy()), "delay read address is not the lag"
    _, s_q, _ = arm._read_phase(b)
    assert s_q.shape == b["t"].shape, "the delay read phase must be per-step"
    curve, _ = H.train_arm(arm, task, sp, seed=0, steps=60, batch=32, lr=cfg["lr"], eval_every=60,
                           width_steps=cfg["width_steps"])
    acc = H.summarize(curve)["last5_id"]
    print(f"  delay: read address == lag; 60 steps of the recipe -> id {acc:.3f}")
    assert all(np.isfinite([r["id"] for r in curve])), "delay recipe run produced non-finite scores"


def test_job_entry_is_launchable_shape():
    j = RC.job_entry("t-a", "index", "t_a", ep_len=16)
    assert j["rows"] == 6 * 3 + 10, j["rows"]
    assert "--select aulc" in j["extra"] and "--search-seeds 3" in j["extra"]
    assert j["steps"] == RC.continuous_address_cfg(14)["steps"]
    import json
    assert " " not in json.dumps(j["arm_cfg"], separators=(",", ":")), "pod arm-cfg JSON must have no spaces"
    print(f"  job entry rows={j['rows']} steps={j['steps']}")


def test_slope_only_recipe_has_no_address_range_constant():
    """RUN_011's mechanism-derived recipe must be usable at any stream length without a task-derived constant."""
    import arms_learned as L

    cfg = RC.slope_only_cfg()
    assert set(cfg) <= set(RC.continuous_address_cfg(14)["arm_cfg"]) | {"affine_offset_zero", "affine_eps"}, cfg
    assert cfg["ls_mode"] == "fixed" and "ls_start" not in cfg and "width_steps" not in cfg, \
        "the recipe must carry no width schedule"
    assert cfg["affine_offset_zero"] is True, "the offset channel is the whole point: it must be pinned to zero"
    # it must build and run at two very different stream lengths with the SAME config
    for T in (16, 48):
        task = tasks_c.IndexTask(n_items=8, ep_len=T)
        arm, _ = L.build_learned("ssp_index_learned", task, budget=512 if T > 32 else 256, seed=0, **cfg)
        assert arm.affine_offset_zero is True and float(arm.read_head.bias.detach()) == 0.0
    print("  slope_only_cfg: no schedule, no range constant, offset pinned to zero at T=16 and T=48")


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
