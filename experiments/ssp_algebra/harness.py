"""Shared supervised probing harness (docs/11 section 4.1-4.3).

Supervised first, RL second: each arm is trained on teacher-forced oracle labels, which isolates what the memory
can represent and how fast it learns from examples, with no exploration or credit assignment in the way.  RL is
run only for arms that survive the decision rules (``analysis.py``).

Protocol, fixed here so it cannot drift:
  * **State size matched in floats** across the memory arms, from ``--budget`` (see ``arms.state_floats``).
    State-size knobs are deliberately *excluded* from the search: tuning them would break the matching the claim
    rests on.  Non-state knobs (lr, readout width, ``length_scale``, cleanup temperature) are tuned.
  * **Equal tuning budget per arm**: the same number of random configs, scored on training-side seeds only.
  * **Report on hold-out seeds only**, mean of the last ``k`` checkpoints, never max-over-epochs (POPGym's MMER is
    selection-inflated, docs/10 section 13.1).
  * **Extrapolation**: every arm is evaluated on the training split and on held-out longer lags / longer episodes.

    OMP_NUM_THREADS=1 ../../.venv/bin/python harness.py --arm ssp --variant delay --configs 4 --seeds 2
"""

import argparse
import json
import pathlib
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

import arms as A
import tasks_c

OUT = pathlib.Path(__file__).with_name("results")
OUT.mkdir(exist_ok=True)


# --------------------------------------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------------------------------------
def splits_for(variant, task, robust=False):
    """Train / in-distribution / extrapolation specs.  Extrapolation is always a *held-out* regime.

    Key names are kept parallel across variants: ``extrap_lag`` is the longer lag / **hop count**, ``extrap_T`` a
    longer episode, so the reported columns mean the same thing everywhere.
    """
    T = task.ep_len
    if variant == "index":
        sp = {"train": {"T": T}, "id": {"T": T}, "extrap_T": {"T": 4 * T}}
        if robust:                       # stream far beyond the D/4 capacity rule (docs/11 section 4.3)
            sp["rob_T128"] = {"T": 8 * T}
        return sp
    if variant == "delay_nolag":
        # Same difficulty axes as delay_distract (never change the alphabet or the input width): a longer stream,
        # longer lags.  No `dense` split: this variant trains at p_dist 0, so a distractor split would change
        # in_dim's meaning rather than its width -- use `delay_distract` for the selective-write question.
        hi = max(task.train_lags)
        return {"train": {"T": T}, "id": {"T": T},
                "extrap_T": {"T": 4 * T},
                "extrap_lag": {"T": 4 * T, "lags": (2 * hi, 3 * hi)}}      # top level: make_batch threads `lags` itself
    if variant == "delay_distract":
        # Difficulty axes that never change the alphabet: more distractors, a longer stream, longer lags.
        lags = task.train_lags
        hi = max(lags)
        return {"train": {"T": T}, "id": {"T": T},
                "dense": {"T": T, "task_kw": {"p_dist": 0.5}},
                "extrap_T": {"T": 4 * T},
                "extrap_lag": {"T": 4 * T, "lags": (2 * hi, 3 * hi)}}      # top level: make_batch threads `lags` itself
    if variant == "spatial":
        # The gate for the robotics direction.  Extrapolation past *both* trained axes: a longer episode (more motion
        # accumulated since each landmark was shown, so more shift steps) and a larger arena/step (more travel per
        # step).  Robustness corrupts the shown landmark positions, never the motion -- the motion is the input the
        # algebra acts on, and corrupting it would change the task rather than stress the memory.
        return {"train": {"T": T}, "id": {"T": T},
                "extrap_T": {"T": 2 * T},
                "extrap_motion": {"T": T, "task_kw": {"step_scale": 2.0 * task.step_scale}},
                "rob_pos": {"T": T, "obs_noise": 0.25}}
    if variant == "spatial_easy":
        # The gate variant (docs/11 4.2).  It keeps the *same split set* as `spatial` on purpose: the only difference
        # from the real task must be difficulty, not code path (`run_sweep` prints a max over the `extrap` keys, so a
        # variant without one crashes there).  With no motion, extrap_T is the same trivial task at 2x length.
        return {"train": {"T": T}, "id": {"T": T}, "extrap_T": {"T": 2 * T}}
    if variant == "assoc_mixed":
        # content-plus-position addressing (handoff 11 step 6).  Same axes as `assoc`: more pairs, and a corrupted
        # query (which now corrupts a *key* or a *position* depending on the mode, so the split is not vacuous).
        P = task.n_pairs
        return {"train": {"T": task.ep_len}, "id": {"T": task.ep_len},
                "extrap_lag": {"T": task.ep_len + 2 * P, "task_kw": {"n_distract": P}},
                "rob_pairs": {"T": task.ep_len, "cue_drop": 0.5}}
    if variant == "assoc":
        # content-keyed recall: no positional addressing anywhere.  Extrapolation = *more pairs* (a regime the arm
        # was never trained on), which is the content-side analogue of a longer lag.
        P = task.n_pairs
        nq = task.n_queries
        return {"train": {"T": task.ep_len}, "id": {"T": task.ep_len},
                "extrap_lag": {"T": task.ep_len + 2 * P, "task_kw": {"n_distract": P}},
                "rob_pairs": {"T": task.ep_len, "cue_drop": 0.5}}     # top level: `make_batch` threads it explicitly
    if variant == "pointer":
        sp = {"train": {"T": T}, "id": {"T": T},
              "extrap_lag": {"T": T, "hops": (4, 8)}}            # train 1-2 hops, test 4-8
        if robust:                       # corrupt INPUTS only; targets always come from the clean stream
            for p in (0.25, 0.5, 0.75):
                sp[f"rob_cue{int(p * 100)}"] = {"T": T, "cue_drop": p}
            for q in (0.1, 0.25):
                sp[f"rob_noise{int(q * 100)}"] = {"T": T, "obs_noise": q}
        return sp
    if variant == "path":
        return {"train": {"T": T}, "id": {"T": T},
                "extrap_T": {"T": 4 * T},                        # a longer path has to be integrated
                "extrap_step": {"T": T, "task_kw": {"step_scale": 2 * task.step_scale,
                                                    "arena": 2 * task.arena}}}
    lags = task.train_lags
    hi = max(lags)
    return {
        "train": {"lags": lags, "T": T},
        "id": {"lags": lags, "T": T},
        # a longer lag *with* enough room to answer it (lag 4*hi in a T=4*hi episode would leave no valid step)
        "extrap_lag": {"lags": (2 * hi, 3 * hi), "T": 4 * hi},
        "extrap_T": {"lags": lags, "T": 4 * T},
    }


def make_batch(task, spec, B, rng, device="cpu"):
    # every split-controlled input is threaded through explicitly: `lags` for the delay task, `hops` for pointer
    # chasing, `cue_drop`/`obs_noise` for the robustness splits, and `task_kw` for continuous task parameters
    # (step size, arena, ...).  A dropped key here would silently evaluate the *training* regime under an
    # extrapolation or corruption label.
    return task.batch(B, spec["T"], device=device, rng=rng, lags=spec.get("lags"), hops=spec.get("hops"),
                      cue_drop=spec.get("cue_drop", 0.0), obs_noise=spec.get("obs_noise", 0.0),
                      **spec.get("task_kw", {}))


def eval_schedule(steps, dense_until=400, dense_every=25, sparse_every=100):
    """Checkpoint steps: every 25 updates up to 400, then every 100 (handoff 09 section 5.2).

    RUN_007 evaluated every 300 updates, so an arm already at ceiling at the first checkpoint had an *unmeasured*
    time-to-threshold.  At ``steps=1500`` this gives 28 checkpoints.
    """
    pts = {s for s in range(0, min(steps, dense_until) + 1, dense_every)}
    pts |= {s for s in range(dense_until + sparse_every, steps + 1, sparse_every)}
    pts.add(0)
    pts.add(int(steps))
    return sorted(pts)


def _take(batch, idx):
    """Minibatch view of a pre-generated batch (pool mode).  Non-tensor entries pass through unchanged."""
    out = {}
    for key, val in batch.items():
        if torch.is_tensor(val):
            out[key] = val[idx]
        elif isinstance(val, dict):
            out[key] = dict(val)
        else:
            out[key] = val
    return out


# --------------------------------------------------------------------------------------------------------
# train / evaluate
# --------------------------------------------------------------------------------------------------------
def loss_and_acc(arm, task, batch, feat, reduction="mean"):
    logits = arm(batch, feat)
    m = batch["mask"]
    # The chain protocol (PointerTask(chain=True)) supervises every sub-step of a query but scores only the answer:
    # without a separate answer mask an arm would get credit for the easy first hops (handoff 09 step 4).
    ans = batch.get("ans_mask", m)
    if batch["loss"] == "ce":
        tgt = batch["target"].clone()
        tgt[~m] = -100
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
        with torch.no_grad():
            pred = logits.argmax(-1)
            acc = ((pred == batch["target"]) & ans).sum().item() / max(ans.sum().item(), 1)
    else:  # mse (variant A)
        err = ((logits - batch["target"]) ** 2).mean(-1)
        loss = (err * m).sum() / m.sum().clamp_min(1)
        with torch.no_grad():
            acc = -float((err * ans).sum() / ans.sum().clamp_min(1))
    return loss, float(acc)


@torch.no_grad()
def evaluate(arm, task, spec, seed, B=64, device="cpu"):
    rng = np.random.default_rng(10_000 + seed)
    accs = []
    for _ in range(3):
        b = make_batch(task, spec, B, rng, device)
        _, acc = loss_and_acc(arm, task, b, task.features(b))
        accs.append(acc)
    return float(np.mean(accs))


def make_optimizer(arm, lr, addr_lr_scale=1.0, weight_decay=0.0):
    """Adam over the trainable parameters; the address parameters get ``lr * addr_lr_scale`` when requested.

    ``weight_decay`` defaults to 0, so every existing caller is unchanged; it is a searched knob for the baseline
    arms only (handoff 09 section 15.4 step 1).
    """
    params = [p for p in arm.parameters() if p.requires_grad]
    if addr_lr_scale == 1.0 or not hasattr(arm, "address_params"):
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    ids = {id(p) for p in arm.address_params()}
    addr = [p for p in params if id(p) in ids]
    rest = [p for p in params if id(p) not in ids]
    groups = [{"params": rest, "lr": lr}] + ([{"params": addr, "lr": lr * addr_lr_scale}] if addr else [])
    return torch.optim.Adam(groups, weight_decay=weight_decay)


def train_arm(arm, task, sp, seed, steps, batch, lr, eval_every, device="cpu", log=False, eval_steps=None,
              pool=None, aux_weight=0.0, curriculum=None, anneal_steps=0, width_steps=0, addr_lr_scale=1.0,
              pretrain_steps=0, pretrain_lr=None, pretrain_mode="retrieval", weight_decay=0.0,
              pretrain_shapes=None, probe_aux_weight=0.0):
    """Teacher-forced training.  Returns the checkpoint curve and the final metrics.

    ``eval_steps``: explicit checkpoint steps (see :func:`eval_schedule`); when it is None, ``eval_every`` is used, so
    every existing caller behaves exactly as before.  ``pool=N``: the low-data regime (handoff 09 section 5.6) --
    N training sequences are generated once and minibatches are sampled from that fixed pool, so the arm is trained
    to convergence on a finite dataset rather than for few updates.  Evaluation always uses fresh data.

    Training aids (docs/11 section 3.4; every one is off by default and must be reported as "aided" with its own
    sample cost): ``aux_weight`` adds ``arm.aux_loss(batch)`` (per-hop / address supervision) when the arm has one;
    ``curriculum(step) -> spec overrides`` widens the *training* split over time (evaluation is unaffected);
    ``anneal_steps`` calls ``arm.set_temp(step)`` to anneal the cleanup temperature.

    Diagnostics (off by default): ``addr_lr_scale`` multiplies the learning rate of ``arm.address_params()``;
    ``pretrain_steps`` first learns the *interface* on fresh synthetic streams, then freezes it, so the readout is
    trained afterwards on a fixed interface (the "staging" recipe component, handoff 09 step 3).  ``pretrain_mode``
    selects what the interface is trained through: ``"retrieval"`` (cross-entropy of the arm's own cleanup scores),
    ``"probe"`` (a throwaway MLP readout on the arm's memory features, discarded afterwards) or ``"full"`` (the whole
    arm on the synthetic task, then the readout is re-initialised so it is not entangled with the synthetic stream).
    The pretraining tokens are returned separately and are NOT counted in ``tokens``.
    """
    torch.manual_seed(1000 + seed)
    rng = np.random.default_rng(seed)
    pretrain_tokens = 0
    if int(pretrain_steps) and hasattr(arm, "address_params"):
        if pretrain_mode not in ("retrieval", "probe", "full"):
            raise ValueError(f"unknown pretrain_mode {pretrain_mode!r} (retrieval | probe | full)")
        if pretrain_mode == "retrieval" and not hasattr(arm, "retrieval_loss"):
            raise ValueError("pretrain_mode='retrieval' needs an arm with retrieval_loss")
        probe = None
        if pretrain_mode == "probe":
            if not hasattr(arm, "memory_features"):
                raise ValueError("pretrain_mode='probe' needs an arm with memory_features (a decodable memory state)")
            import torch.nn as nn

            in_dim = arm.memory_features(make_batch(task, sp["train"], 2, np.random.default_rng(0), device),
                                         task.features(make_batch(task, sp["train"], 2,
                                                                  np.random.default_rng(0), device))).shape[-1]
            probe = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, task.out_dim))
            ptrain = list(arm.address_params()) + list(probe.parameters())
        elif pretrain_mode == "full":
            ptrain = [p for p in arm.parameters() if p.requires_grad]      # the whole arm, interface included
        else:
            ptrain = list(arm.address_params())
        popt = torch.optim.Adam(ptrain, lr=pretrain_lr or lr)
        prng = np.random.default_rng(777_000 + seed)
        plast = None
        for pstep in range(int(pretrain_steps)):
            if hasattr(arm, "set_width"):
                arm.set_width(pstep, int(pretrain_steps), feedback=plast)
            if pretrain_shapes:
                # Phase 2.1: draw a *new shape* every step, so the interface cannot fit one stream length (which is
                # what made the T=16 interface useless at T=32 in RUN_010).  Lengths are drawn from
                # [pretrain_shapes[0], pretrain_shapes[1]]; the alphabet stays the arm's own (the interface never
                # sees the codebook, it only sees addresses and time).
                Tp = int(prng.integers(int(pretrain_shapes[0]), int(pretrain_shapes[1]) + 1))
                ptask = tasks_c.make_task(task.name, n_items=task.n_items, ep_len=Tp,
                                          **({"train_lags": task.train_lags} if task.name == "delay" else {}))
                pb = make_batch(ptask, {"T": Tp}, batch, prng, device)
                pfeat = ptask.features(pb)
            else:
                pb = make_batch(task, sp["train"], batch, prng, device)
                pfeat = task.features(pb)
            arm.train()
            arm(pb, pfeat)
            if pretrain_mode == "retrieval":
                ploss = arm.retrieval_loss(pb)
            elif pretrain_mode == "probe":
                logits = probe(arm.memory_features(pb, pfeat))
                tgt = pb["target"].clone()
                tgt[~pb["mask"]] = -100
                ploss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            else:
                ploss, _ = loss_and_acc(arm, task, pb, pfeat)
            popt.zero_grad(set_to_none=True)
            ploss.backward()
            torch.nn.utils.clip_grad_norm_(ptrain, 1.0)
            popt.step()
            plast = float(ploss.item())
            pretrain_tokens += int(pb["mask"].sum().item())
        del probe                                            # the probe is thrown away: only the interface survives
        arm.freeze_address()
        if pretrain_mode == "full":
            # everything was trained together on synthetic streams, so the readout is entangled with them: the staged
            # protocol is "interface first, readout second on the task data", which needs a fresh readout
            for m in arm.mix.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        if hasattr(arm, "set_width"):
            arm.set_width(10 ** 9, 1)               # final (narrow) kernel for everything that follows
    opt = make_optimizer(arm, lr, addr_lr_scale, weight_decay)
    live_probe = None
    if probe_aux_weight > 0:
        # Phase 2.3: the pretraining stage as an *in-training* auxiliary.  A throwaway probe is trained jointly with
        # the arm from step 0 (no staging, no freezing), so the interface gets a label-free alignment signal in the
        # same run and the same tokens as the task itself.  The probe is discarded afterwards; its parameters are
        # optimised but never evaluated.  This is the form RL can use, because the stream's own past observations are
        # the labels.
        import torch.nn as nn

        b0 = make_batch(task, sp["train"], 2, np.random.default_rng(0), device)
        f0 = task.features(b0)
        if not hasattr(arm, "memory_features"):
            raise ValueError("probe_aux_weight needs an arm with memory_features")
        live_probe = nn.Sequential(nn.Linear(arm.memory_features(b0, f0).shape[-1], 32), nn.ReLU(),
                                   nn.Linear(32, task.out_dim))
        opt = torch.optim.Adam([p for p in arm.parameters() if p.requires_grad] + list(live_probe.parameters()),
                               lr=lr, weight_decay=weight_decay)
    last_loss = None
    curve = []
    tokens = 0
    t0 = time.time()
    schedule = None if eval_steps is None else set(int(s) for s in eval_steps)
    pool_batch = None
    if pool:
        pool_batch = make_batch(task, sp["train"], int(pool), np.random.default_rng(5000 + seed), device)
    for step in range(steps + 1):
        if width_steps and hasattr(arm, "set_width"):
            arm.set_width(step, width_steps, feedback=last_loss)   # before the checkpoint: evaluated at the current width
        due = (step in schedule) if schedule is not None else (step % eval_every == 0)
        if due or step == steps:
            row = {"step": step, "tokens": tokens}
            for name, spec in sp.items():
                row[name] = evaluate(arm, task, spec, seed, device=device)
            curve.append(row)
        if step == steps:
            break
        arm.train()
        if anneal_steps and hasattr(arm, "set_temp"):
            arm.set_temp(step, decay_steps=anneal_steps)
        if pool_batch is None:
            tspec = sp["train"] if curriculum is None else {**sp["train"], **curriculum(step)}
            b = make_batch(task, tspec, batch, rng, device)
        else:
            b = _take(pool_batch, torch.as_tensor(rng.integers(0, int(pool), size=batch)))
        feat = task.features(b)
        loss, _ = loss_and_acc(arm, task, b, feat)
        if aux_weight > 0 and hasattr(arm, "aux_loss"):
            aux = arm.aux_loss(b)
            if aux is not None:
                loss = loss + aux_weight * aux
        if live_probe is not None:
            # label-free alignment auxiliary: decode the *stream's own* tokens at the query positions from the memory
            # features.  It uses the same labels the retrieval loss would (no extra supervision), but it is trained
            # jointly rather than staged, so nothing has to be frozen afterwards.
            plogits = live_probe(arm.memory_features(b, feat))
            ptgt = b["target"].clone()
            ptgt[~b["mask"]] = -100
            loss = loss + probe_aux_weight * F.cross_entropy(plogits.reshape(-1, plogits.shape[-1]),
                                                             ptgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(arm.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.item())
        tokens += int(b["mask"].sum().item())
    return curve, {"wall_s": time.time() - t0, "tokens": tokens, "pretrain_tokens": pretrain_tokens}


def _trapz(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def summarize(curve, k=5, threshold=0.9, loss="ce"):
    """Per-split summary.  ``last5_*`` is kept unchanged for continuity with RUN_007, but it is a short
    learning-curve area, not an accuracy (RUN_007 bug 7.13).  Read instead, and never blend them (docs/11 4.3):

      * ``conv_*``      -- mean of the last 3 checkpoints: the converged score
      * ``t50_*``/``t90_*`` -- tokens to 50% / 90% of ceiling: the *speed*
      * ``aulc_*``      -- area under the learning curve, per token
    """
    out = {}
    ceiling = 1.0 if loss == "ce" else None
    toks = [row["tokens"] for row in curve]
    for key in [k for k in curve[-1] if k not in ("step", "tokens")]:
        vals = [row[key] for row in curve]
        out[f"last{k}_{key}"] = float(np.mean(vals[-k:]))
        out[f"conv_{key}"] = float(np.mean(vals[-3:]))
        out[f"aulc_{key}"] = _trapz(vals, toks) if len(toks) > 1 else float(vals[-1])
        for frac in (0.5, 0.9):
            out[f"t{int(frac * 100)}_{key}"] = next(
                (row["tokens"] for row in curve if ceiling is not None and row[key] >= frac * ceiling), None)
        out[f"tokens_to_{threshold}_{key}"] = next((row["tokens"] for row in curve if row[key] >= threshold), None)
    out["final"] = {key: curve[-1][key] for key in curve[-1] if key not in ("step", "tokens")}
    return out


# --------------------------------------------------------------------------------------------------------
# search spaces (state-size knobs excluded on purpose)
# --------------------------------------------------------------------------------------------------------
def stable_seed(*parts):
    """Deterministic seed from strings/ints.  ``hash()`` is salted per process, so anything that must survive a
    restart (the random-search config draw) has to use this instead: otherwise resuming a partial search in a new
    process silently mixes different configs under the same (variant, arm, cfg_idx) key (RUN_007 bug 7.14)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


def sample_config(arm_name, rng, budget, variant="delay"):
    cfg = {"lr": float(10 ** rng.uniform(-3.5, -1.5))}   # ~3e-4 .. 3e-2: the working rate for these arms is ~1e-2
    if arm_name in ("mlp", "posmlp"):
        cfg["hidden"] = int(rng.choice([32, 64, 128]))
    elif arm_name in ("gru", "lru", "tf_window", "kernel_attn"):
        cfg["hidden"] = int(rng.choice([32, 64]))
    elif arm_name in ("tf_full", "looped_tf"):
        cfg["tf_d_model"] = int(rng.choice([32, 64]))
        cfg["hidden"] = int(rng.choice([32, 64]))
    elif arm_name in ("assoc_ssp", "assoc_fwp"):
        cfg["lam_init"] = float(rng.choice([0.9, 0.99]))
        if arm_name == "assoc_ssp":
            cfg["bind"] = True
            cfg["learned_codes"] = False
    elif arm_name in ("fwp", "fwp_delta"):
        cfg["hidden"] = int(rng.choice([32, 64]))
        cfg["lam_init"] = float(rng.choice([0.9, 0.99]))
    elif arm_name in ("ssp_index_learned", "ssp_pointer_learned"):
        cfg["lam_init"] = float(rng.choice([0.9, 0.99]))
        cfg["temp"] = float(rng.choice([0.05, 0.1, 0.5]))
        if arm_name == "ssp_index_learned":
            cfg["length_scale"] = float(rng.choice([0.5, 1.0, 2.0]))
    elif arm_name == "ssp":
        cfg["hidden"] = int(rng.choice([32, 64]))
        cfg["lam_init"] = float(rng.choice([0.9, 0.99]))     # forgetting horizon: searched, see arms.SSPArm
        if variant == "path":
            # RUN_005 T7: kernel resolution must be matched to the task; the relative vectors span the arena
            cfg["length_scale"] = float(rng.choice([2.0, 4.0, 8.0]))
        elif variant == "pointer":
            cfg["temp"] = float(rng.choice([0.05, 0.1, 0.5]))
        else:
            cfg["length_scale"] = float(rng.choice([0.5, 1.0, 2.0]))
            cfg["temp"] = float(rng.choice([0.05, 0.1, 0.5]))
    if arm_name in ("mlp", "posmlp", "gru", "lru", "tf_full", "tf_window", "kernel_attn", "looped_tf"):
        # Baseline hardening for the low-data regime (handoff 09 section 15.4 step 1): regularisation is the axis the
        # baselines were never given, so a low-data comparison against them is only fair once it is searched.  Drawn
        # *after* the knobs above so the draws those knobs produce are unchanged; every baseline gets the same number
        # of conditions, so the tuning budget stays matched.
        cfg["weight_decay"] = float(rng.choice([0.0, 1e-4, 1e-2]))
        cfg["dropout"] = float(rng.choice([0.0, 0.1, 0.3]))
    return cfg


TRAIN_KEYS = ("lr", "aux_weight", "curriculum", "curr_steps", "anneal_steps", "width_steps", "addr_lr_scale",
              "pretrain_steps", "pretrain_lr", "pretrain_mode", "weight_decay", "pretrain_shapes",
              "probe_aux_weight")


def make_curriculum(kind, task, curr_steps):
    """``index_j``: cap the queried index and widen it linearly; ``pointer_hops``: 1 hop first, then the full set."""
    if not kind:
        return None
    if kind == "index_j":
        hi = max(task.ep_len - 2, 1)
        return lambda step: {"task_kw": {"jmax": max(2, int(2 + (hi - 2) * min(1.0, step / max(curr_steps, 1))))}}
    if kind == "pointer_hops":
        full = tuple(task.train_hops)
        return lambda step: {"hops": (full[0],) if step < curr_steps else full}
    raise ValueError(kind)


def build_any(arm_name, task, budget, seed, cfg):
    """Rung-0 / baseline arms come from ``arms``; learned-structure arms and the FWP control from ``arms_learned``."""
    import arms_learned as L

    arm_cfg = {k: v for k, v in cfg.items() if k not in TRAIN_KEYS}
    if arm_name in L.LEARNED_ARMS:
        return L.build_learned(arm_name, task, budget=budget, seed=seed, **arm_cfg)
    return A.build_arm(arm_name, task, budget=budget, seed=seed, **arm_cfg)


def run_one(arm_name, task, sp, budget, cfg, seed, steps, batch, eval_every, device="cpu", eval_steps=None,
            pool=None):
    arm, info = build_any(arm_name, task, budget, seed, cfg)
    curve, extra = train_arm(arm, task, sp, seed, steps, batch, cfg["lr"], eval_every, device,
                             eval_steps=eval_steps, pool=pool, aux_weight=float(cfg.get("aux_weight", 0.0)),
                             curriculum=make_curriculum(cfg.get("curriculum"), task, int(cfg.get("curr_steps", 0))),
                             anneal_steps=int(cfg.get("anneal_steps", 0)),
                             width_steps=int(cfg.get("width_steps", 0)),
                             addr_lr_scale=float(cfg.get("addr_lr_scale", 1.0)),
                             pretrain_steps=int(cfg.get("pretrain_steps", 0)),
                             pretrain_lr=cfg.get("pretrain_lr"),
                             pretrain_mode=cfg.get("pretrain_mode", "retrieval"),
                             pretrain_shapes=cfg.get("pretrain_shapes"),
                             probe_aux_weight=float(cfg.get("probe_aux_weight", 0.0)),
                             weight_decay=float(cfg.get("weight_decay", 0.0)))
    summary = summarize(curve, threshold=0.9 if task.loss == "ce" else -0.05, loss=task.loss)
    return {"arm": arm_name, "cfg": cfg, "seed": seed, "info": info, "summary": summary,
            "curve": curve, **extra}


def search(arm_names, variant, budget, n_configs, seeds, steps, batch, eval_every, ep_len, n_items,
           train_lags, device="cpu"):
    """Random search at an equal config count per arm; the winner is chosen on training-side seeds only."""
    task = tasks_c.make_task(variant, n_items=n_items, ep_len=ep_len, train_lags=train_lags)
    sp = splits_for(variant, task)
    results = []
    for arm_name in arm_names:
        rng = np.random.default_rng(stable_seed(arm_name, variant, budget))
        trials = []
        for ci in range(n_configs):
            cfg = sample_config(arm_name, rng, budget, variant)
            row = run_one(arm_name, task, sp, budget, cfg, seed=ci, steps=steps, batch=batch,
                          eval_every=eval_every, device=device)
            trials.append(row)
            print(f"  [{arm_name}] cfg{ci} {cfg} -> id={row['summary']['last5_id']:.3f} "
                  f"elag={row['summary']['last5_extrap_lag']:.3f} "
                  f"eT={row['summary']['last5_extrap_T']:.3f} ({row['wall_s']:.1f}s)", flush=True)
        trials.sort(key=lambda r: r["summary"]["last5_id"], reverse=True)
        best = trials[0]
        # hold-out seeds: never seen by the search
        final = []
        for s in range(1000, 1000 + seeds):
            row = run_one(arm_name, task, sp, budget, best["cfg"], seed=s, steps=steps, batch=batch,
                          eval_every=eval_every, device=device)
            final.append(row)
        results.append({"arm": arm_name, "best_cfg": best["cfg"], "search_trials":
                        [{"cfg": t["cfg"], "summary": t["summary"]} for t in trials],
                        "final": [{"seed": r["seed"], "summary": r["summary"], "info": r["info"],
                                   "curve": r["curve"]} for r in final]})
    return {"variant": variant, "budget": budget, "n_configs": n_configs, "seeds": seeds, "steps": steps,
            "batch": batch, "ep_len": ep_len, "n_items": n_items, "train_lags": list(train_lags),
            "results": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="delay", choices=["delay", "index"])
    p.add_argument("--arms", nargs="*", default=list(A.ALL_ARMS))
    p.add_argument("--budget", type=int, default=1024, help="matched state floats for the memory arms")
    p.add_argument("--configs", type=int, default=4, help="random configs per arm (equal across arms)")
    p.add_argument("--seeds", type=int, default=2, help="hold-out seeds for the reported number")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--ep-len", type=int, default=32)
    p.add_argument("--n-items", type=int, default=8)
    p.add_argument("--train-lags", type=int, nargs="*", default=[1, 2, 3, 4])
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    torch.set_num_threads(1)
    out = search(args.arms, args.variant, args.budget, args.configs, args.seeds, args.steps, args.batch,
                 args.eval_every, args.ep_len, args.n_items, tuple(args.train_lags))
    tag = args.tag or f"{args.variant}_b{args.budget}_c{args.configs}_s{args.seeds}"
    path = OUT / f"{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
