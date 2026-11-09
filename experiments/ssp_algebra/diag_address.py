"""Diagnostic for the learnability question on index recall (handoff 09 step E follow-up).

    OMP_NUM_THREADS=1 ../../.venv/bin/python diag_address.py --workers 12 --steps 1500
    OMP_NUM_THREADS=1 ../../.venv/bin/python diag_address.py --smoke

Hypothesis under test (an *inference* from the learn1 results, not yet measured): a randomly initialised read phase starts
so far from the write phase that the SSP similarity kernel (informative only within ~1 index of a match) gives almost no
gradient, so plain training never locks on.  Two measurements:

  1. **Trace.**  Every ``--log-every`` updates, on a fixed evaluation batch: RMS read-address error ``|s_q - j|``, the
     fraction of valid queries already inside the kernel's basin (``|s_q - j| < 0.5``), the gradient norm of the read
     head, and accuracy.  If the hypothesis is right, runs that stall start with (nearly) nothing in the basin and a
     near-zero read-head gradient.
  2. **Basin width.**  Initialise the read head at the correct solution plus noise of size sigma and record final
     accuracy against the *measured* initial RMS address error.  This replaces "exactly right (0.998) vs random
     (0.40)" with a curve: how close to the answer must the learner start?

Plain training only (no aids): the aim is to measure the arm's own optimisation, so every number here is comparable with
``learn1_r1_plain``.  Reads only from ``arms_learned`` / ``harness`` / ``tasks_c``; it edits nothing.
"""

import argparse
import json
import multiprocessing as mp
import pathlib
import time

import numpy as np
import torch

OUT = pathlib.Path(__file__).with_name("results")
SIGMAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)         # "random" (default nn.Linear init) is run as sigma = None


def _task(ep_len=16):
    import tasks_c

    return tasks_c.IndexTask(n_items=8, ep_len=ep_len)


def _addr_stats(arm, batch):
    """RMS read-address error, fraction inside the kernel basin, over valid query steps."""
    with torch.no_grad():
        _, s_q, addr = arm._read_phase(batch)
    m = batch["mask"]
    err = (s_q - addr)[m]
    return float(torch.sqrt((err ** 2).mean())), float((err.abs() < 0.5).float().mean())


def run_one(job):
    """One training run.  ``job``: dict(sigma, seed, steps, log_every, lr)."""
    torch.set_num_threads(1)
    import arms_learned as L
    import feats
    import harness as H

    sigma, seed, steps = job["sigma"], job["seed"], job["steps"]
    task = _task()
    sp = H.splits_for("index", task)
    arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=seed, rung=1, init="random",
                             lam_init=0.99, length_scale=job.get("length_scale", 1.0), temp=0.1)
    if sigma is not None:                     # correct solution (read address = j) plus noise of size sigma
        g = torch.Generator().manual_seed(7000 + seed)
        with torch.no_grad():
            w = torch.zeros(feats.TIME_DIM)
            w[0] = feats.RAMP_SCALE                                   # the ramp feature is j/32 -> weight 32 gives j
            noise = torch.randn(feats.TIME_DIM, generator=g)
            noise[0] *= feats.RAMP_SCALE                              # same units: sigma is ~ an address error in indices
            arm.read_head.weight.copy_((w + sigma * noise)[None, :])
            arm.read_head.bias.copy_(sigma * torch.randn(1, generator=g))
    eval_batch = task.batch(256, task.ep_len, rng=np.random.default_rng(10_000 + seed))
    rng = np.random.default_rng(seed)
    torch.manual_seed(1000 + seed)
    opt = torch.optim.Adam(arm.parameters(), lr=job["lr"])
    trace, init_rms = [], None
    t0 = time.time()
    for step in range(steps + 1):
        if step % job["log_every"] == 0 or step == steps:
            rms, frac = _addr_stats(arm, eval_batch)
            with torch.no_grad():
                _, acc = H.loss_and_acc(arm, task, eval_batch, task.features(eval_batch))
            if init_rms is None:
                init_rms = rms
            gn = None if arm.read_head.weight.grad is None else float(arm.read_head.weight.grad.norm())
            trace.append({"step": step, "rms_err": rms, "frac_in_basin": frac, "read_grad": gn, "acc": float(acc)})
        if step == steps:
            break
        b = task.batch(64, task.ep_len, rng=rng)
        loss, _ = H.loss_and_acc(arm, task, b, task.features(b))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(arm.parameters(), 1.0)
        opt.step()
    return {"sigma": sigma, "seed": seed, "init_rms_err": init_rms, "final_acc": trace[-1]["acc"],
            "final_rms_err": trace[-1]["rms_err"], "trace": trace, "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--length-scale", type=float, default=1.0,
                    help="similarity-kernel width in index units (the kernel is sinc(pi d / length_scale))")
    ap.add_argument("--tag", default="diag1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.seeds, a.log_every, a.workers = 40, 1, 20, 2
    sigmas = list(SIGMAS) + [None]
    jobs = [{"sigma": s, "seed": 1000 + k, "steps": a.steps, "log_every": a.log_every, "lr": a.lr,
             "length_scale": a.length_scale}
            for s in sigmas for k in range(a.seeds)]
    print(f"{len(jobs)} runs ({len(sigmas)} inits x {a.seeds} seeds, {a.steps} updates each)", flush=True)
    with mp.Pool(a.workers, maxtasksperchild=1) as pool:
        rows = list(pool.imap_unordered(run_one, jobs, chunksize=1))
    rows.sort(key=lambda r: (-1 if r["sigma"] is None else r["sigma"], r["seed"]))
    path = OUT / f"{a.tag}.json"
    path.write_text(json.dumps(rows, indent=1))
    print(f"\n{'init':>10s} {'init RMS err':>13s} {'in-basin@0':>11s} {'final err':>10s} {'final acc':>10s}  (mean over seeds)")
    for s in sigmas:
        rs = [r for r in rows if r["sigma"] == s]
        label = "random" if s is None else f"sigma={s:g}"
        fb0 = np.mean([r["trace"][0]["frac_in_basin"] for r in rs])
        print(f"{label:>10s} {np.mean([r['init_rms_err'] for r in rs]):13.2f} {fb0:11.2f} "
              f"{np.mean([r['final_rms_err'] for r in rs]):10.2f} {np.mean([r['final_acc'] for r in rs]):10.3f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
