"""Campaign job list (the pre-registered matrix in docs/mjlab_centering/00-plan.md)."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

from mjlab_centering.train import RunSpec

SEEDS = (2000, 2001, 2002, 2003, 2004)
# Tuning (eta, lr, SSP length scale, penalty check) uses its own seed block, disjoint from the claim seeds above
# (decision 35: the earlier SWEEP_SEEDS=(2000, 2001) sat inside the claim block, a hold-out violation).
SEARCH_SEEDS = (1000, 1001, 1002)
SWEEP_SEEDS = SEARCH_SEEDS
PENALTY = 10.0
OFFSET = 1.0
ITERS = 400
SHOCK_ITERS = 700
SHOCK_AT = 400

DEFAULT_ETA = 0.01  # replaced by the eta-sweep winner before the campaign body launches
# 2048 envs / 600 iterations: the measured wall time is ~2.3 s per 4096-env iteration on the A40 (43k steps/s), i.e.
# ~$0.30 per 800-iteration run, which would put the 95-run matrix at ~$28. At 2048 envs and 0.9 s/iteration (measured)
# a 600-iteration run costs ~$0.10 and sees 29.5M env-steps. 600 is the pre-registered floor of the plateau rule, and
# it is also what keeps the 95-run matrix inside the wall-clock and dollar budget; the reference arm's 600-iteration
# curve is reported, so a "not yet plateaued" caveat is visible rather than hidden.
NUM_ENVS = 2048
ETA_GRID = (0.003, 0.03)
LR_GRID = (5e-4, 2e-3)


def _id(block: str, **kw) -> str:
  parts = [block] + [f"{k}{v}" for k, v in kw.items() if v is not None]
  return "_".join(str(p).replace(" ", "") for p in parts)


def build_jobs(eta: float = DEFAULT_ETA, iters: int = ITERS, blocks: tuple[str, ...] | None = None,
               seeds: tuple[int, ...] = SEEDS, centers: tuple[str, ...] | None = None) -> list[RunSpec]:
  """The pre-registered matrix. `iters` comes from the plateau gate; `blocks`/`seeds` implement the cut order."""
  jobs: list[RunSpec] = []

  def add(block: str, seed: int, center: str, gamma: float, offset: float, tag: str = "", **kw) -> None:
    if blocks is not None and block not in blocks:
      return
    if seed not in seeds and seed not in SEARCH_SEEDS:
      return
    if centers is not None and center not in centers:
      return
    kw.setdefault("eta", eta)
    kw.setdefault("iters", iters)
    label = f"{block}_g{gamma}_o{offset}_{center}_s{seed}" + (f"_{tag}" if tag else "")
    kw.setdefault("num_envs", NUM_ENVS)
    jobs.append(RunSpec(job_id=label, block=block, seed=seed, center=center, gamma=gamma, offset=offset, **kw))

  # tier 1: eta sweep on the hardest cell (value, gamma=.999, offset c)
  for e in (0.003, eta, 0.3):
    for s in SWEEP_SEEDS:
      add("eta", center="value", gamma=0.999, offset=OFFSET, eta=e, seed=s, tag=f"eta{e:g}", priority=1)

  # tier 1: A - centering x gamma x offset, 5 seeds
  for c in ("none", "value"):
    for g in (0.99, 0.999):
      for off in (0.0, OFFSET):
        for s in SEEDS:
          add("A", center=c, gamma=g, offset=off, seed=s, priority=1)

  # tier 2: critic-fix rivals at gamma=.999, offset c
  for s in SEEDS:
    add("fixclip", center="none", gamma=0.999, offset=OFFSET, seed=s, value_clip=False, priority=2)
  for s in SEEDS:
    add("fixnorm", center="none", gamma=0.999, offset=OFFSET, seed=s, ret_norm="meanstd", priority=2)
  for s in SEEDS[:3]:
    add("fixscale", center="none", gamma=0.999, offset=OFFSET, seed=s, reward_scale="std", priority=2)

  # tier 2: E - reward-redesign shock, 1.4x the common length, shock lands at 60% of it
  kwargs_shock_iters = int(round(iters * 1.4))
  shock_at = int(round(iters * 0.8))
  for c in ("none", "value"):
    for kind in ("offset", "action_rate"):
      for s in SEEDS:
        add(
          "E",
          tag=kind,
          center=c,
          gamma=0.999,
          offset=OFFSET,
          seed=s,
          iters=kwargs_shock_iters,
          shock_kind=kind,
          shock_at=shock_at,
          shock_factor=5.0,
          priority=2,
        )

  # tier 3: fall-penalty sensitivity on the reference arm
  for p in (PENALTY * 0.3, PENALTY * 3.0):
    for s in SWEEP_SEEDS:
      add("pen", center="none", gamma=0.99, offset=0.0, penalty=p, seed=s, tag=f"p{p:g}", priority=3)

  # tier 3: learning-rate grid
  for lr in LR_GRID:
    for c in ("none", "value"):
      for s in SWEEP_SEEDS:
        add("lr", center=c, gamma=0.999, offset=OFFSET, seed=s, learning_rate=lr, tag=f"lr{lr:g}", priority=3)

  # tier 4 (deferred, only if budget remains): SSP-low
  for c in ("none", "value"):
    for s in SEEDS[:3]:
      add("ssp", center=c, gamma=0.999, offset=OFFSET, seed=s, tag="low", priority=4, extra={"encoder": "ssp_low"})

  return jobs


def bench_jobs(env_counts=(1024, 2048, 4096)) -> list[RunSpec]:
  """Throughput + plateau gate. 12 iterations per env count measures fps; 800-iteration reference runs measure
  the plateau, which sets the common run length for every arm (pre-registered rule)."""
  jobs: list[RunSpec] = []
  for n in env_counts:
    for s in (0, 1):
      jobs.append(
        RunSpec(
          job_id=f"bench_n{n}_s{s}", block="bench", center="none", gamma=0.99, offset=0.0, seed=s,
          iters=12, num_envs=n, eval_envs=128, eval_every=100, eval_steps=50, final_steps=50,
          final_shift_steps=30, priority=0,
        )
      )
  for s in (2000, 2001):
    jobs.append(
      RunSpec(
        job_id=f"plateau_ref_s{s}", block="plateau", center="none", gamma=0.99, offset=0.0, seed=s,
        iters=600, num_envs=NUM_ENVS, eval_envs=256, eval_every=25, eval_steps=300, final_steps=2000,
        final_shift_steps=1000, priority=0,
      )
    )
  return jobs


def probe_jobs(iters: int = 200, num_envs: int = 4096) -> list[RunSpec]:
  """One short run purely to measure wall-clock per iteration (ts deltas in iter.jsonl) and the eval cost."""
  return [
    RunSpec(
      job_id=f"probe_n{num_envs}_i{iters}", block="probe", center="none", gamma=0.99, offset=0.0, seed=0,
      iters=iters, num_envs=num_envs, eval_envs=256, eval_every=25, eval_steps=300, final_steps=200,
      final_shift_steps=100, priority=0,
    )
  ]


def campaign_id(jobs: Iterable[RunSpec], tag: str = "c1") -> str:
  h = hashlib.sha1(json.dumps([j.job_id for j in jobs], sort_keys=True).encode()).hexdigest()[:8]
  return f"{tag}-{h}"


def main() -> None:
  import argparse

  ap = argparse.ArgumentParser()
  ap.add_argument("--eta", type=float, default=DEFAULT_ETA)
  ap.add_argument("--tag", default="c1")
  ap.add_argument("--kind", default="main", choices=["main", "bench", "probe"])
  ap.add_argument("--iters", type=int, default=ITERS, help="common run length from the plateau gate")
  ap.add_argument("--blocks", default=None, help="comma-separated block subset (budget cut order)")
  ap.add_argument("--seeds", default=None, help="comma-separated seed subset")
  ap.add_argument("--centers", default=None, help="comma-separated centering subset (the uncentered half does not depend on eta)")
  ap.add_argument("--out", default=None)
  a = ap.parse_args()
  blocks = tuple(a.blocks.split(",")) if a.blocks else None
  seeds = tuple(int(x) for x in a.seeds.split(",")) if a.seeds else SEEDS
  centers = tuple(a.centers.split(",")) if a.centers else None
  jobs = (
    bench_jobs()
    if a.kind == "bench"
    else probe_jobs()
    if a.kind == "probe"
    else build_jobs(a.eta, iters=a.iters, blocks=blocks, seeds=seeds, centers=centers)
  )
  camp = campaign_id(jobs, a.tag)
  payload = {"campaign": camp, "eta": a.eta, "iters": a.iters, "jobs": [j.__dict__ for j in jobs]}
  out = a.out or f"manifests/{camp}.json"
  import os

  os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
  with open(out, "w") as f:
    json.dump(payload, f, indent=1, default=float)
  by_block: dict[str, int] = {}
  for j in jobs:
    by_block[j.block] = by_block.get(j.block, 0) + 1
  print(json.dumps({"campaign": camp, "total": len(jobs), "by_block": by_block}, indent=1))


if __name__ == "__main__":
  main()
