"""Manifests for REPORT_004 (pre-registered in docs/mjlab_centering/02-decisions.md, d101 onward).

    python jobs_r4.py sweep            # Step 2: Go1 offset sweep c in {0.1, 0.3} x {none, value}, seeds 3000-3009
    python jobs_r4.py shock            # Step 3: mothers + branches, Go1 offset 0, seeds 3000-3009
    python jobs_r4.py width            # Step 4: SSP width stage, search seeds 1000-1002
    python jobs_r4.py topup-sweep 0.1  # extension to seeds 3010-3019 for one offset (only if the Step-2 rule fires)
    python jobs_r4.py topup-shock 3010-3019 off+0.1,actx5   # extension of the shock block (rule-driven)
    python jobs_r4.py clip20           # clip-off 20 seeds (3003-3019; 3000-3002 are the pilot), only if d100 fires
    python jobs_r4.py off0-20          # Go1 offset-0 top-up, seeds 3010-3019, {none, value}
    python jobs_r4.py g1-20            # G1 gamma .99 offset +1, penalty 5.7, fresh seeds 3000-3019 x {none, value}
    python jobs_r4.py ssp-confirm <enc> <n_rot> <n_scale>   # best SSP config x {none, value}, seeds 3000-3019
    python jobs_r4.py rff-confirm <n_rot> <n_scale>         # matched random-Fourier control, centered, seeds 3000-3019

Every job is cloned from the corrected-protocol E11 job (`w2main_*`, night1) so that settings cannot drift from the
cells these runs are paired with; only the fields named below are overridden. Centered arms: value centering,
eta 0.03, r_bar initialised from the first batch. gamma .999, 2048 envs, 2000 iterations unless stated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NIGHT1 = ROOT / "manifests" / "night1.json"

CLAIM_SEEDS = tuple(range(3000, 3020))
SEARCH_SEEDS = (1000, 1001, 1002)
# d1002: claim-grade blocks save their final policy to R2 (model_final.pt, 4.7 MB per run) so videos and re-evaluation never
# need a retrain. Search/probe/width/pilot/escalation blocks (r4width*, r6probe, r6ls*, r6sspp, r6sspe) stay off.
CLAIM_CKPT_BLOCKS = frozenset({"r4sweep", "r4clip", "r4off0", "r4g1", "r4sspc", "r5bias", "r6g1", "r7vid"})
SHOCK_AT = 1200
BRANCH_ITERS = 1600  # 400 iterations after the shock
G1_KW_VID = dict(robot="g1", gamma=0.99, offset=1.0, penalty=5.7)
SHOCKS = {  # name -> overrides
  "off+0.1": dict(shock_kind="offset", extra={"shock_delta": 0.1}),
  "off+1.0": dict(shock_kind="offset", extra={"shock_delta": 1.0}),
  "actx5": dict(shock_kind="action_rate", shock_factor=5.0),
  "ctrl": dict(shock_kind="none"),
}


def _template(center: str) -> dict:
  jobs = json.loads(NIGHT1.read_text())["jobs"]
  hit = [j for j in jobs if j["job_id"] == f"w2main_go1_raw_{center}_g0.999_o1_s3000"]
  assert len(hit) == 1, center
  return copy.deepcopy(hit[0])


def make(block: str, center: str, seed: int, tag: str, **over) -> dict:
  j = _template(center)
  extra = dict(j.get("extra") or {})
  extra.update(over.pop("extra", {}) or {})
  j["upload_ckpt"] = block in CLAIM_CKPT_BLOCKS
  j.update(over)
  j.update(block=block, seed=int(seed), extra=extra)
  j["job_id"] = f"{block}_{tag}_{center}_s{seed}"
  j["notes"] = "report4"
  return j


def video_runs() -> list[dict]:
  """d1002: four illustrative retrains for the supervisor videos: the median centered claim seed per robot (Go1 3002, G1 3012;
  lower median of the logged last-5 tracking over the 20 claim seeds) and its uncentered partner, same config as the claim
  cell, checkpoints uploaded. Not claim data; block r7vid is never read by the analysis scripts."""
  out = [make("r7vid", c, 3002, "go1_g0.999_o1", **{}) for c in ("none", "value")]
  out += [make("r7vid", c, 3012, "g1_g0.99_o1", iters=2500, **G1_KW_VID) for c in ("none", "value")]
  return out


def sweep(offsets=(0.1, 0.3), seeds=tuple(range(3000, 3010))) -> list[dict]:
  return [make("r4sweep", c, s, f"go1_g0.999_o{o:g}", offset=float(o)) for o in offsets for s in seeds for c in ("none", "value")]


def mother_id(center: str, seed: int) -> str:
  return f"r4mother_go1_g0.999_o0_{center}_s{seed}"


def shock(seeds=tuple(range(3000, 3010)), kinds=tuple(SHOCKS), mothers=True) -> list[dict]:
  out: list[dict] = []
  if mothers:
    for s in seeds:
      for c in ("none", "value"):
        j = make("r4mother", c, s, "go1_g0.999_o0", offset=0.0, iters=SHOCK_AT, save_at=SHOCK_AT, upload_ckpt=True)
        assert j["job_id"] == mother_id(c, s)
        out.append(j)
  for s in seeds:  # branches ordered by mother so each waits for the earliest-finished mother
    for c in ("none", "value"):
      for k in kinds:
        over = copy.deepcopy(SHOCKS[k])
        extra = {"after": mother_id(c, s), **over.pop("extra", {})}
        out.append(make("r4branch", c, s, f"go1_g0.999_o0_{k}", offset=0.0, iters=BRANCH_ITERS, shock_at=SHOCK_AT,
                        eval_every=25, extra=extra, **over))
  return out


def width(seeds=SEARCH_SEEDS) -> list[dict]:
  """Raw, SSP-low and matched random-Fourier code at 2x2 (33 features/term) and 4x4 (129), all centered, offset +1."""
  out = []
  for s in seeds:
    out.append(make("r4width", "value", s, "go1_raw"))
    for n in (2, 4):
      for enc in ("ssp_low", "rff_low"):
        out.append(make("r4width", "value", s, f"go1_{enc}{n}x{n}",
                        extra={"encoder": enc, "ssp_n_rotates": n, "ssp_n_scales": n, "ssp_length_scale": 0.3,
                               "ssp_rff_seed": int(s)}))
  return out


def width8(seeds=SEARCH_SEEDS) -> list[dict]:
  """d105(iv): the 8x8 pilot, added only if 4x4 won the width stage. 513 features per term, both codes."""
  return [make("r4width8", "value", s, f"go1_{enc}8x8",
               extra={"encoder": enc, "ssp_n_rotates": 8, "ssp_n_scales": 8, "ssp_length_scale": 0.3,
                      "ssp_rff_seed": int(s)})
          for s in seeds for enc in ("ssp_low", "rff_low")]


def topup_sweep(offset: float) -> list[dict]:
  return sweep(offsets=(offset,), seeds=tuple(range(3010, 3020)))


def clip20() -> list[dict]:
  return [make("r4clip", "none", s, "go1_clipoff_g0.999_o1", value_clip=False) for s in range(3003, 3020)]


def off0_20() -> list[dict]:
  return [make("r4off0", c, s, "go1_g0.999_o0", offset=0.0) for s in range(3010, 3020) for c in ("none", "value")]


def g1_20() -> list[dict]:
  return [make("r4g1", c, s, "g1_g0.99_o1", robot="g1", gamma=0.99, offset=1.0, penalty=5.7)
          for s in CLAIM_SEEDS for c in ("none", "value")]


# ------------------------------------------------------------------ round 5 (REPORT_005, d113-d121)
MOTHER_CAMPAIGN = "r4topupshock-049efe29"  # d116: the primary copy of the 3010-3019 mothers
SWEEP_CUT = ("r4sweep_go1_g0.999_o0.1_value_s3009", "r4sweep_go1_g0.999_o0.3_none_s3009",
             "r4sweep_go1_g0.999_o0.3_value_s3009")  # the three round-4 cells the cap cut
WIDTH_RAW_DONE = tuple(f"r4width_go1_raw_value_s{s}" for s in SEARCH_SEEDS)  # finished in round 4, reused (d118)


def r5bias(seeds=(3000, 3001, 3002)) -> list[dict]:
  """Step 0 (d114): E11's uncentered cell with the critic's output bias set to c / (1 - gamma)."""
  return [make("r5bias", "none", s, "go1_g0.999_o1", extra={"critic_bias_init": "offset"}) for s in seeds]


def sweep_top20(offsets=(0.1, 0.3)) -> list[dict]:
  """Step 1 (d115): the three cut cells, then seeds 3010-3019 for both offsets and both arms (43 jobs)."""
  cut = [j for j in sweep() if j["job_id"] in SWEEP_CUT]
  assert len(cut) == 3
  return cut + sweep(offsets=offsets, seeds=tuple(range(3010, 3020)))


def shock_branches(seeds=tuple(range(3010, 3020)), mother_campaign: str = MOTHER_CAMPAIGN) -> list[dict]:
  """Step 2 (d116): branches only; the mothers already sit on R2 in `mother_campaign`."""
  out = []
  for j in shock(seeds=seeds, mothers=False):
    j["extra"] = {**j["extra"], "after_campaign": mother_campaign}
    out.append(j)
  return out


def width_rest() -> list[dict]:
  return [j for j in width() if j["job_id"] not in WIDTH_RAW_DONE]


def g1_timing() -> list[dict]:
  """Step 6 timing batch (d120): seeds 3000-3002 x both arms, part of the claim set."""
  return [j for j in g1_20() if j["seed"] in (3000, 3001, 3002)]


def g1_rest() -> list[dict]:
  t = {j["job_id"] for j in g1_timing()}
  return [j for j in g1_20() if j["job_id"] not in t]


def r5all() -> list[dict]:
  """One queue for one scarce pod: every Go1 step of round 5 in priority order (d113 order, d122 scarcity).

  A captured community pod is the bottleneck (one in ~2 h), so a pod must do as much as possible rather than one shard.
  podrun runs the manifest in order with --procs workers, and every finished job mirrors to R2 with a DONE marker, so a
  preempted pod loses only the in-flight jobs.
  """
  out = r5bias() + shock_branches() + off0_20() + sweep_top20()
  assert len(out) == 3 + 80 + 20 + 43, len(out)
  return out


def shard(jobs: list[dict], i: int, n: int) -> list[dict]:
  """Disjoint shards for one-pod-per-shard runs. Branches stay with the pod that holds their mother's checkpoint."""
  groups: dict[str, int] = {}
  out = []
  for j in jobs:
    key = f"{j['seed']}_{j['center']}" if j["block"] == "r4branch" else j["job_id"]
    g = groups.setdefault(key, len(groups))
    if g % n == i:
      out.append(j)
  return out


def ssp_confirm(enc: str, n_rot: int, n_scale: int) -> list[dict]:
  ex = {"encoder": enc, "ssp_n_rotates": n_rot, "ssp_n_scales": n_scale, "ssp_length_scale": 0.3}
  return [make("r4sspc", c, s, f"go1_{enc}{n_rot}x{n_scale}_g0.999_o1", extra={**ex, "ssp_rff_seed": int(s)})
          for s in CLAIM_SEEDS for c in ("none", "value")]


def rff_confirm(n_rot: int, n_scale: int) -> list[dict]:
  ex = {"encoder": "rff_low", "ssp_n_rotates": n_rot, "ssp_n_scales": n_scale, "ssp_length_scale": 0.3}
  return [make("r4sspc", "value", s, f"go1_rff_low{n_rot}x{n_scale}_g0.999_o1", extra={**ex, "ssp_rff_seed": int(s)})
          for s in CLAIM_SEEDS]


# ------------------------------------------------------------------ round 6 (REPORT_006, d126 onward)
G1_KW = dict(robot="g1", gamma=0.99, offset=1.0, penalty=5.7)  # the d98 / d120 G1 cell
G1_PROBE_ITERS = 3000
G1_PROBE_EVAL_EVERY = 25  # d126: the 250-iteration slope window then holds 11 evals instead of 6
BIAS = {"critic_bias_init": "offset"}  # critic output bias = offset / (1 - gamma) = 100 for G1 at gamma .99


def g1_probe(seeds=SEARCH_SEEDS) -> list[dict]:
  """d126 Step 1: centered and bias-init G1, search seeds, 3000 iterations, evals every 25. 6 jobs."""
  out = []
  for s in seeds:
    out.append(make("r6probe", "value", s, "g1_g0.99_o1", iters=G1_PROBE_ITERS, eval_every=G1_PROBE_EVAL_EVERY, **G1_KW))
    out.append(make("r6probe", "none", s, "g1_g0.99_o1_bias", iters=G1_PROBE_ITERS, eval_every=G1_PROBE_EVAL_EVERY,
                    extra=dict(BIAS), **G1_KW))
  return out


def g1_claim(iters: int, seeds=CLAIM_SEEDS) -> list[dict]:
  """d127 Step 2: fresh seeds x {none, value, bias-init}; `iters` comes from analysis/g1_slope.py (2000 or 3000)."""
  out = []
  for s in seeds:
    out.append(make("r6g1", "none", s, "g1_g0.99_o1", iters=int(iters), **G1_KW))
    out.append(make("r6g1", "value", s, "g1_g0.99_o1", iters=int(iters), **G1_KW))
    out.append(make("r6g1", "none", s, "g1_g0.99_o1_bias", iters=int(iters), extra=dict(BIAS), **G1_KW))
  return out


def g1_claim_timing(iters: int) -> list[dict]:
  """d127 cost gate: seeds 3000-3001 x all three arms (6 jobs) are measured before the other 54 launch."""
  return [j for j in g1_claim(iters) if j["seed"] in (3000, 3001)]


def g1_queue(iters: int) -> list[dict]:
  """One queue for the claim set: the 6 timing jobs first (they fix $/job), then the other 54 (d128)."""
  t = g1_claim_timing(iters)
  for j in t:
    j["priority"] = 0
  rest = g1_claim_rest(iters)
  for j in rest:
    j["priority"] = 1
  out = t + rest
  assert len(out) == 60, len(out)
  return out


def g1_claim_rest(iters: int) -> list[dict]:
  t = {j["job_id"] for j in g1_claim_timing(iters)}
  return [j for j in g1_claim(iters) if j["job_id"] not in t]


SSP_CELL_KEYS = ("enc", "n_rot", "n_scale", "ls")


def _ssp_extra(enc: str, n_rot: int, n_scale: int, ls: float, seed: int) -> dict:
  return {"encoder": enc, "ssp_n_rotates": int(n_rot), "ssp_n_scales": int(n_scale), "ssp_length_scale": float(ls),
          "ssp_rff_seed": int(seed)}


def ssp_ls_check(n: int, lengths=(0.15, 0.6), seeds=SEARCH_SEEDS) -> list[dict]:
  """d128: the conditional length-scale check. The best Stage-A width at two more length scales, same search seeds,
  centered, offset +1. 6 jobs for one width."""
  return [make("r6ls", "value", s, f"go1_ssp_low{n}x{n}_ls{ls:g}", extra=_ssp_extra("ssp_low", n, n, ls, s))
          for s in seeds for ls in lengths]


LS_SWEEP = (0.03, 0.1, 0.3, 1.0)


def ssp_ls_sweep(n: int, lengths=LS_SWEEP, seeds=SEARCH_SEEDS) -> list[dict]:
  """d138: the best width at a log-spaced length scale, unconditionally. ls 0.3 is the width-stage job, so this
  returns only the three new values (9 jobs for one width)."""
  return [make("r6lss", "value", s, f"go1_ssp_low{n}x{n}_ls{ls:g}", extra=_ssp_extra("ssp_low", n, n, ls, s))
          for s in seeds for ls in lengths if float(ls) != 0.3]


def ssp_pilot(n: int, ls: float, seeds=tuple(range(3000, 3005)), offsets=(0.1, 1.0), centers=("none", "value"),
              block: str = "r6sspp") -> list[dict]:
  """d129: Stage B, best SSP x {none, value} x offset {0.1, 1}, seeds 3000-3004 (20 jobs); also the escalation builder
  (seeds 3005-3019 for the cells that trip the rule). gamma .999, 2000 iterations, corrected protocol."""
  return [make(block, c, s, f"go1_ssp_low{n}x{n}_ls{ls:g}_g0.999_o{o:g}", offset=float(o),
               extra=_ssp_extra("ssp_low", n, n, ls, s))
          for o in offsets for c in centers for s in seeds]


def ssp_width_queue(widths=(2, 4)) -> list[dict]:
  """d146: width stage + the length-scale sweep for *both* candidate widths, in one queue.

  The sweep is only *needed* at the best width, but a pod runs the manifest it downloaded and cannot be retargeted, and
  the sweep jobs start after the twelve width jobs - by which time the winning width is known to a human but not to the
  running pod. Carrying **both widths' sweeps** (18 jobs instead of 9) is therefore the only way to guarantee that the
  sweep runs at the width that won, in a single capture. The 9 wasted jobs are about $0.6 at the measured rate, which is
  cheaper than the extra capture a retarget would need. Selection is unchanged and pre-registered: best width by mean
  tracking at ls 0.3 (ties to the narrower code), then the best length scale within that width (d105, d138); the Stage-B
  gate uses the best length scale from the sweep of the winning width.
  """
  out = width_rest()
  for n in widths:
    out += ssp_ls_sweep(n)
  for j in out[:12]:
    j["priority"] = 1
  for j in out[12:]:
    j["priority"] = 2
  assert len(out) == 12 + 9 * len(widths), len(out)
  return out


def r7_queue(save_every: int = 250) -> list[dict]:
  """Round 7 (d1020-d1022): the pre-registered Stage-B escalation for one cell, plus the length-scale boundary check.

  * escalation: SSP 2x2 @ ls 1.0, uncentered, offset 0.1, seeds 3005-3019 (15 jobs), paired with the existing 20-seed raw
    cell on seeds 3000-3019.
  * boundary: SSP 2x2, centered, offset 1, length scales 2.0 and 4.0, search seeds 1000-1002 (6 jobs), because 1.0 won the
    sweep at its upper edge and the optimum may lie above it.
  Both carry `save_every` so a preempted pod resumes from its rolling checkpoint (d1019).
  """
  esc = ssp_pilot(2, 1.0, seeds=tuple(range(3005, 3020)), offsets=(0.1,), centers=("none",), block="r6sspe")
  bnd = ssp_ls_sweep(2, lengths=(2.0, 4.0))
  for j in bnd:
    j["block"] = "r7bnd"
  for j in esc:
    j["priority"] = 1
  for j in bnd:
    j["priority"] = 2
  out = esc + bnd
  for j in out:
    j["save_every"] = int(save_every)
    j["notes"] = "round7"
  assert len(out) == 21, len(out)
  return out


def r6a() -> list[dict]:
  """One queue for the first GPU pod: the G1 probe first (it gates the biggest spend), then the SSP width stage."""
  probe = g1_probe()
  for j in probe:
    j["priority"] = 0
  out = probe + width_rest()
  for j in out:
    j.setdefault("priority", 1)
  assert len(out) == 6 + 12, len(out)
  return out


def check(jobs: list[dict]) -> None:
  ids = [j["job_id"] for j in jobs]
  assert len(ids) == len(set(ids)), "duplicate job ids"
  for j in jobs:
    assert j["time_outs"] and j["falls_reset"], j["job_id"]  # corrected protocol
    if j["center"] == "value":
      assert j["eta"] == 0.03 and j["r_bar_init"] == "first_batch", j["job_id"]
    else:
      assert j["center"] == "none"
    ex = j.get("extra") or {}
    after = ex.get("after")
    if after and not ex.get("after_campaign"):  # branch-only manifests (d116) take the mother from another campaign
      assert after in ids and ids.index(after) < ids.index(j["job_id"]), "mother must precede its branch"


def campaign_id(tag: str, jobs: list[dict]) -> str:
  return f"{tag}-{hashlib.sha1(json.dumps([j['job_id'] for j in jobs]).encode()).hexdigest()[:8]}"


R7_WHAT = {"video-runs"}
R6_WHAT = {"r6a", "g1-probe", "g1-claim", "g1-queue", "g1-claim-timing", "g1-claim-rest", "ssp-ls-check", "ssp-ls-sweep", "r7-queue", "ssp-width-queue", "ssp-pilot", "ssp-escalate"}
R5_WHAT = {"r5all", "r5bias", "r5bias-ext", "sweep-top20", "shock-branches", "off0-20", "width-rest", "width8", "g1-timing",
           "g1-rest", "ssp-confirm", "rff-confirm"}


def main(argv: list[str]) -> None:
  shard_i = shard_n = None
  if "--shard" in argv:  # `... --shard i n`, anywhere at the end; applies to every builder
    k = argv.index("--shard")
    shard_i, shard_n = int(argv[k + 1]), int(argv[k + 2])
    argv = argv[:k] + argv[k + 3:]
  what = argv[0]
  if what == "sweep":
    jobs = sweep()
  elif what == "shock":
    jobs = shock()
  elif what == "shock-shard":  # disjoint seed groups, one pod each; mothers stay with their branches
    i, n = int(argv[1]), int(argv[2])
    seeds = tuple(range(3000, 3010))
    mine = seeds[i::n]
    jobs = shock(seeds=mine)
  elif what == "width":
    jobs = width()
  elif what == "sweep-shard":  # disjoint seed groups: one pod per shard, 6 procs each (GPU-bound, cheapest per iteration)
    i, n = int(argv[1]), int(argv[2])
    jobs = sweep(seeds=tuple(range(3000, 3010))[i::n])
  elif what == "width-shard":
    i, n = int(argv[1]), int(argv[2])
    jobs = width(seeds=tuple(SEARCH_SEEDS)[i::n])
  elif what == "width8":
    jobs = width8()
  elif what == "topup-sweep":
    jobs = topup_sweep(float(argv[1]))
  elif what == "topup-shock":
    lo, hi = (int(x) for x in argv[1].split("-"))
    jobs = shock(seeds=tuple(range(lo, hi + 1)), kinds=tuple(argv[2].split(",")) + ("ctrl",))
  elif what == "video-runs":
    jobs = video_runs()
  elif what == "clip20":
    jobs = clip20()
  elif what == "off0-20":
    jobs = off0_20()
  elif what == "g1-20":
    jobs = g1_20()
  elif what == "ssp-confirm":
    jobs = ssp_confirm(argv[1], int(argv[2]), int(argv[3]))
  elif what == "rff-confirm":
    jobs = rff_confirm(int(argv[1]), int(argv[2]))
  elif what == "r5all":
    jobs = r5all()
  elif what == "r5bias":
    jobs = r5bias()
  elif what == "r5bias-ext":  # 3 of 3 walked in the pilot, so the arm goes to 20 seeds (d124)
    jobs = r5bias(seeds=tuple(range(3003, 3020)))
  elif what == "sweep-top20":
    jobs = sweep_top20()
  elif what == "shock-branches":
    jobs = shock_branches()
  elif what == "width-rest":
    jobs = width_rest()
  elif what == "g1-timing":
    jobs = g1_timing()
  elif what == "g1-rest":
    jobs = g1_rest()
  elif what == "r7-queue":
    jobs = r7_queue()
  elif what == "ssp-width-queue":
    jobs = ssp_width_queue()
  elif what == "r6a":
    jobs = r6a()
  elif what == "g1-probe":
    jobs = g1_probe()
  elif what == "g1-claim":  # g1-claim <iters>
    jobs = g1_claim(int(argv[1]))
  elif what == "g1-queue":  # g1-queue <iters>
    jobs = g1_queue(int(argv[1]))
  elif what == "g1-claim-timing":
    jobs = g1_claim_timing(int(argv[1]))
  elif what == "g1-claim-rest":
    jobs = g1_claim_rest(int(argv[1]))
  elif what == "ssp-ls-sweep":  # ssp-ls-sweep <n>
    jobs = ssp_ls_sweep(int(argv[1]))
  elif what == "ssp-ls-check":  # ssp-ls-check <n>
    jobs = ssp_ls_check(int(argv[1]))
  elif what == "ssp-pilot":  # ssp-pilot <n> <length_scale>
    jobs = ssp_pilot(int(argv[1]), float(argv[2]))
  elif what == "ssp-escalate":  # ssp-escalate <n> <length_scale> <offset> <none|value|none,value>
    jobs = ssp_pilot(int(argv[1]), float(argv[2]), seeds=tuple(range(3005, 3020)), offsets=(float(argv[3]),),
                     centers=tuple(argv[4].split(",")), block="r6sspe")
  else:
    raise SystemExit(__doc__)
  if shard_n:
    jobs = shard(jobs, shard_i, shard_n)
  check(jobs)
  prefix = "r7" if what in R7_WHAT else "r6" if what in R6_WHAT else "r5" if what in R5_WHAT else "r4"
  tag = what.replace("-", "") if not shard_n else f"{what.replace('-', '')}s{shard_i}of{shard_n}"
  camp = campaign_id(f"{prefix}{tag}" if not tag.startswith(("r5", "r6", "r7")) else tag, jobs)
  out = ROOT / "manifests" / f"{camp}.json"
  out.write_text(json.dumps({"campaign": camp, "jobs": jobs}, indent=1))
  by: dict[str, int] = {}
  for j in jobs:
    by[j["block"]] = by.get(j["block"], 0) + 1
  iters = sum(j["iters"] - (0 if not (j.get("extra") or {}).get("after") else j["shock_at"]) for j in jobs)
  print(json.dumps({"campaign": camp, "manifest": str(out), "jobs": len(jobs), "by_block": by, "total_iterations": iters}))


if __name__ == "__main__":
  main(sys.argv[1:])
