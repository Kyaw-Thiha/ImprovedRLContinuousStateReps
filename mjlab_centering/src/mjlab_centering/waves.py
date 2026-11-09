"""Wave manifests (docs/mjlab_centering/02-decisions.md entries 35-42).

Wave 1  tuning on search seeds 1000-1002: eta (+ r_bar init), SSP length scale, G1 eta.
Wave 2  Go1, claim seeds 2000-2009: core 2x2x2 (representation x centering x gamma, offset c), critic-fix rivals,
        shock branches (mother = the raw gamma=0.999 core run of the same seed, branched at `shock_at`).
Wave 3  G1 (its own tuned defaults), claim seeds 2000-2009: 2x2 at gamma 0.999 plus the raw gamma=0.99 tie check.
Wave 4  seeds 2010-2019 for the four gamma=0.999 core cells per robot (run only if the precision rule fires).

Run length (`iters`), the shock iteration, eta, the SSP length scale and the G1 eta are arguments because they are
frozen from the gate / wave-1 results before the later waves are generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mjlab_centering.train import RunSpec

SEARCH_SEEDS = (1000, 1001, 1002)
CLAIM_SEEDS = tuple(range(2000, 2010))
TOPUP_SEEDS = tuple(range(2010, 2020))
OFFSET = 1.0
PENALTY = 10.0
EVAL = dict(eval_envs=256, eval_every=50, eval_steps=300, final_steps=0, final_shift_steps=0)


def _spec(job_id, block, robot, seed, center, gamma, iters, eta, rep="raw", ssp_ls=0.3, offset=OFFSET,
          penalty=PENALTY, **kw):
  extra = dict(kw.pop("extra", {}))
  if rep != "raw":
    extra.update({"encoder": rep, "ssp_length_scale": ssp_ls})
  return RunSpec(job_id=job_id, block=block, robot=robot, seed=seed, center=center, gamma=gamma, offset=offset,
                 penalty=penalty, iters=iters, eta=eta, num_envs=2048, extra=extra, **EVAL, **kw)


def core_id(robot, rep, center, gamma, seed):
  return f"core_{robot}_{rep}_{center}_g{gamma}_s{seed}"


def wave1(iters: int, etas=(0.01, 0.03, 0.1, 0.3), ssp_lengths=(0.15, 0.3, 0.6), g1_etas=(0.01, 0.03, 0.1),
          init_variant_eta: float = 0.03) -> list[RunSpec]:
  jobs: list[RunSpec] = []
  for s in SEARCH_SEEDS:
    for e in etas:
      jobs.append(_spec(f"eta_go1_zero_eta{e:g}_s{s}", "eta", "go1", s, "value", 0.999, iters, e))
    jobs.append(_spec(f"eta_go1_first_eta{init_variant_eta:g}_s{s}", "eta", "go1", s, "value", 0.999, iters,
                      init_variant_eta, r_bar_init="first_batch"))
    for ls in ssp_lengths:
      jobs.append(_spec(f"sslen_go1_ls{ls:g}_s{s}", "sslen", "go1", s, "none", 0.99, iters, 0.0, rep="ssp_low", ssp_ls=ls))
    for e in g1_etas:
      jobs.append(_spec(f"eta_g1_zero_eta{e:g}_s{s}", "g1eta", "g1", s, "value", 0.999, iters, e))
  return jobs


def wave2(iters: int, shock_at: int, eta: float, ssp_ls: float, seeds=CLAIM_SEEDS, shock_factor: float = 5.0,
          r_bar_init: str = "first_batch", include_ssp: bool = True,
          shock_kinds: tuple[str, ...] = ("none", "offset", "action_rate"),
          gammas: tuple[float, ...] = (0.99, 0.999)) -> list[RunSpec]:
  """`include_ssp=False` emits only the raw-input cells (core, rivals, shock). Used to start the centering main effect
  before wave 5 has decided whether the SSP cells are informative at this width (decision 48)."""
  assert 0 < shock_at < iters
  jobs: list[RunSpec] = []
  for s in seeds:
    for rep in (("raw", "ssp_low") if include_ssp else ("raw",)):
      for center in ("none", "value"):
        for gamma in gammas:
          kw = {}
          if rep == "raw" and gamma == 0.999:
            kw["save_at"] = shock_at  # mother of the shock branches
          jobs.append(_spec(core_id("go1", rep, center, gamma, s), "core", "go1", s, center, gamma, iters,
                            eta if center != "none" else 0.0, rep=rep, ssp_ls=ssp_ls,
                            r_bar_init=r_bar_init if center != "none" else "zero", **kw))
  for s in seeds:
    jobs.append(_spec(f"rival_clipoff_s{s}", "rival", "go1", s, "none", 0.999, iters, 0.0, value_clip=False))
    jobs.append(_spec(f"rival_retnorm_s{s}", "rival", "go1", s, "none", 0.999, iters, 0.0, ret_norm="meanstd"))
  for s in seeds:
    for center in ("none", "value"):
      mother = core_id("go1", "raw", center, 0.999, s)
      for kind in shock_kinds:  # "none" = resumed-but-unshocked control
        jobs.append(_spec(f"shock_{kind}_{center}_s{s}", "shock", "go1", s, center, 0.999, iters,
                          eta if center != "none" else 0.0, shock_kind=kind, shock_at=shock_at,
                          shock_factor=shock_factor, r_bar_init=r_bar_init if center != "none" else "zero",
                          extra={"after": mother}))
  return jobs


def wave3(iters: int, eta: float, ssp_ls: float, seeds=CLAIM_SEEDS, r_bar_init: str = "first_batch") -> list[RunSpec]:
  jobs: list[RunSpec] = []
  for s in seeds:
    for rep in ("raw", "ssp_low"):
      for center in ("none", "value"):
        jobs.append(_spec(core_id("g1", rep, center, 0.999, s), "core_g1", "g1", s, center, 0.999, iters,
                          eta if center != "none" else 0.0, rep=rep, ssp_ls=ssp_ls,
                          r_bar_init=r_bar_init if center != "none" else "zero"))
    for center in ("none", "value"):
      jobs.append(_spec(core_id("g1", "raw", center, 0.99, s), "tie_g1", "g1", s, center, 0.99, iters,
                        eta if center != "none" else 0.0,
                        r_bar_init=r_bar_init if center != "none" else "zero"))
  return jobs


def wave4(iters: int, eta_go1: float, eta_g1: float, ssp_ls: float) -> list[RunSpec]:
  jobs: list[RunSpec] = []
  for s in TOPUP_SEEDS:
    for robot, eta in (("go1", eta_go1), ("g1", eta_g1)):
      for rep in ("raw", "ssp_low"):
        for center in ("none", "value"):
          jobs.append(_spec(core_id(robot, rep, center, 0.999, s), f"core_{robot}_topup", robot, s, center, 0.999,
                            iters, eta if center != "none" else 0.0, rep=rep, ssp_ls=ssp_ls))
  return jobs


def diag_width(iters: int, eta: float, ssp_ls: float, seeds=SEARCH_SEEDS) -> list[RunSpec]:
  """Equal-width control for the SSP arm: raw vs random-Fourier-at-the-same-width vs HexSSP, all 1068-wide when encoded.

  The SSP arm's adaptive learning rate collapses (KL-driven) because its input is 22x wider than raw's; without this
  control "SSP hurts" and "a wider input takes smaller steps" cannot be separated.
  """
  jobs: list[RunSpec] = []
  for s in seeds:
    for rep in ("raw", "rff_low", "ssp_low"):
      jobs.append(_spec(f"diagw_go1_{rep}_s{s}", "diagw", "go1", s, "none", 0.99, iters, 0.0,
                        rep=rep, ssp_ls=ssp_ls))
  return jobs


def g1_length_probe(iters: int, seeds=(1000, 1001)) -> list[RunSpec]:
  """Does G1 learn at all, and at what length? Standing baseline measured locally: J_task 0.0436, track_lin 0.267,
  10.3 falls/1k. At 800 iterations the G1 arms sit at J_task 0.023-0.040, i.e. at or below standing, so 800 is too
  short for the humanoid and any G1 centering/SSP contrast at that length is uninformative."""
  return [_spec(f"g1len_raw_none_g0.99_s{s}", "g1len", "g1", s, "none", 0.99, iters, 0.0) for s in seeds]


def eta_confirm(eta: float, iters: int, seed: int = 1002) -> list[RunSpec]:
  """Third search seed for the eta cells that were still open when the pod queues were re-tasked."""
  return [
    _spec(f"eta_go1_zero_eta{eta:g}_s{seed}", "eta", "go1", seed, "value", 0.999, iters, eta),
    _spec(f"eta_go1_first_eta{eta:g}_s{seed}", "eta", "go1", seed, "value", 0.999, iters, eta, r_bar_init="first_batch"),
  ]


def diag_width_centered(iters: int, eta: float, ssp_ls: float, seeds=SEARCH_SEEDS) -> list[RunSpec]:
  """Equal-width representation diagnostic WITH centering on.

  The first version of this diagnostic (wave 5) ran all three arms uncentered at offset c and every arm failed
  (J_task 0.054-0.073, tracking 0.30-0.32, the standing floor) - which is the offset prediction, not a representation
  effect. With centering on, raw / rff_low / ssp_low differ only in the encoder, so the comparison is clean.
  """
  return [
    _spec(f"diagwc_go1_{rep}_s{s}", "diagwc", "go1", s, "value", 0.99, iters, eta, rep=rep, ssp_ls=ssp_ls,
          r_bar_init="first_batch")
    for s in seeds for rep in ("raw", "rff_low", "ssp_low")
  ]


def g1_diag(iters: int, eta: float = 0.03, seed: int = 1000) -> list[RunSpec]:
  """Decision 73: the G1 reference probe was run at offset 1.0 WITHOUT centering, which is the cell that also fails on
  Go1, so it did not test the continuing conversion. Two runs, same seed, same length, gamma .99 (one change against the
  failed probe):

  A `g1diag_raw_none_g0.99_o0_s...`  offset 0, no centering. Walks -> the offset explains the G1 failure.
  B `g1diag_raw_value_g0.99_o1_s...` offset 1, value centering, eta 0.03 + first-batch r_bar (Go1's frozen setting).
    Walks -> centering rescues G1 as it rescues Go1, i.e. the main effect holds on a humanoid.
  """
  return [
    _spec(f"g1diag_raw_none_g0.99_o0_s{seed}", "g1diag", "g1", seed, "none", 0.99, iters, 0.0, offset=0.0),
    _spec(f"g1diag_raw_value_g0.99_o1_s{seed}", "g1diag", "g1", seed, "value", 0.99, iters, eta,
          r_bar_init="first_batch"),
  ]


def g1_episodic(iters: int, seed: int = 1000) -> list[RunSpec]:
  """Decision 73 follow-up, run only because the offset-0 arm did not walk: the offset is not what breaks G1, so the
  remaining suspect is the conversion itself. This is mjlab's own env - time-outs and the fell-over termination intact,
  no fall penalty, no offset - through our harness and the same pinned versions."""
  return [_spec(f"g1diag_episodic_none_g0.99_o0_s{seed}", "g1diag", "g1", seed, "none", 0.99, iters, 0.0,
                offset=0.0, episodic=True)]


def g1_timeouts_no_falls(iters: int, seed: int = 1000) -> list[RunSpec]:
  """Decision 73/74, third rung: time-outs kept (so the episode still resets and the value bootstraps on them) but a
  fall is not terminal. Separates "the fell-over termination teaches G1 the gait" from "losing the episode resets is
  what freezes exploration". Only launched if the fully episodic control walks."""
  return [_spec(f"g1diag_timeouts_nofalls_g0.99_s{seed}", "g1diag", "g1", seed, "none", 0.99, iters, 0.0,
                offset=0.0, episodic=True, falls_reset=False)]


def pilot_timeouts(iters: int, seeds_go1=(1000, 1001), seeds_g1=(1000, 1001, 1002),
                   gamma: float = 0.999) -> list[RunSpec]:
  """Round 1 pilots under the decision-77 wrapper (time-outs kept, a fall is not terminal, no fall penalty).

  Go1: does the centering main effect survive the protocol change, and at what length? G1: does centering matter on a
  humanoid once the wrapper stops breaking it? Both at gamma .999, the study's main setting, offset 0 vs offset c.
  """
  jobs: list[RunSpec] = []

  def add(robot: str, seeds, tag: str = "") -> None:
    for s in seeds:
      for c in ("none", "value"):
        for off in (0.0, OFFSET):
          jobs.append(_spec(
            f"pilot{tag}_{robot}_raw_{c}_o{off:g}_s{s}", "pilot", robot, s, c, gamma, iters,
            0.03 if c == "value" else 0.0, offset=off, penalty=0.0,
            r_bar_init="first_batch" if c == "value" else "zero"))

  add("go1", seeds_go1)
  add("g1", seeds_g1)
  return jobs


def g1_wrapper_replication(iters: int, seeds=(1001, 1002), gamma: float = 0.99) -> list[RunSpec]:
  """Two more seeds of the two G1 wrapper arms of RUN_M04 (offset 0, no centering): no-time-outs vs time-outs-kept."""
  jobs: list[RunSpec] = []
  for s in seeds:
    jobs.append(_spec(f"g1rep_legacy_none_o0_s{s}", "g1rep", "g1", s, "none", gamma, iters, 0.0, offset=0.0,
                      time_outs=False, falls_reset=True))
    jobs.append(_spec(f"g1rep_timeouts_none_o0_s{s}", "g1rep", "g1", s, "none", gamma, iters, 0.0, offset=0.0,
                      penalty=0.0))
  return jobs


def g1_episodic_rep(iters: int, seeds=(1003, 1004), gamma: float = 0.99) -> list[RunSpec]:
  """Decision 79: the episodic path (mjlab's own vec wrapper, no ContinuingVecEnv) on fresh seeds, to separate
  "the walking seed was lucky" from "the new wrapper path learns worse than the episodic path"."""
  return [_spec(f"g1rep_episodic_none_o0_s{s}", "g1rep", "g1", s, "none", gamma, iters, 0.0, offset=0.0,
                episodic=True) for s in seeds]


def g1_long_probe(iters: int = 4000, seed: int = 1001, gamma: float = 0.99) -> list[RunSpec]:
  """Decision 79: "G1 is slow" vs "G1 is stuck under the new wrapper", on the same seed and at 2x the pilot length.
  Priority 10 so these two run after the short cells; they are the last thing this campaign needs."""
  return [
    _spec(f"g1long_timeouts_none_o0_s{seed}_i{iters}", "g1long", "g1", seed, "none", gamma, iters, 0.0,
          offset=0.0, penalty=0.0, priority=10),
    _spec(f"g1long_legacy_none_o0_s{seed}_i{iters}", "g1long", "g1", seed, "none", gamma, iters, 0.0,
          offset=0.0, time_outs=False, falls_reset=True, priority=10),
  ]


def g1_gamma99_gap(iters: int, seeds_none=(1000, 1003, 1004), seeds_value=(1000, 1003),
                   gamma: float = 0.99) -> list[RunSpec]:
  """Decision 79: the only G1 regime where walking has ever been observed is gamma .99 on seed 1000, and the
  replication on seeds 1001/1002 did not walk in either wrapper arm. These cells put seed 1000 through the *new*
  wrapper path (a direct replication of RUN_M04's walking arm) and add two more seeds at offset 0, plus two centered
  seeds, so "G1 walks under time-outs" is measured as a rate rather than a single seed."""
  jobs: list[RunSpec] = []
  for s in seeds_none:
    jobs.append(_spec(f"pilot99_g1_raw_none_o0_s{s}", "pilot99", "g1", s, "none", gamma, iters, 0.0,
                      offset=0.0, penalty=0.0))
  for s in seeds_value:
    jobs.append(_spec(f"pilot99_g1_raw_value_o0_s{s}", "pilot99", "g1", s, "value", gamma, iters, 0.03,
                      offset=0.0, penalty=0.0, r_bar_init="first_batch"))
  return jobs


def corrected_pilots(iters: int, g1_seeds=(1000, 1001, 1002), go1_seeds=(1000, 1001),
                     gamma_g1: float = 0.99, gamma_go1: float = 0.999) -> list[RunSpec]:
  """Decision 81: the pilot re-run under the corrected default (time-outs kept AND a fall still resets the robot).

  G1 first: offset 0, no centering, gamma .99, penalty 0 - the exact RUN_M04 walking cell, now with the fall reset
  restored, so "does G1 walk at all in this stack" is answered before anything about centering. Go1 second: the
  centering main effect, penalty 10 as in the legacy study cells, gamma .999.
  """
  jobs: list[RunSpec] = []
  for s in g1_seeds:
    jobs.append(_spec(f"fix_g1_raw_none_o0_s{s}", "pilotfix", "g1", s, "none", gamma_g1, iters, 0.0,
                      offset=0.0, penalty=0.0, priority=0))
  for s in go1_seeds:
    for c in ("none", "value"):
      for off in (0.0, OFFSET):
        jobs.append(_spec(
          f"fix_go1_raw_{c}_o{off:g}_s{s}", "pilotfix", "go1", s, c, gamma_go1, iters,
          0.03 if c == "value" else 0.0, offset=off, penalty=PENALTY, priority=1,
          r_bar_init="first_batch" if c == "value" else "zero"))
  return jobs


def g1_penalty_check(iters: int, seeds=(1000,), gamma: float = 0.99) -> list[RunSpec]:
  """Decision 82: one G1 seed at penalty 10 with the corrected wrapper, run only if the penalty-0 G1 cell walks.

  A free reset in a continuing task invites exploits (Farrahi & Mahmood's continuing Hopper dove forward and fell on
  purpose when a fall cost nothing), so the protocol needs one penalty rule for both robots. Go1's frozen P = 10 is
  ~130x its standing per-step task reward (0.077); the same multiple on G1 (standing 0.0436) is P ~= 5.7. If P = 10
  breaks G1 but P = 5.7 does not, the rule is "penalty = 130 x the standing per-step reward", logged per robot.
  """
  return [_spec(f"fix_g1_pen10_none_o0_s{s}", "pilotfix", "g1", s, "none", gamma, iters, 0.0, offset=0.0,
                penalty=10.0, priority=2) for s in seeds]


def g1_centering_pilots(iters: int, seeds=(1000, 1001, 1002), penalty: float = 10.0) -> list[RunSpec]:
  """Decision 84: G1 centering pilots under the corrected wrapper, run only if the gate fires (>= 2 of 3 corrected
  G1 seeds walk and the penalty-10 arm does not break it).

  Two questions, four cells, gamma .99 with offset 1 (the main effect on a second robot) and gamma .999 with offset 0
  (the pure high-gamma effect). The G1 penalty is whatever the gate resolved: 10 if G1 tolerates it, else the
  pre-registered 130 x standing-reward rule (5.7).
  """
  jobs: list[RunSpec] = []
  for gam, off in ((0.99, OFFSET), (0.999, 0.0)):
    for s in seeds:
      for c in ("none", "value"):
        jobs.append(_spec(
          f"g1c_raw_{c}_g{gam}_o{off:g}_s{s}", "g1c", "g1", s, c, gam, iters,
          0.03 if c == "value" else 0.0, offset=off, penalty=penalty, priority=1,
          r_bar_init="first_batch" if c == "value" else "zero"))
  return jobs


CLAIM_SEEDS_2 = tuple(range(3000, 3020))  # fresh, never used in any earlier wave


def overnight(iters: int, g1_penalty: float = 10.0, go1_seeds=CLAIM_SEEDS_2, off0_seeds=(3000, 3001, 3002, 3003, 3004,
                                                                                 3005, 3006, 3007, 3008, 3009),
          g99_seeds=(3000, 3001, 3002)) -> list[RunSpec]:
  """The overnight run: the Go1 main effect confirmed at 20 fresh seeds under the corrected protocol, the G1
  centering pilots, the offset-0 block at gamma .999, and a small gamma .99 control.

  Priorities: 1 Go1 offset c (the paper's claim), 2 G1 centering (second robot), 3 Go1 offset 0 at gamma .999, 4 the
  gamma .99 control for the offset-0 question. Everything uses the decision-81 corrected wrapper.
  """
  jobs: list[RunSpec] = []

  def add(tag: str, robot: str, seeds, gamma: float, offset: float, penalty: float, prio: int) -> None:
    for s in seeds:
      for c in ("none", "value"):
        jobs.append(_spec(
          f"{tag}_{robot}_raw_{c}_g{gamma}_o{offset:g}_s{s}", tag, robot, s, c, gamma, iters,
          0.03 if c == "value" else 0.0, offset=offset, penalty=penalty, priority=prio,
          r_bar_init="first_batch" if c == "value" else "zero"))

  add("w2main", "go1", go1_seeds, 0.999, OFFSET, PENALTY, 1)          # the 20-seed confirmation, offset c
  add("w2g1c", "g1", (1000, 1001, 1002), 0.99, OFFSET, g1_penalty, 2)  # G1 centering, gamma .99, offset 1
  add("w2g1c999", "g1", (1000, 1001, 1002), 0.999, 0.0, g1_penalty, 2)  # G1 centering, gamma .999, offset 0
  add("w2off0", "go1", off0_seeds, 0.999, 0.0, PENALTY, 3)             # offset-0 block at the main gamma
  add("w2off0g99", "go1", g99_seeds, 0.99, 0.0, PENALTY, 4)            # does gamma .99 rescue offset 0?

  # Shock block (decision 88): the practitioner question. Same seed, same policy path, shocked at iteration 1200; the
  # unshocked mother is the `w2main` value cell of the same seed, so its 1200-2000 segment is the control.
  for s in (3000, 3001, 3002, 3003, 3004):
    for kind, factor, tag in (("offset", 1.1, "off+0.1"), ("action_rate", 5.0, "actx5")):
      jobs.append(_spec(f"w2shock_go1_value_{tag}_s{s}", "w2shock", "go1", s, "value", 0.999, iters, 0.03,
                        offset=OFFSET, penalty=PENALTY, priority=5, r_bar_init="first_batch",
                        shock_kind=kind, shock_at=1200, shock_factor=factor))
  return jobs


def clipoff_pilot(iters: int = 2000, seeds=(3000, 3001, 3002), gamma: float = 0.999) -> list[RunSpec]:
  """Decision 100: value clipping OFF, uncentered, under the corrected protocol - the rival that recovered reward but
  not walking in the legacy study. Same settings as the 20-seed main effect (Go1, gamma .999, offset +1, penalty 10),
  same seeds 3000-3002, so each run pairs with an existing uncentered and an existing centered run at no extra cost.
  Pilot only: 3 seeds decide whether a 20-seed rival run is worth paying for (rule fixed in decision 100)."""
  return [_spec(f"clip1_go1_raw_none_clipoff_g{gamma}_o1_s{s}", "clip1", "go1", s, "none", gamma, iters, 0.0,
                offset=OFFSET, penalty=PENALTY, value_clip=False, priority=1) for s in seeds]


def to_manifest(jobs: list[RunSpec]) -> dict:
  from dataclasses import asdict

  return {"jobs": [asdict(j) for j in jobs]}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--wave", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
  ap.add_argument("--g1-penalty", type=float, default=10.0, help="resolved G1 penalty for wave 10")
  ap.add_argument("--iters", type=int, required=True)
  ap.add_argument("--shock-at", type=int, default=None)
  ap.add_argument("--eta", type=float, default=0.03)
  ap.add_argument("--eta-g1", type=float, default=0.03)
  ap.add_argument("--g1-seed", type=int, default=1000, help="wave 6 (G1 diagnosis) seed")
  ap.add_argument("--ssp-ls", type=float, default=0.3)
  ap.add_argument("--no-ssp", action="store_true", help="wave 2 without the SSP-input cells")
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  if a.wave == 1:
    jobs = wave1(a.iters)
  elif a.wave == 2:
    jobs = wave2(a.iters, a.shock_at, a.eta, a.ssp_ls, include_ssp=not a.no_ssp)
  elif a.wave == 3:
    jobs = wave3(a.iters, a.eta_g1, a.ssp_ls)
  elif a.wave == 4:
    jobs = wave4(a.iters, a.eta, a.eta_g1, a.ssp_ls)
  elif a.wave == 5:
    jobs = diag_width(a.iters, a.eta, a.ssp_ls)
  elif a.wave == 6:
    jobs = g1_diag(a.iters, a.eta, a.g1_seed)
  elif a.wave == 7:  # the G1 diagnosis followed by the episodic conversion control
    jobs = g1_diag(a.iters, a.eta, a.g1_seed) + g1_episodic(a.iters, a.g1_seed)
  elif a.wave == 8:  # round-1 pilots under the decision-77 wrapper, plus the G1 wrapper replication
    jobs = pilot_timeouts(a.iters) + g1_wrapper_replication(a.iters)
  elif a.wave == 9:  # the corrected-protocol pilot (decision 81)
    jobs = corrected_pilots(a.iters)
  elif a.wave == 10:  # the overnight run (decisions 85-87)
    jobs = overnight(a.iters, g1_penalty=a.g1_penalty)
  else:  # wave 11: the clip-off rival pilot under the corrected protocol (decision 100)
    jobs = clipoff_pilot(a.iters)
  Path(a.out).write_text(json.dumps(to_manifest(jobs), indent=1))
  print(f"{len(jobs)} jobs -> {a.out}")


if __name__ == "__main__":
  main()
