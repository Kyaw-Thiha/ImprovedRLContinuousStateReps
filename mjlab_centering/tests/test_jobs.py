"""Job-list invariants: the pre-registered matrix in docs/mjlab_centering/00-plan.md must be reproducible."""

from __future__ import annotations

import collections

from mjlab_centering.jobs import ITERS, SEARCH_SEEDS, SEEDS, build_jobs, campaign_id


def test_counts_match_the_plan():
  jobs = build_jobs()
  by = collections.Counter(j.block for j in jobs)
  # decision 35: tuning blocks moved to 3 search seeds (1000-1002), disjoint from the claim seeds
  assert by["eta"] == 9  # 3 eta values x 3 search seeds
  assert by["A"] == 40  # 2 centering x 2 gamma x 2 offset x 5 seeds
  assert by["fixclip"] == 5
  assert by["fixnorm"] == 5
  assert by["fixscale"] == 3
  assert by["E"] == 20  # 2 centering x 2 shock kinds x 5 seeds
  assert by["pen"] == 6  # 2 penalty multipliers x 3 search seeds
  assert by["lr"] == 12  # 2 non-default learning rates x 2 centering x 3 search seeds (the default lr is block A)
  assert by["ssp"] == 6
  assert len(jobs) == 106


def test_ids_are_unique_and_do_not_depend_on_the_eta_winner():
  a = [j.job_id for j in build_jobs(eta=0.01)]
  b = [j.job_id for j in build_jobs(eta=0.1)]
  assert len(set(a)) == len(a)
  assert a != b  # only the eta block's ids may move
  moved = {x for x, y in zip(a, b) if x != y}
  assert all(m.startswith("eta_") for m in moved)


def test_claim_arms_use_the_shared_claim_seeds_and_tuning_arms_the_search_seeds():
  jobs = build_jobs()
  tuning = {"eta", "pen", "lr"}
  assert {j.seed for j in jobs if j.block not in tuning} <= set(SEEDS)
  assert {j.seed for j in jobs if j.block in tuning} <= set(SEARCH_SEEDS)


def test_shock_runs_are_longer_than_the_common_length_and_land_mid_run():
  """The shock must land after the pre-shock baseline exists and leave room for recovery, so E runs 1.4x the common
  length with the shock at 0.8x of it (0.8/1.4 = 57% of the run)."""
  for j in build_jobs():
    if j.block == "E":
      assert j.iters == round(ITERS * 1.4)
      assert j.shock_at == round(ITERS * 0.8)
      assert j.shock_kind in ("offset", "action_rate")


def test_campaign_id_is_content_addressed():
  jobs = build_jobs()
  assert campaign_id(jobs, "c1") == campaign_id(list(jobs), "c1")
  assert campaign_id(jobs, "c1").startswith("c1-")


def test_reference_cell_exists_and_is_the_normaliser():
  """J_ref = raw, no centering, gamma=0.99, offset 0, run through the continuing wrapper."""
  ref = [j for j in build_jobs() if j.block == "A" and j.center == "none" and j.gamma == 0.99 and j.offset == 0.0]
  assert len(ref) == len(SEEDS)
