"""Hold-out hygiene: tuning blocks must never use a claim seed."""

from __future__ import annotations

from mjlab_centering import jobs


def test_search_and_claim_seeds_are_disjoint():
  assert not set(jobs.SEARCH_SEEDS) & set(jobs.SEEDS)


def test_tuning_blocks_use_only_search_seeds_and_claim_blocks_only_claim_seeds():
  built = jobs.build_jobs()
  tuning = [j for j in built if j.block in ("eta", "pen", "lr")]
  claim = [j for j in built if j.block not in ("eta", "pen", "lr")]
  assert tuning and claim
  assert all(j.seed in jobs.SEARCH_SEEDS for j in tuning)
  assert all(j.seed in jobs.SEEDS for j in claim)
