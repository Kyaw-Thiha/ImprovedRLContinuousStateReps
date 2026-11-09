"""Claim semantics against the real R2 bucket, under a self-test prefix that is deleted afterwards.

Covers what the campaign depends on:
  - exactly one of N concurrent claimers wins a job (conditional put),
  - a live claim blocks a second claimer,
  - a claim older than the TTL can be taken over, and only one taker wins,
  - every key written lives under mjlab_ssp_centering/.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
import uuid

import pytest

from mjlab_centering import r2


@pytest.fixture()
def campaign():
  c = f"_selftest/{uuid.uuid4().hex[:8]}"
  yield c
  for k in r2.list_keys(c):
    r2.client().delete_object(Bucket=r2.bucket(), Key=k)


def test_exactly_one_claimer_wins(campaign):
  wins = []
  with cf.ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(r2.claim, campaign, "jobA", f"pod{i}") for i in range(8)]
    wins = [f.result() for f in futs]
  assert sum(wins) == 1, wins


def test_live_claim_blocks_and_stale_claim_is_taken_over(campaign):
  assert r2.claim(campaign, "jobB", "pod1", ttl_s=600)
  assert not r2.claim(campaign, "jobB", "pod2", ttl_s=600)

  # age the claim past the TTL, then race two takers
  k = r2.key(f"{campaign}/claims/jobB.json")
  r2.client().put_object(Bucket=r2.bucket(), Key=k, Body=b'{"pod": "pod1", "ts": 0}')

  with cf.ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(r2.claim, campaign, "jobB", p, ttl_s=1) for p in ("pod2", "pod3")]
    wins = [f.result() for f in futs]
  assert sum(wins) == 1, wins


def test_done_markers_and_prefix_discipline(campaign):
  r2.put_json(f"{campaign}/runs/jobC/DONE", {"job_id": "jobC"})
  assert r2.done_set(campaign) == {"jobC"}
  keys = r2.list_keys(campaign)
  assert all(k.startswith(f"{r2.PREFIX}/") for k in keys), keys
  assert not any(k.startswith(f"{r2.PREFIX}/stage0") for k in keys)


def test_manifest_roundtrip(campaign):
  jobs = [{"job_id": "j1", "seed": 2000, "block": "A", "priority": 1}]
  r2.put_json(f"{campaign}/manifest.json", {"campaign": campaign, "jobs": jobs})
  back = r2.get_json(f"{campaign}/manifest.json")
  assert back["jobs"] == jobs
  assert r2.get_json(f"{campaign}/nope.json") is None


def test_heartbeat_refreshes_the_claim(campaign):
  assert r2.claim(campaign, "jobD", "pod1", ttl_s=600)
  time.sleep(0.05)
  r2.heartbeat(campaign, "jobD", "pod1")
  cur = r2.get_json(f"{campaign}/claims/jobD.json")
  assert cur["pod"] == "pod1" and cur["ts"] > 1e9


def test_release_claims_of_drops_claims_held_by_its_owner_field(campaign):
  """podrun writes {"ts", "owner"}; release must honour it (2026-11-06: a dead pod's claims were never released)."""
  r2.put_json(f"{campaign}/claims/a_s1000.json", {"ts": time.time(), "owner": "podA"})
  r2.put_json(f"{campaign}/claims/b_s1001.json", {"ts": time.time(), "pod": "podA"})
  r2.put_json(f"{campaign}/claims/c_s1002.json", {"ts": time.time(), "owner": "podB"})
  n = r2.release_claims_of(campaign, "podA")
  assert n == 2
  left = {k.rsplit("/", 1)[-1] for k in r2.list_keys(f"{campaign}/claims")}
  assert left == {"c_s1002.json"}
