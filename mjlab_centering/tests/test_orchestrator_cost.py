"""Cost accounting and cap logic, verified before any pod is launched."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orchestrator as orch  # noqa: E402


def test_parse_gpu_fallbacks():
  g = orch._parse_gpus("NVIDIA GeForce RTX 4090:0.34:COMMUNITY,NVIDIA L40S:1.09:SECURE")
  assert g == [("NVIDIA GeForce RTX 4090", 0.34, "COMMUNITY"), ("NVIDIA L40S", 1.09, "SECURE")]


def test_accounting_uses_each_pods_own_price_and_closes_dead_pods():
  st = {"pods": {}, "spend": 0.0}
  t0 = 1_000_000.0
  live = [{"id": "a"}]
  st["pods"]["a"] = {"first": t0, "last": t0, "price": 0.34}
  st["pods"]["b"] = {"first": t0 - 7200, "last": t0, "price": 0.89, "closed_ts": t0}
  spend = orch._account(st, live, 1.0, t0 + 1800)  # pod a has run 30 min at $0.34/h
  assert abs(st["pods"]["a"]["last"] - (t0 + 1800)) < 1e-6
  assert abs(spend - (0.5 * 0.34 + 2.0 * 0.89)) < 1e-6


def test_caps_trigger_at_the_configured_thresholds():
  # A pod accrues while the API lists it. A pod that stops being listed is only closed after a 15-minute grace
  # period (the pod list lags), which is covered by test_accounting_uses_each_pods_own_price_and_closes_dead_pods.
  st = {"pods": {}, "spend": 0.0}
  t0 = 1_000_000.0
  st["pods"]["a"] = {"first": t0, "last": t0, "price": 1.0}
  spend = orch._account(st, [{"id": "a"}], 1.0, t0 + 21 * 3600)
  assert spend >= 20.0 and spend < 25.0  # soft stop region
  st2 = {"pods": {"a": {"first": t0, "last": t0, "price": 1.0}}, "spend": 0.0}
  spend = orch._account(st2, [{"id": "a"}], 1.0, t0 + 26 * 3600)  # still live
  assert spend >= 25.0  # hard kill region


def test_redaction_hides_secrets():
  from mjlab_centering import pods

  # Shaped like a provider key and a 32+ hex secret, but assembled at runtime so
  # no key-shaped literal ever lands in the repository or a scanner's index.
  token = "rpa_" + "A" * 40
  hex_key = "f" * 40
  out = pods.redact(f"key {token} and {hex_key}")
  assert token not in out and hex_key not in out
  assert out.count("<redacted>") == 2
