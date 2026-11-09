"""Wiring check for the round-9a arms: does each run's own log show the knob it was supposed to run with?

    python analysis/r9_wiring.py [--campaign r9pilot-3ae3cc84]

Run this as soon as the first run of each arm completes (and again at the end). It reads `config.json` plus the first
60 rows of `iter.jsonl` and asserts the arm's signature, so a mis-wired arm is caught on one run rather than after ten:

  a1_*     config bias_lr in {0.05, 0.5}, value_clip false, `bias_lr` logged with that value, `critic_bias` present and
           moving away from 0 (a fast-bias arm whose bias never moves is the failure this catches)
  clipoff  value_clip false, no `critic_bias`
  a2_*     warmup true, `warmup_active` = 1 in the early rows, `actor_lr_eff` = 0 while frozen, `warmup_done_at` set in
           one of the two directions (always -1 in an unfinished run is fine, but the flag must be 0/1, not missing)
  a3       actor_lr_scale 0.1, actor_lr_iters 500, `actor_lr_eff` ≈ 0.1 x `lr` in the first rows and equal to `lr` after
           iteration 500 (the second half is only checkable once the run reaches it)
  round 9b (r9leak_/r9diag_): `adv_mode` and `lam` in config.json match the job id; `adv_pos_mean` is logged with one
           entry per rollout position; `pos_norm`/`pos_center` runs also log `adv_mode` in their rows

Every check prints PASS / FAIL / PENDING (not enough rows yet). Exit status is non-zero if any FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mjlab_centering import r2  # noqa: E402

MERGED = ROOT / "results" / "merged_r9"


def rows_of(campaign: str, jid: str) -> tuple[dict, list[dict]]:
  d = MERGED / jid
  if not (d / "iter.jsonl").exists():
    d.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "iter.jsonl"):
      with tempfile.TemporaryDirectory() as td:
        p = Path(td) / name
        if r2.download_file(f"{campaign}/runs/{jid}/{name}", str(p)):
          (d / name).write_bytes(p.read_bytes())
  cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
  it = [json.loads(l) for l in (d / "iter.jsonl").read_text().splitlines() if l.strip()] if (d / "iter.jsonl").exists() else []
  return cfg, it


LEAK_EXPECT = {"posnorm": ("pos_norm", 0.95), "poscenter": ("pos_center", 0.95),
               "lam0_none": ("global", 0.0), "lam0_value": ("global", 0.0), "len96": ("pos_norm", 0.95),
               "diag_none": ("global", 0.95), "diag_value": ("global", 0.95)}


def check_leak(cfg: dict, it: list[dict], arm: str) -> list[tuple[str, str, str]]:
  out: list[tuple[str, str, str]] = []

  def ck(name: str, ok: bool | None, detail: str = "") -> None:
    out.append((arm, "PENDING" if ok is None else ("PASS" if ok else "FAIL"), f"{name} {detail}".strip()))

  if not cfg:
    return [(arm, "PENDING", "no config.json yet")]
  mode, lam = LEAK_EXPECT[arm]
  ck("adv_mode", cfg.get("adv_mode") == mode, f"{cfg.get('adv_mode')} (want {mode})")
  ck("lam", cfg.get("lam") == lam, f"{cfg.get('lam')} (want {lam})")
  ck("log_adv_pos", cfg.get("log_adv_pos") is True, str(cfg.get("log_adv_pos")))
  if arm == "lam0_value":
    ck("centred template", cfg.get("center") == "value" and cfg.get("eta") == 0.03, f"{cfg.get('center')}/{cfg.get('eta')}")
  if arm == "len96":
    ck("rollout 96 x 512", cfg.get("num_steps_per_env") == 96 and cfg.get("num_envs") == 512,
       f"{cfg.get('num_steps_per_env')}x{cfg.get('num_envs')}")
  if not it:
    return out + [(arm, "PENDING", "no iter rows yet")]
  n_pos = int(cfg.get("num_steps_per_env") or 24)  # the profile has one entry per ROLLOUT STEP, not always 24
  pm = [r.get("adv_pos_mean") for r in it[:5] if r.get("adv_pos_mean")]
  ck(f"adv_pos_mean {n_pos}-long", bool(pm) and all(len(v) == n_pos for v in pm),
     f"n={len(pm)} lengths={sorted({len(v) for v in pm})}")
  r0 = it[0]
  if mode != "global":
    ck("adv_mode in rows", r0.get("adv_mode") == mode, str(r0.get("adv_mode")))
  ck("lam in rows", abs(float(r0.get("lam", -1)) - lam) < 1e-12, str(r0.get("lam")))
  return out


def check(cfg: dict, it: list[dict], arm: str) -> list[tuple[str, str, str]]:
  if arm in LEAK_EXPECT:
    return check_leak(cfg, it, arm)
  out: list[tuple[str, str, str]] = []

  def ck(name: str, ok: bool | None, detail: str = "") -> None:
    out.append((arm, "PENDING" if ok is None else ("PASS" if ok else "FAIL"), f"{name} {detail}".strip()))

  if not cfg:
    return [(arm, "PENDING", "no config.json yet")]
  ck("bias_lr", (cfg.get("bias_lr") == 0.5 or cfg.get("bias_lr") == 0.05) if arm.startswith("a1") else None,
     str(cfg.get("bias_lr")))
  ck("value_clip", (cfg.get("value_clip") in (False, None) if arm.startswith("a1") or arm == "clipoff" else None),
     str(cfg.get("value_clip")))
  if not it:
    return out + [(arm, "PENDING", "no iter rows yet")]
  bias = [r.get("critic_bias") for r in it[:60] if r.get("critic_bias") is not None]
  ck("critic_bias logged", bool(bias) if arm.startswith("a1") else (not bias if arm != "a2_on" and arm != "a2_off" else None),
     f"n={len(bias)}")
  if arm.startswith("a1") and bias:
    ck("bias moves", abs(bias[-1] - bias[0]) > 1.0, f"{bias[0]:.2f} -> {bias[-1]:.2f}")
    ck("bias lr logged", all(abs((r.get("bias_lr") or 0) - cfg["bias_lr"]) < 1e-9 for r in it[:20] if "bias_lr" in r))
  if arm.startswith("a2"):
    ck("warmup flag present", all("warmup_active" in r for r in it[:5]))
    frozen = [r for r in it[:20] if r.get("warmup_active") == 1.0]
    ck("actor lr 0 while frozen", all(abs(r.get("actor_lr_eff", 1)) < 1e-12 for r in frozen), f"n_frozen={len(frozen)}")
  if arm == "a3":
    early = [r for r in it[:20] if r["iter"] < 500]
    ck("actor lr = 0.1 x lr", all(abs(r["actor_lr_eff"] - 0.1 * r["lr"]) < 1e-9 for r in early), f"n={len(early)}")
    late = [r for r in it if r["iter"] >= 505]
    if late:
      ck("actor lr restored", abs(late[0]["actor_lr_eff"] - late[0]["lr"]) < 1e-9)
    else:
      ck("actor lr restored", None, "(run has not reached iteration 505)")
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", default="r9pilot-3ae3cc84")
  a = ap.parse_args()
  man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
  done = r2.done_set(a.campaign)
  seen = set()
  bad = 0
  for j in man["jobs"]:
    jid = j["job_id"]
    if "_go1_diag_" in jid:  # round 9b diagnostic ids: r9diag_go1_diag_{arm}_g0.999_o1_s3000
      arm = "diag_" + jid.split("_go1_diag_")[1].split("_g0.999")[0]
    else:
      arm = jid.split("_go1_")[1].rsplit("_g0.999", 1)[0]
    if arm in seen:
      continue
    done_here = jid in done or any(r2.exists(f"{a.campaign}/runs/{jid}/iter.jsonl") for _ in [0])
    if not done_here:
      continue
    seen.add(arm)
    cfg, it = rows_of(a.campaign, jid)
    print(f"== {arm} ({jid})")
    for _, status, msg in check(cfg, it, arm):
      if status == "FAIL":
        bad += 1
      print(f"  {status:7s} {msg}")
  if not seen:
    print("no run has started yet")
  sys.exit(1 if bad else 0)


if __name__ == "__main__":
  main()
