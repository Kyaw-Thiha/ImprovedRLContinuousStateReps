"""Targeted, READ-ONLY fetch of the G1 claim runs for the supervisor figures (used instead of restore_runs.py, which
pulls every file of every run). Reads only: the campaign manifest, the DONE markers (one list call), and for each
finished run `eval.jsonl`, `final.json`, `config.json`. Writes only into results/supervisor_cache/g1. Files already in
the cache are skipped, so rerunning when the claim set finishes only downloads what is new; a cached `eval.jsonl` that
holds fewer rows than the run's `final.json` recorded (a mid-upload copy) is fetched again.

    python analysis/fetch_supervisor_data.py            # from mjlab_centering/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from mjlab_centering import r2  # noqa: E402

CAMPAIGN = "r6g1queue-0ede5636"
DEST = HERE.parent / "results" / "supervisor_cache" / "g1"
FILES = ("eval.jsonl", "final.json", "config.json")


def _rows(p: Path) -> int:
  return sum(1 for l in p.read_text().splitlines() if l.strip()) if p.exists() else 0


def main() -> None:
  DEST.mkdir(parents=True, exist_ok=True)
  man = r2.get_json(f"{CAMPAIGN}/manifest.json")  # read
  (DEST / "manifest.json").write_text(json.dumps(man))
  done = sorted(r2.done_set(CAMPAIGN))  # read (list)
  new = skipped = missing = refetched = 0
  for jid in done:
    d = DEST / jid
    d.mkdir(exist_ok=True)
    short = False  # a cached eval.jsonl can be a mid-upload copy; final.json records how many evals the run wrote
    if (d / "eval.jsonl").exists() and (d / "final.json").exists():
      short = _rows(d / "eval.jsonl") < (json.loads((d / "final.json").read_text()).get("n_evals") or 0)
    for f in FILES:
      if (d / f).exists() and not (short and f == "eval.jsonl"):
        skipped += 1
        continue
      try:
        r2.client().download_file(r2.bucket(), r2.key(f"{CAMPAIGN}/runs/{jid}/{f}"), str(d / f))
        new += 1
      except Exception:  # noqa: BLE001 - config.json may not exist for every run
        missing += 1
    refetched += int(short)
    if (d / "eval.jsonl").exists():
      (d / "DONE").write_text("{}")
  log = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "campaign": CAMPAIGN, "n_done": len(done), "downloaded": new,
         "skipped_cached": skipped, "missing": missing, "refetched_short_eval": refetched,
         "r2_prefixes_read": [r2.key(f"{CAMPAIGN}/manifest.json"), r2.key(f"{CAMPAIGN}/runs/<job>/DONE (list)"),
                              *[r2.key(f"{CAMPAIGN}/runs/<DONE job>/{f}") for f in FILES]]}
  (DEST / "fetch_log.json").write_text(json.dumps(log, indent=1))
  print(json.dumps({k: v for k, v in log.items() if k != "r2_prefixes_read"}))


if __name__ == "__main__":
  main()
