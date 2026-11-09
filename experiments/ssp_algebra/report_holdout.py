"""Hold-out-only reporting for pod job lists (RUN_013 correction).

    OMP_NUM_THREADS=1 ../../.venv/bin/python report_holdout.py jobs/p14_equal_aid.json jobs/p19_spatial.json

Why this exists.  A sweep file mixes two kinds of rows: **search** rows (the config-selection runs, ``phase == "search"``,
seeds below 1000) and **final** rows (the hold-out seeds, ``phase == "final"``, seeds from 1000).  The first RUN_013
tables averaged whatever was in the file, so (a) every ``configs=1`` cell reported 11 seeds instead of the protocol's 10,
because the single search run was counted, and (b) every *tuned* cell (``configs>1``) pooled the search runs of
*different configs* with the final runs -- 28 rows averaged across configs that were never the reported one.  Claims
may rest on hold-out rows only (docs/11 4.3; ``06-related-work.md``), so this script reads ``phase == "final"`` and
nothing else, and states ``n`` next to every number so a wrong denominator cannot pass unnoticed.

Regression tasks (``variant == "spatial"``) store ``-MSE`` as the score; this prints MSE (lower is better).
"""

import argparse
import glob
import json
import pathlib
import statistics as st

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
REGRESSION = {"spatial", "path"}


def final_rows(tag):
    files = sorted(glob.glob(str(RESULTS / f"sweep_{tag}.jsonl")))
    if not files:
        return None
    rows = [json.loads(l) for l in open(files[0]) if l.strip()]
    rows = [r for r in rows if isinstance(r.get("summary"), dict) and "conv_id" in r["summary"]]
    if rows and "phase" in rows[0]:
        return [r for r in rows if r["phase"] == "final"]
    return [r for r in rows if r.get("seed", 0) >= 1000]          # older files without a phase key


def describe(job):
    rows = final_rows(job["tag"])
    if rows is None:
        return f"{job['name']:38s} NO FILE"
    if not rows:
        return f"{job['name']:38s} no hold-out rows"
    reg = job["variant"] in REGRESSION
    conv = [r["summary"]["conv_id"] for r in rows]
    if reg:
        conv = [-c for c in conv]
    n = len(conv)
    mean, sd = st.mean(conv), (st.stdev(conv) if n > 1 else 0.0)
    head = f"{job['name']:38s} n={n:2d} {'MSE' if reg else 'conv'}={mean:.3f}+-{sd:.3f}"
    if not reg:
        head += f"  >=0.95:{sum(c >= 0.95 for c in conv):2d}/{n}  >=0.90:{sum(c >= 0.90 for c in conv):2d}/{n}"
        t90 = [r["summary"]["t90_id"] for r in rows if r["summary"].get("t90_id")]
        t50 = [r["summary"]["t50_id"] for r in rows if r["summary"].get("t50_id")]
        head += f"  t50={round(st.mean(t50)) if t50 else '-'}  t90={round(st.mean(t90)) if t90 else '-'}"
        if t90:
            head += f" ({len(t90)}/{n} reached)"
    extras = []
    for key in sorted(rows[0]["summary"]):
        if key.startswith("conv_") and key not in ("conv_id", "conv_train"):
            v = [r["summary"][key] for r in rows if r["summary"].get(key) is not None]
            if v:
                v = [-x for x in v] if reg else v
                extras.append(f"{key[5:]}={st.mean(v):.3f}")
    return head + ("   " + " ".join(extras) if extras else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    a = ap.parse_args()
    for jf in a.jobs:
        path = pathlib.Path(jf)
        if not path.exists():
            path = HERE / jf
        print(f"\n## {path.name}  (hold-out rows only)")
        for job in json.load(open(path)):
            print("  " + describe(job))


if __name__ == "__main__":
    main()
