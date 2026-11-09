"""Pull finished pod jobs from R2 and print one table per run: the end-of-session reporting step.

    .venv/bin/python experiments/ssp_algebra/report_jobs.py jobs/p0_halt.json jobs/p1_multiscale.json ...

For each job: fetch ``sweep_<tag>.jsonl`` when it is complete (``rows`` satisfied), save it under ``results/``, and
print, over the *hold-out* seeds only, the mean/min converged in-distribution score, how many seeds clear 0.90 and
0.95, the median tokens to 0.90, and the mean held-out column (``extrap_lag`` for pointer/delay, ``extrap_T`` for
index).  Also reports the columns needed for the pass rules in handoff 10 section 2.
"""

import json
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import launch_cell_pod as lc          # noqa: E402
import launch_pilot_pod as lp          # noqa: E402
import ssp_pod_batch as B              # noqa: E402

OUT = HERE / "results"


def pull(jobs, s3, bucket):
    done = []
    for j in jobs:
        fn, body, n = B.r2_rows(s3, bucket, j)
        state = "saved" if n >= j["rows"] else f"{n}/{j['rows']}"
        if n >= j["rows"]:
            (OUT / fn).write_text(body)
            try:
                s3.download_file(bucket, f"ssp_algebra/{j['name']}/winners_{j['tag']}.json",
                                 str(OUT / f"winners_{j['tag']}.json"))
            except Exception:              # noqa: BLE001
                pass
            done.append((j, fn))
        print(f"  {j['name']:24s} {state}")
    return done


def table(jobs_rows):
    print(f"\n{'tag':26s} {'n':>3s} {'conv':>6s} {'min':>6s} {'>=0.9':>5s} {'>=0.95':>6s} {'t90_tok':>9s} "
          f"{'held-out':>9s} {'train_side':>10s}")
    for j, fn in jobs_rows:
        rows = [json.loads(line) for line in (OUT / fn).read_text().splitlines() if line.strip()]
        s = [r["summary"] for r in rows if r["seed"] >= 1000]      # hold-out seeds only (1000+)
        if not s:
            continue
        conv = [x["conv_id"] for x in s]
        t90 = [x.get("tokens_to_0.9_id") for x in s if x.get("tokens_to_0.9_id") is not None]
        held = [x.get("conv_extrap_lag", x.get("conv_extrap_T")) for x in s]
        held = [h for h in held if h is not None]
        train = [x.get("conv_train") for x in s if x.get("conv_train") is not None]
        print(f"{j['tag']:26s} {len(s):3d} {st.mean(conv):6.3f} {min(conv):6.3f} "
              f"{sum(1 for c in conv if c >= 0.9):5d} {sum(1 for c in conv if c >= 0.95):6d} "
              f"{st.median(t90) if t90 else float('nan'):9.0f} {st.mean(held) if held else float('nan'):9.3f} "
              f"{st.mean(train) if train else float('nan'):10.3f}")


if __name__ == "__main__":
    env = lp.load_env()
    s3 = lc.s3_client(env)
    jobs = []
    for path in sys.argv[1:]:
        jobs += B.load_jobs(path)
    done = pull(jobs, s3, env["R2_BUCKET"])
    table(done)
