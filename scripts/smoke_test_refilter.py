"""
End-to-end smoke test of the chained cell pipeline at a TINY budget (minutes, local, no R2):
search -> stale-RUNNING handling -> parallel resumable filter -> chained final eval -> resume.

    python scripts/smoke_test_refilter.py

Checks (asserts):
  1. fresh run: search target reached, filter pool/JSONL/winner written, final eval summary written;
  2. a stale RUNNING trial is marked FAIL and the target is raised by one (finished count unchanged);
  3. a re-run with everything recorded skips all jobs (nothing retrained);
  4. a re-run after the filter JSONL is truncated (incl. a half-written last line) keeps the surviving
     records untouched, drops the garbage line, re-runs ONLY the missing jobs, and rebuilds the
     final-eval summary. (Re-run jobs are not bit-reproducible: the env reset is unseeded, so the
     winner among near-tied candidates may legitimately change; that is not asserted.)
All files go to a temp dir; the repo's outputs/ and R2 are not touched.
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import optuna  # noqa: E402
from optuna.trial import TrialState  # noqa: E402

import run_hparam_search_dqn as m  # noqa: E402


def run_main(db, extra):
    sys.argv = ["run_hparam_search_dqn.py", "--representation", "discrete", "--centering", "none",
                "--db-path", db, "--no-sync", "--n-evaluations", "15", "--n-eval-episodes", "2",
                "--n-startup-trials", "50",          # no pruning: every trial keeps 15 checkpoints
                "--pool-min", "3", "--pool-max", "3", "--stage1-seeds", "2", "--reeval-seeds", "4",
                "--workers", "3", "--final-eval", "--final-seeds", "4", "--final-eval-episodes", "3",
                ] + extra
    m.main()


def n_lines(path):
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def main():
    tmp = tempfile.mkdtemp(prefix="refilter_smoke_")
    try:
        m.DATA_DIR = os.path.join(tmp, "hparam_search")
        m.BEST_DIR = os.path.join(m.DATA_DIR, "best")
        m.BASE_PARAMS.update(trials=30, steps=60)     # 30 episodes -> eval every 2nd episode -> 15 checkpoints
        db = os.path.join(tmp, "smoke.db")
        study_name = "dqn_discrete_none_v4_hparam_search"
        cell = os.path.join(m.DATA_DIR, "dqn", "discrete", "none")
        runs_path = os.path.join(cell, "reeval_v2", m.REEVAL_RUNS_FILE)
        final_runs = os.path.join(cell, "final_eval_v2", m.FINAL_RUNS_FILE)
        summary_path = os.path.join(cell, "final_eval_v2", m.FINAL_SUMMARY_FILE)

        print("\n===== 1. fresh run =====", flush=True)
        run_main(db, ["--n-trials", "8"])
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db}")
        finished = [t for t in study.trials if t.state.is_finished()]
        assert len(finished) == 8, len(finished)
        assert study.user_attrs["reeval_v2_done"] is True
        winner = study.user_attrs["reeval_v2_best_trial"]
        assert n_lines(runs_path) == 3 * 4, n_lines(runs_path)          # pool 3 x 4 seeds
        assert n_lines(final_runs) == 4
        with open(summary_path) as f:
            s1 = json.load(f)
        assert len(s1["final_eval"]["per_seed_scores"]) == 4
        assert s1["final_eval"]["seed_offset"] == 2000
        assert s1["final_eval"]["final_policy_eval_mean"] is not None
        print("  winner trial", winner, "final mean", s1["final_eval"]["score_mean"])

        print("\n===== 2. stale RUNNING trial =====", flush=True)
        zombie = study.ask()                       # leaves a RUNNING row, like a killed pod
        assert study.get_trials(deepcopy=False)[-1].state == TrialState.RUNNING
        before_runs = n_lines(runs_path)
        run_main(db, ["--n-trials", "8"])
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db}")
        states = [t.state for t in study.trials]
        assert TrialState.RUNNING not in states
        assert states.count(TrialState.FAIL) == 1
        assert study.user_attrs["target_n_trials"] == 9
        assert study.user_attrs["stale_running_failed"] == 1
        assert sum(1 for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)) == 8
        assert n_lines(runs_path) == before_runs, "filter must not retrain when everything is recorded"
        print("  zombie", zombie.number, "-> FAIL; finished still 8; target 9")

        print("\n===== 3. truncate JSONL, resume =====", flush=True)
        with open(runs_path) as f:
            lines = [ln for ln in f if ln.strip()]
        with open(runs_path, "w") as f:
            f.writelines(lines[:5])
            f.write('{"kind": "reeval", "tri')       # simulate a write cut off by a killed pod
        os.remove(summary_path)
        os.remove(final_runs)
        study.set_user_attr("reeval_v2_done", False)   # force the filter section to run again
        run_main(db, ["--n-trials", "8"])
        with open(runs_path) as f:
            after = [ln for ln in f]
        assert all(ln.endswith("\n") for ln in after), "garbage line must be repaired, not glued"
        assert after[:5] == lines[:5], "surviving records must be kept untouched"
        assert len(after) == 12, len(after)            # exactly the 12 distinct jobs, none duplicated
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db}")
        with open(summary_path) as f:
            s3 = json.load(f)
        assert len(s3["final_eval"]["per_seed_scores"]) == 4
        print("  resumed OK; kept 5 records, re-ran the other 7 jobs; winner now",
              study.user_attrs["reeval_v2_best_trial"])
        print("\nSMOKE TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
