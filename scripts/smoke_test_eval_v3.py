"""
Local end-to-end smoke test of the eval_v3 pipeline at a TINY budget (no R2, no network):

  for discrete/none/linear, ssp/none/deep, tile_coding/none/linear:
    policy pass (2 seeds, --tiny) -> 2 verified policy files + train_runs/summary/manifest, re-run skips all
    -> run_robustness.py (3 conditions, 2 episodes), re-run skips all
    -> run_deployment.py (tiny)
  then analyze_robustness.py on the produced JSONL files, and a variant-mode pass that must save NO policy.

Every script is invoked as a subprocess, the way a pod would. Everything lives in a temp dir.

    python scripts/smoke_test_eval_v3.py            # all three cells
    python scripts/smoke_test_eval_v3.py --cells discrete/none/linear --keep
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import eval_v3_layout as layout  # noqa: E402

WINNERS = {
    "discrete/none/linear": {"n_bins": 4, "eps": 0.2, "lr": 0.05},
    "ssp/none/deep": {"n_rotates": 2, "length_scale": 0.2, "eps": 0.2, "lr": 0.003},
    "tile_coding/none/linear": {"num_tilings": 4, "iht_size": 512, "tiles_per_dim": 6, "eps": 0.2, "lr": 0.01},
}
ROB_CONDITIONS = "baseline,obs_noise_0.05,action_delay_1"
N_SEEDS = 2


def check(cond, msg):
    if not cond:
        print(f"ASSERTION FAILED: {msg}", flush=True)
        sys.exit(1)


def run_script(script, args, label):
    env = dict(os.environ)
    for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        env[k] = ""      # blank (not absent): parallel_jobs then neither reads .env nor talks to R2
    env["TQDM_DISABLE"] = "1"
    cmd = [sys.executable, os.path.join("scripts", script)] + [str(a) for a in args]
    print(f"--- {label}: {' '.join(cmd[1:])}", flush=True)
    p = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if p.returncode != 0:
        print(f"{label} FAILED (exit {p.returncode}); tail of output:\n" + "\n".join(out.splitlines()[-40:]),
              flush=True)
        sys.exit(1)
    return out


def n_lines(path):
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_cell(cell, root, tmp):
    study = layout.study_name(cell)
    ypath = os.path.join(tmp, study + ".yaml")
    with open(ypath, "w") as f:
        import yaml
        yaml.safe_dump({"params": WINNERS[cell]}, f)
    common = ["--cell", cell, "--local-root", root, "--no-r2", "--workers", 2, "--n-seeds", N_SEEDS]

    # 1. policy pass
    pp = common + ["--tiny", "--params-yaml", ypath, "--winner-trial", 7]
    run_script("run_policy_pass.py", pp, f"{cell} policy pass")
    pdir = os.path.dirname(layout.local_path(layout.policy_key(study, layout.POLICY_SEED_OFFSET), root))
    files = sorted(glob.glob(os.path.join(pdir, "seed_*.pt")))
    check(len(files) == N_SEEDS, f"{cell}: expected {N_SEEDS} policy files, got {files}")
    runs_path = layout.local_path(layout.train_runs_key(study), root)
    recs = read_jsonl(runs_path)
    check(len(recs) == N_SEEDS and all("error" not in r for r in recs), f"{cell}: train_runs {recs}")
    for r in recs:
        check(r["policy_verified"] is True and r["verify_max_abs_diff"] == 0.0, f"{cell}: verify {r}")
        check(r["kind"] == "policy" and r["trial"] == 7 and r["seed"] == layout.POLICY_SEED_OFFSET + r["seed_idx"],
              f"{cell}: record identity {r}")
        check(len(r["eval_history"]) == 6 and r.get("probe_history"), f"{cell}: eval/probe history {r}")
        check(r.get("final_eval_returns") and len(r["final_eval_returns"]) == 4, f"{cell}: final_eval_returns")
    for key in (layout.train_summary_key(study), layout.manifest_key(study)):
        check(os.path.exists(layout.local_path(key, root)), f"{cell}: missing {key}")
    with open(layout.local_path(layout.manifest_key(study), root)) as f:
        man = json.load(f)
    check(len(man["policy_files"]) == N_SEEDS and man["versions"]["torch"], f"{cell}: manifest {man.keys()}")
    with open(layout.local_path(layout.train_summary_key(study), root)) as f:
        summ = json.load(f)
    check(summ["n_seeds"] == N_SEEDS and len(summ["per_seed_scores"]) == N_SEEDS, f"{cell}: summary")
    out = run_script("run_policy_pass.py", pp, f"{cell} policy pass re-run")
    check("0 to run" in out, f"{cell}: re-run did not skip everything:\n{out[-600:]}")
    check(n_lines(runs_path) == N_SEEDS, f"{cell}: re-run appended records")

    # 2. robustness
    rob = common + ["--conditions", ROB_CONDITIONS, "--episodes", 2]
    run_script("run_robustness.py", rob, f"{cell} robustness")
    rpath = layout.local_path(layout.robustness_key(study), root)
    rrecs = read_jsonl(rpath)
    check(len(rrecs) == N_SEEDS * 3 and all("error" not in r for r in rrecs), f"{cell}: robustness records")
    check(all(len(r["returns"]) == 2 for r in rrecs), f"{cell}: robustness episodes")
    out = run_script("run_robustness.py", rob, f"{cell} robustness re-run")
    check("0 to run" in out and n_lines(rpath) == N_SEEDS * 3, f"{cell}: robustness re-run not skipped")

    # 3. deployment
    run_script("run_deployment.py", common + ["--quant-episodes", 2, "--latency-calls", 50],
               f"{cell} deployment")
    dpath = layout.local_path(layout.deployment_key(study), root)
    check(os.path.exists(dpath), f"{cell}: deployment.json missing")
    with open(dpath) as f:
        dep = json.load(f)
    missing = [k for k in ("size", "quantization", "latency") if not dep.get(k)]
    check(not missing, f"{cell}: deployment.json sections missing/empty: {missing} (has {list(dep)})")


def test_variant(cell, root, tmp):
    study = layout.study_name(cell)
    ypath = os.path.join(tmp, study + ".yaml")
    before = sorted(glob.glob(os.path.join(
        os.path.dirname(layout.local_path(layout.policy_key(study, layout.POLICY_SEED_OFFSET), root)), "*.pt")))
    args = ["--cell", cell, "--local-root", root, "--no-r2", "--workers", 2, "--n-seeds", N_SEEDS, "--tiny",
            "--params-yaml", ypath, "--variant-buffer-size", 200]
    run_script("run_policy_pass.py", args, f"{cell} variant pass")
    variant = layout.variant_name(200, None)
    vpath = layout.local_path(layout.train_runs_key(study, variant), root)
    recs = read_jsonl(vpath)
    check(len(recs) == N_SEEDS and all("error" not in r for r in recs), f"variant records {recs}")
    check(all("policy_key" not in r and r.get("eval_history") and r.get("probe_history") for r in recs),
          "variant records must carry histories and no policy")
    vdir = os.path.dirname(vpath)
    check(not glob.glob(os.path.join(vdir, "**", "*.pt"), recursive=True), "variant pass saved a policy")
    after = sorted(glob.glob(os.path.join(
        os.path.dirname(layout.local_path(layout.policy_key(study, layout.POLICY_SEED_OFFSET), root)), "*.pt")))
    check(before == after, "variant pass touched the baseline policies")
    with open(layout.local_path(layout.manifest_key(study, variant), root)) as f:
        man = json.load(f)
    check(man["variant"] == variant and man["trial_params"]["buffer_size"] == 200, "variant manifest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(WINNERS))
    ap.add_argument("--keep", action="store_true", help="keep the temp dir")
    a = ap.parse_args()
    cells = [c for c in a.cells.split(",") if c]
    tmp = tempfile.mkdtemp(prefix="eval_v3_smoke_")
    root = os.path.join(tmp, "eval_v3")
    try:
        for cell in cells:
            print(f"\n===== {cell} =====", flush=True)
            test_cell(cell, root, tmp)
        print("\n===== analyze_robustness =====", flush=True)
        out_dir = os.path.join(tmp, "analysis")
        run_script("analyze_robustness.py", ["--local-root", root, "--cells", ",".join(cells),
                                             "--out-dir", out_dir, "--n-boot", 100], "analysis")
        check(os.path.isdir(out_dir) and os.listdir(out_dir), "analysis wrote nothing")
        print("\n===== variant pass =====", flush=True)
        test_variant(cells[0], root, tmp)
        print("\nSMOKE TEST PASSED", flush=True)
    finally:
        if a.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
