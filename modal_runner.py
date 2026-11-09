"""
Modal.com runner for A2C and DQN hyperparameter searches.

Mirrors scripts/run_hparam_search_dqn.py and scripts/run_hparam_search_a2c_nengo.py's
methodology exactly (see those modules' docstrings and
docs/02-experimental-design.md) rather than reusing their main() directly —
Modal's container/Volume model needs its own storage-path wiring, so the
search loop (study creation, sampler/pruner, warm-start, post-hoc
re-evaluation) is duplicated here, kept in sync by hand.

All artifacts (CSV trial data, YAML best params, Optuna DB) are written to a
persistent Modal Volume named 'rl-outputs'. They survive container shutdown and
can be retrieved with the modal CLI.

Prerequisites:
    pip install modal
    modal token new          # first-time auth (opens browser)

Usage:
    # Run ALL 9 DQN conditions in parallel (recommended)
    modal run modal_runner.py::dqn_all

    # Single DQN condition
    modal run modal_runner.py::dqn --representation discrete --centering none

    # Run ALL 9 A2C conditions in parallel
    modal run modal_runner.py::a2c_all

    # Single A2C condition
    modal run modal_runner.py::a2c --representation discrete --centering none

    # Retrieve results after run
    modal volume ls rl-outputs
    modal volume get rl-outputs outputs/ ./outputs/
    modal volume get rl-outputs optuna_studies.db ./optuna_studies_modal.db

    # Upload existing A2C best params so DQN warm-start can read them
    modal volume put rl-outputs outputs/hparam_search/best/ outputs/hparam_search/best/

Notes:
  - The remote Optuna DB is at /vol/optuna_studies.db (separate from the local one).
  - DQN warm-start reads A2C best params from the Volume; run A2C first or
    upload local best params with 'modal volume put' above.
  - Each condition runs in its own container. dqn_all / a2c_all spawn all 9
    simultaneously — total wall time = slowest single condition, not 9x.
  - CPU per container: set cpu= in @app.function. Currently 2; raise to 4 if
    you enable n_jobs > 1 in the study.optimize calls below.
"""
import modal

REPS = ["discrete", "ssp", "tile_coding"]
CENTERINGS = ["none", "simple", "value"]

VOL = "/vol"

volume = modal.Volume.from_name("rl-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "gymnasium>=0.28.1",
        "numpy==1.22.4",
        "pandas==1.4.4",
        "scipy==1.7.3",
        "scikit-learn>=1.0",
        "optuna>=2.10.0,<3.0.0",
        "SQLAlchemy<2",
        "alembic<1.9",
        "stable-baselines3>=2.0,<=2.3.2",
        "nengo==3.2.0",
        "nengo-spa==1.3.0",
        "pytry==0.9.2",
        "tqdm==4.64.0",
        "pyyaml",
    )
    # Source code baked into the image; Modal re-builds only this layer when
    # files change (pip layers stay cached).
    .add_local_dir("experiments", remote_path="/project/experiments")
    .add_local_dir("rl", remote_path="/project/rl")
    .add_local_dir("scripts", remote_path="/project/scripts")
    # Best HPO params baked in so ablation containers read the locally-verified YAMLs.
    .add_local_dir("outputs/hparam_search/best", remote_path="/project/outputs/hparam_search/best")
)

app = modal.App("rl-hparam-search", image=image)


# ---------------------------------------------------------------------------
# DQN search
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOL: volume},
    timeout=86400,  # 24 h — no documented Modal cap; raise further if needed
    cpu=4,
    memory=16384,  # 16 GB — Discrete(n_bins=19) needs ~5 GB, TileCoding(65536) needs ~5 GB at buf=10000
)
def _dqn_search(representation: str, centering: str, architecture: str, n_trials: int,
                 n_evaluations: int, n_eval_episodes: int, n_startup_trials: int,
                 top_k: int, reeval_seeds: int, n_jobs: int = 1):
    import sys, os
    sys.path.insert(0, "/project")
    sys.path.insert(0, "/project/scripts")

    import run_hparam_search_dqn as dqn_search

    # Redirect all artifact paths into the mounted Volume
    dqn_search.DATA_DIR = f"{VOL}/outputs/hparam_search"
    dqn_search.BEST_DIR = f"{VOL}/outputs/hparam_search/best"
    dqn_search.A2C_BEST_DIR = f"{VOL}/outputs/hparam_search/best"

    import optuna
    from optuna.trial import TrialState

    os.makedirs(dqn_search.DATA_DIR, exist_ok=True)
    os.makedirs(dqn_search.BEST_DIR, exist_ok=True)

    arch_suffix = "" if architecture == "linear" else f"_{architecture}"
    # _v4 must match STUDY_GEN in run_hparam_search_dqn.py's main().
    study_name = f"dqn_{representation}_{centering}{arch_suffix}_v4_hparam_search"
    # One DB file per condition — eliminates cross-container SQLite write contention.
    storage = f"sqlite:///{VOL}/optuna_dqn_{representation}_{centering}{arch_suffix}.db"

    # TPESampler + MedianPruner, matching run_hparam_search_dqn.py's recipe:
    # no trial can be pruned before 1/3 of its evaluation checkpoints.
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=n_startup_trials,
                                            multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials,
                                            n_warmup_steps=n_evaluations // 3),
        load_if_exists=True,
    )

    # Mark any stale RUNNING trials (from killed containers) as FAILED so the
    # TPE sampler doesn't treat them as in-progress and skew future suggestions.
    stale = [t for t in study.trials if t.state == TrialState.RUNNING]
    if stale:
        print(f"Marking {len(stale)} stale RUNNING trial(s) as FAILED.")
        for t in stale:
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)

    n_existing = len(study.trials)
    warm_started = False
    if n_existing == 0:
        warm_started = dqn_search._enqueue_warmstart(study, representation, centering)
    else:
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if completed:
            print(f"Study '{study_name}': {n_existing} trials recorded. Best: {study.best_value:.2f}")
        else:
            print(f"Study '{study_name}': {n_existing} trials recorded. No completed trials yet.")

    objective = dqn_search.make_objective(representation, centering, architecture,
                                           n_evaluations, n_eval_episodes)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if warm_started:
        # Run warm-start trial single-threaded to avoid workers racing on Trial#0
        study.optimize(objective, n_trials=1, n_jobs=1, show_progress_bar=True)

    n_productive = len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
    remaining = max(0, n_trials - n_productive)
    print(f"Productive trials so far: {n_productive}. Running {remaining} more (target={n_trials}).")
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=n_jobs, show_progress_bar=True)

    print(f"\nSearch complete. Raw search winner (single seed, may be a lucky seed): "
          f"value={study.best_value:.2f}  params={study.best_params}")

    # Post-hoc re-evaluation (araffin.github.io/post/optuna/'s "find lucky
    # seeds" tip), cached on the study so re-invoking doesn't redo it.
    REEVAL_DONE_ATTR = "reeval_done"
    if study.user_attrs.get(REEVAL_DONE_ATTR):
        best_params = study.user_attrs["reeval_best_params"]
        best_value = study.user_attrs["reeval_best_value"]
        print("Post-hoc re-evaluation already recorded for this study — reusing it.")
    else:
        print(f"Re-evaluating top {top_k} trials with {reeval_seeds} seeds each...")
        best_params, best_value, best_std = dqn_search.reevaluate_top_trials(
            study, representation, centering, architecture, top_k, reeval_seeds,
        )
        study.set_user_attr("reeval_best_params", best_params)
        study.set_user_attr("reeval_best_value", best_value)
        study.set_user_attr("reeval_best_std", best_std)
        study.set_user_attr(REEVAL_DONE_ATTR, True)

    print(f"Post-hoc winner: mean={best_value:.2f}  params={best_params}")

    dqn_search.save_best_params(representation, centering, best_value, best_params,
                                 architecture=architecture)
    volume.commit()
    return {"best_value": float(best_value), "best_params": best_params}


@app.local_entrypoint()
def dqn(
    representation: str,
    centering: str,
    architecture: str = "linear",
    n_trials: int = 500,
    n_evaluations: int = 15,
    n_eval_episodes: int = 10,
    n_startup_trials: int = 10,
    top_k: int = 10,
    reeval_seeds: int = 10,
    n_jobs: int = 1,
):
    """Run a single DQN hyperparameter search condition remotely on Modal."""
    result = _dqn_search.remote(representation, centering, architecture, n_trials,
                                 n_evaluations, n_eval_episodes, n_startup_trials,
                                 top_k, reeval_seeds, n_jobs)
    print(f"Done. Best value: {result['best_value']:.2f}")
    print(f"Best params: {result['best_params']}")


@app.local_entrypoint()
def dqn_all(
    architecture: str = "linear",
    n_trials: int = 500,
    n_evaluations: int = 15,
    n_eval_episodes: int = 10,
    n_startup_trials: int = 10,
    top_k: int = 10,
    reeval_seeds: int = 10,
    n_jobs: int = 1,
):
    """Spawn all 9 DQN conditions (for one architecture) in parallel and exit immediately."""
    for rep in REPS:
        for cen in CENTERINGS:
            _dqn_search.spawn(rep, cen, architecture, n_trials, n_evaluations,
                               n_eval_episodes, n_startup_trials, top_k, reeval_seeds, n_jobs)
            print(f"Spawned: {rep}/{cen}")
    print("All 9 DQN jobs submitted. Monitor at modal.com/apps")


# ---------------------------------------------------------------------------
# A2C search
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOL: volume},
    timeout=86400,  # 24 h — no documented Modal cap; raise further if needed
    cpu=4,
)
def _a2c_search(representation: str, centering: str, n_trials: int, n_startup_trials: int,
                 top_k: int, reeval_seeds: int, n_jobs: int = 2):
    import sys, os
    sys.path.insert(0, "/project")
    sys.path.insert(0, "/project/scripts")

    import run_hparam_search_a2c_nengo as a2c_search

    a2c_search.DATA_DIR = f"{VOL}/outputs/hparam_search"
    a2c_search.BEST_DIR = f"{VOL}/outputs/hparam_search/best"

    import optuna
    from optuna.trial import TrialState

    os.makedirs(a2c_search.DATA_DIR, exist_ok=True)
    os.makedirs(a2c_search.BEST_DIR, exist_ok=True)

    # _v2 must match the study-name versioning in run_hparam_search_a2c_nengo.py's main().
    study_name = f"{representation}_{centering}_v2_hparam_search"
    # One DB file per condition — eliminates cross-container SQLite write contention.
    storage = f"sqlite:///{VOL}/optuna_a2c_{representation}_{centering}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=n_startup_trials,
                                            multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials,
                                            n_warmup_steps=a2c_search.BASE_PARAMS["trials"] // 3),
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Study '{study_name}': {n_existing} trials already recorded")
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if completed:
            print(f"Current best: {study.best_value:.2f}  params: {study.best_params}")
        else:
            print("No completed trials yet.")

    # Mark any stale RUNNING trials (from killed containers) as FAILED so the
    # TPE sampler doesn't treat them as in-progress and skew future suggestions.
    stale = [t for t in study.trials if t.state == TrialState.RUNNING]
    if stale:
        print(f"Marking {len(stale)} stale RUNNING trial(s) as FAILED.")
        for t in stale:
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)

    objective = a2c_search.make_objective(representation, centering)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_productive = len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
    remaining = max(0, n_trials - n_productive)
    print(f"Productive trials so far: {n_productive}. Running {remaining} more (target={n_trials}).")
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=n_jobs, show_progress_bar=True)

    print(f"\nSearch complete. Raw search winner (single seed, may be a lucky seed): "
          f"value={study.best_value:.2f}  params={study.best_params}")

    REEVAL_DONE_ATTR = "reeval_done"
    if study.user_attrs.get(REEVAL_DONE_ATTR):
        best_params = study.user_attrs["reeval_best_params"]
        best_value = study.user_attrs["reeval_best_value"]
        print("Post-hoc re-evaluation already recorded for this study — reusing it.")
    else:
        print(f"Re-evaluating top {top_k} trials with {reeval_seeds} seeds each...")
        best_params, best_value, best_std = a2c_search.reevaluate_top_trials(
            study, representation, centering, top_k, reeval_seeds,
        )
        study.set_user_attr("reeval_best_params", best_params)
        study.set_user_attr("reeval_best_value", best_value)
        study.set_user_attr("reeval_best_std", best_std)
        study.set_user_attr(REEVAL_DONE_ATTR, True)

    print(f"Post-hoc winner: mean={best_value:.2f}  params={best_params}")

    a2c_search.save_best_params(representation, centering, best_value, best_params)
    volume.commit()
    return {"best_value": float(best_value), "best_params": best_params}


@app.local_entrypoint()
def a2c(
    representation: str,
    centering: str,
    n_trials: int = 500,
    n_startup_trials: int = 10,
    top_k: int = 10,
    reeval_seeds: int = 10,
    n_jobs: int = 2,
):
    """Run a single A2C hyperparameter search condition remotely on Modal."""
    result = _a2c_search.remote(representation, centering, n_trials, n_startup_trials,
                                 top_k, reeval_seeds, n_jobs)
    print(f"Done. Best value: {result['best_value']:.2f}")
    print(f"Best params: {result['best_params']}")


@app.local_entrypoint()
def a2c_all(
    n_trials: int = 500,
    n_startup_trials: int = 10,
    top_k: int = 10,
    reeval_seeds: int = 10,
    n_jobs: int = 2,
):
    """Spawn all 9 A2C conditions in parallel and exit immediately."""
    for rep in REPS:
        for cen in CENTERINGS:
            _a2c_search.spawn(rep, cen, n_trials, n_startup_trials, top_k, reeval_seeds, n_jobs)
            print(f"Spawned: {rep}/{cen}")
    print("All 9 A2C jobs submitted. Monitor at modal.com/apps")


# ---------------------------------------------------------------------------
# Ablation runs (best params, fixed seeds)
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOL: volume},
    timeout=86400,
    cpu=2,
    memory=8192,
)
def _a2c_ablation(representation: str, centering: str, n_seeds: int = 20, trials: int = 1000):
    import sys, os, numpy as np, yaml
    sys.path.insert(0, "/project")
    sys.path.insert(0, "/project/scripts")
    sys.path.insert(0, "/project/experiments")

    from experiments.trial_cartpole_nengo import ACTrial
    import run_hparam_search_a2c_nengo as a2c_search

    best_path = f"/project/outputs/hparam_search/best/{representation}_{centering}.yaml"
    with open(best_path) as f:
        best_params = yaml.safe_load(f)["params"]

    params = dict(a2c_search.BASE_PARAMS)
    params["trials"] = trials
    params["learnTrials"] = trials
    params.update(best_params)

    data_dir = f"{VOL}/outputs/ablation/a2c/{representation}/{centering}"
    os.makedirs(data_dir, exist_ok=True)

    ac = ACTrial()
    terminal_rewards = []
    for seed in range(n_seeds):
        metadata = ac.run(
            seed=seed,
            data_dir=data_dir,
            pre_comment=f"ablation rep={representation} centering={centering} seed={seed}",
            **params,
        )
        terminal_rewards.append(metadata["terminal_reward"])
        print(f"  seed={seed}  terminal_reward={metadata['terminal_reward']:.1f}"
              f"  episodes_to_learn={metadata['episodes_to_learn']}")
        volume.commit()

    mean_r = float(np.mean(terminal_rewards))
    std_r  = float(np.std(terminal_rewards))
    print(f"\n{representation}/{centering}: mean={mean_r:.1f}  std={std_r:.1f}")
    volume.commit()
    return {"representation": representation, "centering": centering,
            "mean_terminal_reward": mean_r, "std_terminal_reward": std_r,
            "terminal_rewards": terminal_rewards}


@app.local_entrypoint()
def a2c_ablation(
    representation: str,
    centering: str,
    n_seeds: int = 20,
    trials: int = 1000,
):
    """Run a single A2C ablation condition with tuned params."""
    result = _a2c_ablation.remote(representation, centering, n_seeds, trials)
    print(f"Done. mean={result['mean_terminal_reward']:.1f}  std={result['std_terminal_reward']:.1f}")


@app.local_entrypoint()
def a2c_ablation_all(n_seeds: int = 20, trials: int = 1000):
    """Spawn all 9 A2C ablation conditions in parallel."""
    for rep in REPS:
        for cen in CENTERINGS:
            _a2c_ablation.spawn(rep, cen, n_seeds, trials)
            print(f"Spawned: {rep}/{cen}")
    print("All 9 A2C ablation jobs submitted. Monitor at modal.com/apps")
