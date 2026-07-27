"""
Modal.com runner for A2C and DQN hyperparameter searches.

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
)

app = modal.App("rl-hparam-search", image=image)


# ---------------------------------------------------------------------------
# DQN search
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOL: volume},
    timeout=86400,  # 24 h — no documented Modal cap; raise further if needed
    cpu=4,
)
def _dqn_search(representation: str, centering: str, n_trials: int, n_seeds: int, n_jobs: int = 2):
    import sys, os
    sys.path.insert(0, "/project")
    sys.path.insert(0, "/project/scripts")

    import run_hparam_search_dqn as dqn_search

    # Redirect all artifact paths into the mounted Volume
    dqn_search.DB_PATH = f"{VOL}/optuna_studies.db"
    dqn_search.DATA_DIR = f"{VOL}/outputs/hparam_search"
    dqn_search.BEST_DIR = f"{VOL}/outputs/hparam_search/best"
    dqn_search.A2C_BEST_DIR = f"{VOL}/outputs/hparam_search/best"

    import optuna
    from optuna.trial import TrialState

    os.makedirs(dqn_search.DATA_DIR, exist_ok=True)
    os.makedirs(dqn_search.BEST_DIR, exist_ok=True)

    study_name = f"dqn_{representation}_{centering}_hparam_search"
    storage = f"sqlite:///{dqn_search.DB_PATH}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=50,
            max_resource=dqn_search.BASE_PARAMS["trials"],
            reduction_factor=3,
        ),
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    warm_started = False
    if n_existing > 0:
        print(f"Study '{study_name}': {n_existing} trials already recorded")
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if completed:
            print(f"Current best: {study.best_value:.2f}  params: {study.best_params}")
        else:
            print("No completed trials yet.")
    else:
        warm_started = dqn_search._enqueue_warmstart(study, representation, centering)

    objective = dqn_search.make_objective(representation, centering, n_seeds)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    remaining = n_trials
    if warm_started:
        # Run warm-start trial single-threaded to avoid workers racing on Trial#0
        study.optimize(objective, n_trials=1, n_jobs=1, show_progress_bar=True)
        remaining = max(0, n_trials - 1)

    study.optimize(objective, n_trials=remaining, n_jobs=n_jobs, show_progress_bar=True)

    print(f"\nSearch complete. Best value: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")

    dqn_search.save_best_params(representation, centering, study.best_value, study.best_params)
    volume.commit()
    return {"best_value": float(study.best_value), "best_params": study.best_params}


@app.local_entrypoint()
def dqn(
    representation: str,
    centering: str,
    n_trials: int = 100,
    n_seeds: int = 10,
    n_jobs: int = 2,
):
    """Run a single DQN hyperparameter search condition remotely on Modal."""
    result = _dqn_search.remote(representation, centering, n_trials, n_seeds, n_jobs)
    print(f"Done. Best value: {result['best_value']:.2f}")
    print(f"Best params: {result['best_params']}")


@app.local_entrypoint()
def dqn_all(
    n_trials: int = 100,
    n_seeds: int = 10,
    n_jobs: int = 2,
):
    """Spawn all 9 DQN conditions in parallel and exit immediately."""
    conditions = [(r, c, n_trials, n_seeds, n_jobs) for r in REPS for c in CENTERINGS]
    for rep, cen, n_t, n_s, n_j in conditions:
        _dqn_search.spawn(rep, cen, n_t, n_s, n_j)
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
def _a2c_search(representation: str, centering: str, n_trials: int, n_seeds: int, n_jobs: int = 2):
    import sys, os
    sys.path.insert(0, "/project")
    sys.path.insert(0, "/project/scripts")

    import run_hparam_search as a2c_search

    a2c_search.DB_PATH = f"{VOL}/optuna_studies.db"
    a2c_search.DATA_DIR = f"{VOL}/outputs/hparam_search"
    a2c_search.BEST_DIR = f"{VOL}/outputs/hparam_search/best"

    import optuna
    from optuna.trial import TrialState

    os.makedirs(a2c_search.DATA_DIR, exist_ok=True)
    os.makedirs(a2c_search.BEST_DIR, exist_ok=True)

    study_name = f"{representation}_{centering}_hparam_search"
    storage = f"sqlite:///{a2c_search.DB_PATH}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200),
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

    objective = a2c_search.make_objective(representation, centering, n_seeds)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    print(f"\nSearch complete. Best value: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")

    a2c_search.save_best_params(representation, centering, study.best_value, study.best_params)
    volume.commit()
    return {"best_value": float(study.best_value), "best_params": study.best_params}


@app.local_entrypoint()
def a2c(
    representation: str,
    centering: str,
    n_trials: int = 100,
    n_seeds: int = 10,
    n_jobs: int = 2,
):
    """Run a single A2C hyperparameter search condition remotely on Modal."""
    result = _a2c_search.remote(representation, centering, n_trials, n_seeds, n_jobs)
    print(f"Done. Best value: {result['best_value']:.2f}")
    print(f"Best params: {result['best_params']}")


@app.local_entrypoint()
def a2c_all(
    n_trials: int = 100,
    n_seeds: int = 10,
    n_jobs: int = 2,
):
    """Spawn all 9 A2C conditions in parallel and exit immediately."""
    conditions = [(r, c, n_trials, n_seeds, n_jobs) for r in REPS for c in CENTERINGS]
    for rep, cen, n_t, n_s, n_j in conditions:
        _a2c_search.spawn(rep, cen, n_t, n_s, n_j)
        print(f"Spawned: {rep}/{cen}")
    print("All 9 A2C jobs submitted. Monitor at modal.com/apps")
