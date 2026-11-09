"""
Single source of truth for the `eval_v3` phase: R2 key layout and seed conventions.

Every script that reads or writes eval_v3 artifacts MUST build keys through this module (never hand-write
a key string), so the layout in the bucket stays consistent. Nothing here touches R2; it only builds strings.

R2 layout (bucket root = the R2_BUCKET from .env):

  stage0_cartpole/eval_v3/
    <study> for A2C = a2c_<rep>_<centering>_port (cell "rep/centering/a2c"); same files as a DQN study except
    no dqn_settings variants.
    _code/                                    content-hashed code tarballs for this phase
    _shared/probe_states.npz                  fixed probe states + pooled observation-noise sigma (see below)
    <study>/                                  <study> = dqn_<rep>_<centering>[_deep]_v4_hparam_search
      policies/seed_<SEED>.pt                 final policy of each seed (SEED = 3000 + seed_idx)
      train_runs.jsonl                        one record per seed (metrics, probe history, return lists)
      train_summary.json                      aggregate of train_runs.jsonl
      manifest.json                           code hash, library versions, winner params, policy sha256s
      robustness_runs.jsonl                   one record per (seed, condition)
      deployment.json                         size / quantization / latency
    dqn_settings/<study>/<variant>/           the DQN-settings test (no policies saved)
      train_runs.jsonl  train_summary.json  manifest.json

Never write into refilter_v2/, best_params/, optuna_dbs/ or backup_*/ from eval_v3 code.

Seeds: the policy pass trains seeds POLICY_SEED_OFFSET + idx (3000..3019). The DQN-settings variants reuse
seeds 3000..3009 (VARIANT_N_SEEDS = 10) so every variant seed is paired with the same seed of the baseline pass
(same initial weights, same env seed stream). Earlier phases used 0..499 (search), 1000+ (filter), 2000+ (final).
"""

STAGE = "stage0_cartpole"
ROOT = f"{STAGE}/eval_v3"
POLICY_SEED_OFFSET = 3000
POLICY_N_SEEDS = 20
VARIANT_N_SEEDS = 10

REPS = ("discrete", "ssp", "tile_coding")
CENTERINGS = ("none", "simple", "value")
ARCHS = ("linear", "deep")
ALL_CELLS = [f"{r}/{c}/{a}" for r in REPS for c in CENTERINGS for a in ARCHS]
# A2C port cells: "rep/centering/a2c" (the third field is the algorithm, there is no linear/deep split: A2C is one
# linear actor-critic). They are NOT in ALL_CELLS so every DQN default ("all cells") is unchanged.
A2C_ARCH = "a2c"
A2C_CELLS = [f"{r}/{c}/{A2C_ARCH}" for r in REPS for c in CENTERINGS]
# The cells on which the pre-declared SSP-vs-tile robustness comparison is made (solid baselines only).
PRIMARY_REPS = ("ssp", "tile_coding")
PRIMARY_CENTERINGS = ("none", "value")


def parse_cell(cell: str):
    rep, cen, arch = cell.split("/")
    assert rep in REPS and cen in CENTERINGS and arch in ARCHS + (A2C_ARCH,), cell
    return rep, cen, arch


def is_a2c(cell: str) -> bool:
    return cell.split("/")[-1] == A2C_ARCH


def study_name(cell: str) -> str:
    rep, cen, arch = parse_cell(cell)
    if arch == A2C_ARCH:
        return f"a2c_{rep}_{cen}_port"
    return f"dqn_{rep}_{cen}{'' if arch == 'linear' else '_deep'}_v4_hparam_search"


def cell_from_study(study: str) -> str:
    """Inverse of study_name()."""
    for cell in ALL_CELLS + A2C_CELLS:
        if study_name(cell) == study:
            return cell
    raise ValueError(f"not an eval cell study: {study!r}")


def winner_params_key(cell: str) -> str:
    """The filter winner's config written by the previous phase (READ ONLY for eval_v3)."""
    rep, cen, arch = parse_cell(cell)
    if arch == A2C_ARCH:
        raise ValueError("A2C winners are repo files (conf/a2c_winners/), not on R2: use a2c_winner_path()")
    return f"{STAGE}/best_params/dqn_{rep}_{cen}{'' if arch == 'linear' else '_deep'}_v2.yaml"


def a2c_winner_path(cell: str) -> str:
    """The legacy A2C winner yaml of an A2C cell: a tracked copy under conf/a2c_winners/ (the originals in
    outputs/hparam_search/best/ are git-ignored and not in the pod code tarball)."""
    import os
    rep, cen, arch = parse_cell(cell)
    assert arch == A2C_ARCH, cell
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf", "a2c_winners",
                        f"{rep}_{cen}.yaml")


def variant_name(buffer_size: int = None, target_update_freq: int = None) -> str:
    """e.g. buf50000_tu100. None means 'keep the baseline value' and is omitted; baseline -> 'baseline'."""
    bits = []
    if buffer_size is not None:
        bits.append(f"buf{int(buffer_size)}")
    if target_update_freq is not None:
        bits.append(f"tu{int(target_update_freq)}")
    return "_".join(bits) or "baseline"


def code_prefix() -> str:
    return f"{ROOT}/_code/"


def shared_key(name: str) -> str:
    return f"{ROOT}/_shared/{name}"


PROBE_KEY = shared_key("probe_states.npz")


def cell_prefix(study: str, variant: str = None) -> str:
    """Per-study folder; with `variant` it is the DQN-settings test folder instead."""
    if variant:
        return f"{ROOT}/dqn_settings/{study}/{variant}"
    return f"{ROOT}/{study}"


def policy_key(study: str, seed: int) -> str:
    return f"{cell_prefix(study)}/policies/seed_{int(seed)}.pt"


def policies_prefix(study: str) -> str:
    return f"{cell_prefix(study)}/policies/"


def train_runs_key(study: str, variant: str = None) -> str:
    return f"{cell_prefix(study, variant)}/train_runs.jsonl"


def train_summary_key(study: str, variant: str = None) -> str:
    return f"{cell_prefix(study, variant)}/train_summary.json"


def manifest_key(study: str, variant: str = None) -> str:
    return f"{cell_prefix(study, variant)}/manifest.json"


def robustness_key(study: str) -> str:
    return f"{cell_prefix(study)}/robustness_runs.jsonl"


def deployment_key(study: str) -> str:
    return f"{cell_prefix(study)}/deployment.json"


def local_path(key: str, local_root: str = None) -> str:
    """Local mirror of an eval_v3 key: <local_root>/<key without the 'stage0_cartpole/eval_v3/' prefix>.
    Default local_root = <repo>/outputs/eval_v3. Every runner takes a --local-root override (tests use a tmp dir)."""
    import os
    assert key.startswith(ROOT + "/"), key
    root = local_root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "outputs", "eval_v3")
    return os.path.join(root, key[len(ROOT) + 1:])
