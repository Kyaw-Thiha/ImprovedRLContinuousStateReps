"""
Run a single experiment configuration for N seeds.

Usage:
    python scripts/run_experiment.py representation=discrete centering=none
    python scripts/run_experiment.py representation=ssp centering=value n_seeds=5

The Hydra config in conf/ drives all hyperparameters. Results are saved to:
    outputs/{model_type}/{rep_}/{reward_center_mode}/
"""

import sys, os
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# project root on path so rl/ and experiments/ are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

from trial_cartpole import ACTrial


def cfg_to_trial_kwargs(cfg: DictConfig) -> dict:
    """Flatten the Hydra config into kwargs for ACTrial.run()."""
    # Convert to plain dict (resolves interpolations like learnTrials: ${trials})
    flat = OmegaConf.to_container(cfg, resolve=True)

    # Remove runner-only keys not accepted by pytry
    for key in ("n_seeds", "output_root", "model_type"):
        flat.pop(key, None)

    return flat


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    # Derive output dir from config values (consistent naming across runs)
    data_dir = os.path.join(
        cfg.output_root,
        cfg.model_type,
        cfg.rep_,
        cfg.reward_center_mode,
    )
    os.makedirs(data_dir, exist_ok=True)

    kwargs = cfg_to_trial_kwargs(cfg)

    ac = ACTrial()
    terminal_rewards = []

    for seed in range(cfg.n_seeds):
        pre_comment = f"rep={cfg.rep_}, centering={cfg.reward_center_mode}, seed={seed}"
        metadata = ac.run(
            seed=seed,
            data_dir=data_dir,
            pre_comment=pre_comment,
            **kwargs,
        )
        terminal_rewards.append(metadata["terminal_reward"])
        print(f"  seed={seed}  terminal_reward={metadata['terminal_reward']:.1f}  "
              f"episodes_to_learn={metadata['episodes_to_learn']}")

    mean_reward = float(np.mean(terminal_rewards))
    print(f"\nmean terminal reward over {cfg.n_seeds} seeds: {mean_reward:.1f}")

    # Return mean reward so Optuna can use this script directly for HPO
    return mean_reward


if __name__ == "__main__":
    main()
