"""
Run the full 3x3 ablation grid (representation × centering) for a given experiment type.

Usage:
    python scripts/run_ablation.py                          # all 3x3 combos, AC
    python scripts/run_ablation.py experiment=actor_critic  # explicit
    python scripts/run_ablation.py n_seeds=5                # quick test run

Each combination is run sequentially. Results land in:
    outputs/{model_type}/{rep_}/{reward_center_mode}/
"""

import sys, os
import subprocess

REPRESENTATIONS = ["discrete", "ssp", "tile_coding"]
CENTERINGS = ["none", "simple", "value"]

RUNNER = os.path.join(os.path.dirname(__file__), "run_experiment.py")


def main():
    # Pass through any extra args (e.g. n_seeds=5, experiment=actor_critic)
    extra_args = sys.argv[1:]

    total = len(REPRESENTATIONS) * len(CENTERINGS)
    done = 0

    for rep in REPRESENTATIONS:
        for centering in CENTERINGS:
            done += 1
            print(f"\n[{done}/{total}] representation={rep}  centering={centering}")
            cmd = [
                sys.executable, RUNNER,
                f"representation={rep}",
                f"centering={centering}",
            ] + extra_args

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  FAILED (exit code {result.returncode}) — continuing...")

    print(f"\nAblation complete: {total} configurations run.")


if __name__ == "__main__":
    main()
