import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "reward_centering")
OUT_DIR = os.path.join(BASE_DIR, "processed")
OUT_FILE = os.path.join(OUT_DIR, "reward-centering-metadata-summary.csv")


FLOAT_COLS = [
    "env_dt",
    "length_scale",
    "eps",
    "lr",
    "act_dis",
    "state_dis",
    "active_prop",
    "terminal_reward_learning",
    "terminal_reward",
    "build_time",
    "total_time",
    "avg_trial_time",
    "reward_center_beta",
    "reward_center_eta",
    "reward_center_init",
]
INT_COLS = [
    "seed",
    "trials",
    "steps",
    "n_done",
    "n_reset",
    "n_bins",
    "n_rotates",
    "learnTrials",
    "state_neurons",
    "dimensionality",
    "episodes_to_learn",
]


def parse_line(line):
    if " = " not in line:
        return None

    col, value = line.split(" = ", 1)
    value = value.strip()

    if col in FLOAT_COLS:
        try:
            return col, float(value)
        except ValueError:
            return col, np.nan

    if col in INT_COLS:
        try:
            return col, int(value)
        except ValueError:
            return col, np.nan

    return col, value.strip("'")


def main():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Reward-centering data directory not found: {DATA_DIR}"
        )

    files = sorted(
        os.path.join(root, file_name)
        for root, _, file_names in os.walk(DATA_DIR)
        for file_name in file_names
        if file_name.endswith(".txt")
    )
    if not files:
        raise FileNotFoundError(
            f"No metadata .txt files found in reward-centering directory: {DATA_DIR}"
        )

    rows = []
    col_labels = []

    with open(files[0]) as metadata_file:
        for line in metadata_file:
            parsed = parse_line(line)
            if parsed is not None:
                col_labels.append(parsed[0])

    for file_path in files:
        temp_data = {}
        with open(file_path) as metadata_file:
            for line in metadata_file:
                parsed = parse_line(line)
                if parsed is not None:
                    col, value = parsed
                    temp_data[col] = value
        rows.append(temp_data)

    out_df = pd.DataFrame(rows, columns=col_labels)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_df.to_csv(OUT_FILE)
    print(f"Saved metadata summary to {OUT_FILE}")
    print(out_df.head())


if __name__ == "__main__":
    main()
