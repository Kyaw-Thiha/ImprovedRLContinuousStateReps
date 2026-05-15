import argparse
import os

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "processed")
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "reward_centering_20trials")
DEFAULT_OUT_FILE = os.path.join(OUT_DIR, "reward-centering-20trials-metadata-summary.csv")


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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Parse reward-centering trial metadata into a single CSV summary."
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing reward-centering trial folders and metadata .txt files.",
    )
    parser.add_argument(
        "--out-file",
        default=DEFAULT_OUT_FILE,
        help="Output CSV path for the metadata summary.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    data_dir = os.path.abspath(args.data_dir)
    out_file = os.path.abspath(args.out_file)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Reward-centering data directory not found: {data_dir}"
        )

    files = sorted(
        os.path.join(root, file_name)
        for root, _, file_names in os.walk(data_dir)
        for file_name in file_names
        if file_name.endswith(".txt")
    )
    if not files:
        raise FileNotFoundError(
            f"No metadata .txt files found in reward-centering directory: {data_dir}"
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

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    out_df.to_csv(out_file)
    print(f"Saved metadata summary to {out_file}")
    print(out_df.head())


if __name__ == "__main__":
    main()
