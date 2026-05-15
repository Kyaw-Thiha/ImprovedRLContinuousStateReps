import argparse
import os

import pandas as pd


BASE_DIR = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "reward_centering_20trials")
DEFAULT_METADATA_FILE = os.path.join(PROCESSED_DIR, "reward-centering-20trials-metadata-summary.csv")
DEFAULT_OUT_FILE = os.path.join(PROCESSED_DIR, "reward-centering-20trials-all-episodic-rewards.csv")


def find_rewards_file(trial_id, data_dir):
    expected_suffix = os.path.join(trial_id, "rewards.csv")
    for root, _, file_names in os.walk(data_dir):
        if "rewards.csv" in file_names:
            rewards_path = os.path.join(root, "rewards.csv")
            if rewards_path.endswith(expected_suffix):
                return rewards_path
    raise FileNotFoundError(
        f"Rewards file not found for {trial_id} under reward-centering directory: {data_dir}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Merge per-trial episodic rewards into a single reward-centering CSV."
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing reward-centering trial folders.",
    )
    parser.add_argument(
        "--metadata-file",
        default=DEFAULT_METADATA_FILE,
        help="Metadata summary CSV produced by parse_reward_centering_metadata.py.",
    )
    parser.add_argument(
        "--out-file",
        default=DEFAULT_OUT_FILE,
        help="Output CSV path for the merged episodic rewards.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    data_dir = os.path.abspath(args.data_dir)
    metadata_file = os.path.abspath(args.metadata_file)
    out_file = os.path.abspath(args.out_file)

    if not os.path.isfile(metadata_file):
        raise FileNotFoundError(
            f"Metadata summary not found: {metadata_file}. "
            "Run parse_reward_centering_metadata.py first."
        )

    mddf = pd.read_csv(metadata_file, index_col=0)
    if mddf.empty:
        raise ValueError(f"Metadata summary is empty: {metadata_file}")

    reward_columns = {}

    for _, row in mddf.iterrows():
        trial_id = row["trial_ID"]
        rewards_path = find_rewards_file(trial_id, data_dir)

        rdf = pd.read_csv(rewards_path, index_col=0)
        rtotal = rdf.sum(axis=0).reset_index(drop=True)
        reward_columns[str(trial_id)] = rtotal

    out_df = pd.DataFrame(reward_columns)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    out_df.to_csv(out_file)
    print(f"Saved merged episodic rewards to {out_file}")
    print(out_df.head())
    print(out_df.tail())


if __name__ == "__main__":
    main()
