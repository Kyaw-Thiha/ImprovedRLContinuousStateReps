import os

import pandas as pd


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "reward_centering")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
METADATA_FILE = os.path.join(PROCESSED_DIR, "reward-centering-metadata-summary.csv")
OUT_FILE = os.path.join(PROCESSED_DIR, "reward-centering-all-episodic-rewards.csv")


def find_rewards_file(trial_id):
    expected_suffix = os.path.join(trial_id, "rewards.csv")
    for root, _, file_names in os.walk(DATA_DIR):
        if "rewards.csv" in file_names:
            rewards_path = os.path.join(root, "rewards.csv")
            if rewards_path.endswith(expected_suffix):
                return rewards_path
    raise FileNotFoundError(
        f"Rewards file not found for {trial_id} under reward-centering directory: {DATA_DIR}"
    )


def main():
    if not os.path.isfile(METADATA_FILE):
        raise FileNotFoundError(
            f"Metadata summary not found: {METADATA_FILE}. "
            "Run parse_reward_centering_metadata.py first."
        )

    mddf = pd.read_csv(METADATA_FILE, index_col=0)
    if mddf.empty:
        raise ValueError(f"Metadata summary is empty: {METADATA_FILE}")

    out_df = pd.DataFrame(index=range(1000))

    for _, row in mddf.iterrows():
        trial_id = row["trial_ID"]
        rewards_path = find_rewards_file(trial_id)

        rdf = pd.read_csv(rewards_path, index_col=0)
        rtotal = rdf.sum(axis=0).reset_index(drop=True)

        reward_mode = row.get("reward_center_mode", "unknown")
        seed = row.get("seed", "na")
        rep = row.get("rep_", "unknown")
        n_bins = row.get("n_bins", "na")
        col_name = f"{reward_mode}-{rep}-{n_bins}-seed{seed}"
        out_df[col_name] = rtotal

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_df.to_csv(OUT_FILE)
    print(f"Saved merged episodic rewards to {OUT_FILE}")
    print(out_df.head())
    print(out_df.tail())


if __name__ == "__main__":
    main()
