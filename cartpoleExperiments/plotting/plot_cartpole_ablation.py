import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ablation_plotting import (
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_SOLVE_THRESHOLD,
    compute_learning_curve_stats,
    condition_order,
    load_metadata,
    load_rewards,
    plot_interaction,
    plot_learning_curves,
    plot_summary_panels,
    save_summary_table,
    summarize_conditions,
)


def apply_plot_style():
    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style_name)
            return style_name
        except OSError:
            continue
    return "default"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate reusable CartPole ablation figures from processed metadata and episodic reward CSVs."
    )
    parser.add_argument("--metadata", required=True, help="Path to processed metadata summary CSV.")
    parser.add_argument("--rewards", required=True, help="Path to merged episodic rewards CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory where plots and tables will be saved.")
    parser.add_argument(
        "--prefix",
        default="cartpole-ablation",
        help="Filename prefix for generated outputs.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=DEFAULT_ROLLING_WINDOW,
        help="Rolling window size for the learning-curve figure.",
    )
    parser.add_argument(
        "--solve-threshold",
        type=float,
        default=DEFAULT_SOLVE_THRESHOLD,
        help="Reward threshold used to define solved runs.",
    )
    parser.add_argument(
        "--uncertainty",
        choices=["std", "quantile"],
        default="quantile",
        help="Uncertainty band for the learning-curve figure.",
    )
    parser.add_argument(
        "--quantile-low",
        type=float,
        default=5.0,
        help="Lower percentile for quantile bands.",
    )
    parser.add_argument(
        "--quantile-high",
        type=float,
        default=95.0,
        help="Upper percentile for quantile bands.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    rewards_path = Path(args.rewards)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    rewards, reward_columns = load_rewards(rewards_path)

    conditions = condition_order(metadata)
    _, curve_stats = compute_learning_curve_stats(rewards, reward_columns, rolling_window=args.rolling_window)
    summary_df = summarize_conditions(
        metadata,
        solve_threshold=args.solve_threshold,
        censor_unsolved=False,
        total_episodes=len(rewards.index),
    )

    style_name = apply_plot_style()

    learning_curve_path = out_dir / f"{args.prefix}-learning-curves.png"
    summary_path = out_dir / f"{args.prefix}-summary-panels.png"
    interaction_terminal_path = out_dir / f"{args.prefix}-interaction-terminal-reward.png"
    interaction_learn_path = out_dir / f"{args.prefix}-interaction-episodes-to-learn.png"
    summary_table_path = out_dir / f"{args.prefix}-summary-table.csv"

    plot_learning_curves(
        curve_stats,
        conditions,
        learning_curve_path,
        rolling_window=args.rolling_window,
        solve_threshold=args.solve_threshold,
        uncertainty=args.uncertainty,
        quantile_low=args.quantile_low,
        quantile_high=args.quantile_high,
    )
    plot_summary_panels(metadata, conditions, summary_path, total_episodes=len(rewards.index))
    plot_interaction(
        summary_df,
        interaction_terminal_path,
        metric_col="terminal_reward_mean",
        ylabel="Mean Terminal Reward",
        title="Interaction: Representation x Reward Centering",
    )
    plot_interaction(
        summary_df,
        interaction_learn_path,
        metric_col="episodes_to_learn_mean",
        ylabel="Mean Episodes To Learn (Solved Runs)",
        title="Interaction: Sample Efficiency by Representation",
    )
    save_summary_table(summary_df, summary_table_path)

    print(f"Saved learning curves to {learning_curve_path}")
    print(f"Saved summary panels to {summary_path}")
    print(f"Saved terminal-reward interaction plot to {interaction_terminal_path}")
    print(f"Saved sample-efficiency interaction plot to {interaction_learn_path}")
    print(f"Saved summary table to {summary_table_path}")
    print(f"Plot style: {style_name}")


if __name__ == "__main__":
    main()
