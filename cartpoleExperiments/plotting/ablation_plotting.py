from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_SOLVE_THRESHOLD = 195.0
DEFAULT_ROLLING_WINDOW = 100
CONDITION_COLORS = {
    ("Discrete", "none"): "#6c757d",
    ("Discrete", "simple"): "#f28e2b",
    ("Discrete", "value"): "#4e79a7",
    ("PlaceSSP", "none"): "#6c757d",
    ("PlaceSSP", "simple"): "#f28e2b",
    ("PlaceSSP", "value"): "#4e79a7",
}
MODE_COLORS = {"none": "#4c78a8", "simple": "#e15759", "value": "#76b7b2"}
MODE_LABELS = {"none": "No Centering", "simple": "Simple Centering", "value": "Value Centering"}
REP_LABELS = {"Discrete": "Discrete", "PlaceSSP": "PlaceSSP", "HexSSP": "HexSSP"}
ETA_COLOR_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#b07aa1",
    "#76b7b2",
    "#9c755f",
    "#edc948",
]


@dataclass(frozen=True)
class Condition:
    rep: str
    condition_col: str
    condition_value: object

    @property
    def key(self):
        return (self.rep, self.condition_value)

    def label(self):
        rep_label = REP_LABELS.get(self.rep, self.rep)
        if self.condition_col == "reward_center_mode":
            value_label = MODE_LABELS.get(self.condition_value, self.condition_value)
        elif self.condition_col == "reward_center_eta":
            value_label = f"eta={self.condition_value:g}"
        else:
            value_label = f"{self.condition_col}={self.condition_value}"
        return f"{rep_label} / {value_label}"

    @property
    def linestyle(self):
        if self.rep == "PlaceSSP":
            return "--"
        return "-"


def load_metadata(metadata_path):
    metadata = pd.read_csv(metadata_path, index_col=0)
    required = {"rep_", "reward_center_mode", "seed", "terminal_reward", "episodes_to_learn"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata file is missing required columns: {sorted(missing)}")
    return metadata


def load_rewards(rewards_path):
    rewards = pd.read_csv(rewards_path, index_col=0)
    if rewards.empty:
        raise ValueError(f"Rewards file is empty: {rewards_path}")
    return rewards


def condition_order(metadata, condition_col):
    order = []
    values = list(metadata[condition_col].drop_duplicates())
    if condition_col == "reward_center_eta":
        values = sorted(values)
    for rep in metadata["rep_"].drop_duplicates():
        for value in values:
            group = metadata[(metadata["rep_"] == rep) & (metadata[condition_col] == value)]
            if not group.empty:
                order.append(Condition(rep=rep, condition_col=condition_col, condition_value=value))
    return order


def summarize_conditions(
    metadata,
    condition_col,
    solve_threshold=DEFAULT_SOLVE_THRESHOLD,
    censor_unsolved=False,
    total_episodes=None,
):
    rows = []
    for condition in condition_order(metadata, condition_col):
        group = metadata[
            (metadata["rep_"] == condition.rep) & (metadata[condition_col] == condition.condition_value)
        ].copy()
        solved_mask = group["episodes_to_learn"].notna()
        solve_rate = solved_mask.mean()

        solved_episodes = group.loc[solved_mask, "episodes_to_learn"].copy()
        episodes = solved_episodes.copy()
        if censor_unsolved and total_episodes is not None:
            episodes = group["episodes_to_learn"].fillna(total_episodes)

        rows.append(
            {
                "rep_": condition.rep,
                condition_col: condition.condition_value,
                "condition_label": condition.label(),
                "n_runs": len(group),
                "n_solved": int(solved_mask.sum()),
                "solve_rate": solve_rate,
                "terminal_reward_mean": group["terminal_reward"].mean(),
                "terminal_reward_std": group["terminal_reward"].std(ddof=1),
                "terminal_reward_median": group["terminal_reward"].median(),
                "episodes_to_learn_mean": np.nan if episodes.empty else episodes.mean(),
                "episodes_to_learn_std": np.nan if len(episodes) <= 1 else episodes.std(ddof=1),
                "episodes_to_learn_median": np.nan if episodes.empty else episodes.median(),
            }
        )
    return pd.DataFrame(rows)


def build_run_info(metadata, rewards, condition_col):
    run_info = metadata.copy()
    if "trial_ID" not in run_info.columns:
        raise ValueError("Metadata must contain a trial_ID column.")

    reward_columns = {str(col) for col in rewards.columns}
    run_info["column"] = run_info["trial_ID"].astype(str)
    run_info["condition_value"] = run_info[condition_col]

    missing_rewards = sorted(set(run_info["column"]) - reward_columns)
    if missing_rewards:
        raise ValueError(
            "Some metadata trial_ID values are missing from the rewards CSV. "
            f"Examples: {missing_rewards[:5]}"
        )

    extra_rewards = sorted(reward_columns - set(run_info["column"]))
    if extra_rewards:
        raise ValueError(
            "Some rewards columns do not have matching metadata trial_ID rows. "
            f"Examples: {extra_rewards[:5]}"
        )

    return run_info.reset_index(drop=True)


def compute_learning_curve_stats(rewards, run_info, condition_col, rolling_window=DEFAULT_ROLLING_WINDOW):
    frames = []
    for _, row in run_info.iterrows():
        series = rewards[row["column"]].astype(float)
        rolling = series.rolling(rolling_window, min_periods=rolling_window).mean()
        tmp = pd.DataFrame(
            {
                "episode": rewards.index.astype(int),
                "reward": series.to_numpy(),
                "rolling_reward": rolling.to_numpy(),
                "seed": row["seed"],
                "rep_": row["rep_"],
                condition_col: row[condition_col],
            }
        )
        tmp = tmp.dropna(subset=["rolling_reward"]).reset_index(drop=True)
        frames.append(tmp)
    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(["rep_", condition_col, "episode"])["rolling_reward"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    quantiles = (
        combined.groupby(["rep_", condition_col, "episode"])["rolling_reward"]
        .quantile([0.05, 0.95])
        .unstack()
        .reset_index()
        .rename(columns={0.05: "q05", 0.95: "q95"})
    )
    grouped = grouped.merge(quantiles, on=["rep_", condition_col, "episode"], how="left")
    return combined, grouped


def condition_colors(conditions, condition_col):
    if condition_col == "reward_center_mode":
        return {
            condition.key: CONDITION_COLORS.get((condition.rep, condition.condition_value), "#4e79a7")
            for condition in conditions
        }

    unique_values = []
    for condition in conditions:
        if condition.condition_value not in unique_values:
            unique_values.append(condition.condition_value)

    if condition_col == "reward_center_eta":
        unique_values = sorted(unique_values)

    value_to_color = {
        value: ETA_COLOR_PALETTE[i % len(ETA_COLOR_PALETTE)] for i, value in enumerate(unique_values)
    }
    return {condition.key: value_to_color[condition.condition_value] for condition in conditions}


def _format_mode_text(mode):
    return MODE_LABELS.get(mode, mode)


def _format_rep_text(rep):
    return REP_LABELS.get(rep, rep)


def save_summary_table(summary_df, out_path):
    table_df = summary_df.copy()
    table_df["solve_rate_pct"] = (100 * table_df["solve_rate"]).round(1)
    table_df["terminal_reward_mean_std"] = table_df.apply(
        lambda row: f'{row["terminal_reward_mean"]:.2f} +/- {row["terminal_reward_std"]:.2f}', axis=1
    )
    table_df["episodes_to_learn_mean_std"] = table_df.apply(
        lambda row: (
            "NA"
            if pd.isna(row["episodes_to_learn_mean"])
            else f'{row["episodes_to_learn_mean"]:.2f} +/- {row["episodes_to_learn_std"]:.2f}'
        ),
        axis=1,
    )
    keep = [
        "condition_label",
        "n_runs",
        "n_solved",
        "solve_rate_pct",
        "terminal_reward_mean_std",
        "terminal_reward_median",
        "episodes_to_learn_mean_std",
        "episodes_to_learn_median",
    ]
    table_df[keep].to_csv(out_path, index=False)


def plot_learning_curves(
    curve_stats,
    conditions,
    condition_col,
    out_path,
    rolling_window,
    solve_threshold,
    uncertainty="quantile",
    quantile_low=5.0,
    quantile_high=95.0,
):
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    colors = condition_colors(conditions, condition_col)

    for condition in conditions:
        subset = curve_stats[
            (curve_stats["rep_"] == condition.rep) & (curve_stats[condition_col] == condition.condition_value)
        ].sort_values("episode")
        y = subset["mean"].to_numpy()
        if uncertainty == "std":
            spread = subset["std"].fillna(0).to_numpy()
            lower = y - spread
            upper = y + spread
        else:
            lower_col = f"q{int(quantile_low):02d}"
            upper_col = f"q{int(quantile_high):02d}"
            if lower_col not in subset.columns or upper_col not in subset.columns:
                raise ValueError(
                    f"Quantile columns {lower_col}/{upper_col} are not available in learning-curve stats."
                )
            lower = subset[lower_col].to_numpy()
            upper = subset[upper_col].to_numpy()

        ax.plot(
            subset["episode"],
            y,
            lw=2.4,
            color=colors[condition.key],
            linestyle=condition.linestyle,
            label=condition.label(),
        )
        ax.fill_between(subset["episode"], lower, upper, color=colors[condition.key], alpha=0.10)

    ax.axhline(solve_threshold, color="#222222", lw=1.2, ls="--", alpha=0.8)
    ax.set_title(f"CartPole Learning Curves ({rolling_window}-Episode Rolling Mean)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling Mean Episodic Reward")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2, linewidth=0.6)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_summary_panels(metadata, conditions, condition_col, out_path, total_episodes):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), constrained_layout=True)
    positions = np.arange(len(conditions))
    colors = condition_colors(conditions, condition_col)

    terminal_groups = []
    episode_groups = []
    point_rng = np.random.default_rng(7)

    for condition in conditions:
        group = metadata[
            (metadata["rep_"] == condition.rep) & (metadata[condition_col] == condition.condition_value)
        ]
        terminal_groups.append(group["terminal_reward"].dropna().to_numpy())
        episode_groups.append(group["episodes_to_learn"].fillna(total_episodes).to_numpy())

    term_ax = axes[0]
    term_box = term_ax.boxplot(
        terminal_groups,
        positions=positions,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#111111", "linewidth": 1.6},
    )
    for patch, condition in zip(term_box["boxes"], conditions):
        patch.set_facecolor(colors[condition.key])
        patch.set_alpha(0.35)
        patch.set_edgecolor(colors[condition.key])
    for i, (condition, values) in enumerate(zip(conditions, terminal_groups)):
        jitter = point_rng.normal(0, 0.045, size=len(values))
        term_ax.scatter(np.full(len(values), i) + jitter, values, color=colors[condition.key], alpha=0.7, s=26)
    term_ax.set_title("Terminal Reward")
    term_ax.set_ylabel("Reward")
    term_ax.set_xticks(positions)
    term_ax.set_xticklabels([condition.label() for condition in conditions], rotation=18, ha="right")
    term_ax.grid(axis="y", alpha=0.2, linewidth=0.6)

    ep_ax = axes[1]
    ep_box = ep_ax.boxplot(
        episode_groups,
        positions=positions,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#111111", "linewidth": 1.6},
    )
    for patch, condition in zip(ep_box["boxes"], conditions):
        patch.set_facecolor(colors[condition.key])
        patch.set_alpha(0.35)
        patch.set_edgecolor(colors[condition.key])
    for i, condition in enumerate(conditions):
        group = metadata[
            (metadata["rep_"] == condition.rep) & (metadata[condition_col] == condition.condition_value)
        ].copy()
        group["episode_plot"] = group["episodes_to_learn"].fillna(total_episodes)
        jitter = point_rng.normal(0, 0.045, size=len(group))
        solved = group["episodes_to_learn"].notna().to_numpy()
        ep_ax.scatter(
            np.full(len(group), i)[solved] + jitter[solved],
            group.loc[solved, "episode_plot"],
            color=colors[condition.key],
            alpha=0.75,
            s=26,
        )
        ep_ax.scatter(
            np.full(len(group), i)[~solved] + jitter[~solved],
            group.loc[~solved, "episode_plot"],
            color=colors[condition.key],
            alpha=0.9,
            s=42,
            marker="X",
        )
        solve_rate = 100 * solved.mean()
        ep_ax.text(i, total_episodes + total_episodes * 0.03, f"{solve_rate:.0f}% solved", ha="center", va="bottom", fontsize=9)

    ep_ax.set_title("Episodes To Learn")
    ep_ax.set_ylabel("Episode")
    ep_ax.set_xticks(positions)
    ep_ax.set_xticklabels([condition.label() for condition in conditions], rotation=18, ha="right")
    ep_ax.set_ylim(0, total_episodes * 1.1)
    ep_ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    unsolved_handle = Line2D([0], [0], marker="X", color="none", markerfacecolor="#444444", markersize=8, label="Unsolved (censored)")
    ep_ax.legend(handles=[unsolved_handle], frameon=False, loc="lower right")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_interaction(summary_df, condition_col, out_path, metric_col, ylabel, title):
    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    reps = list(summary_df["rep_"].drop_duplicates())
    values = list(summary_df[condition_col].drop_duplicates())
    x = np.arange(len(reps))
    colors = condition_colors(
        [Condition(rep=reps[0], condition_col=condition_col, condition_value=value) for value in values],
        condition_col,
    )

    for value in values:
        subset = summary_df[summary_df[condition_col] == value].set_index("rep_").reindex(reps)
        y = subset[metric_col].to_numpy(dtype=float)
        valid = ~np.isnan(y)
        if condition_col == "reward_center_mode":
            label = MODE_LABELS.get(value, value)
        elif condition_col == "reward_center_eta":
            label = f"eta={value:g}"
        else:
            label = f"{condition_col}={value}"
        ax.plot(
            x[valid],
            y[valid],
            marker="o",
            lw=2.2,
            ms=7,
            color=colors[(reps[0], value)],
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_format_rep_text(rep) for rep in reps])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
