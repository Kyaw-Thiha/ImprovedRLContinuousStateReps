"""Figures for the RUN_013 report.  Reads saved result files only; trains nothing.

    OMP_NUM_THREADS=1 ../../.venv/bin/python make_report_figures.py

Every supervised number is a HOLD-OUT row (``phase == "final"``, ``n = 10``) via the same filter as
``report_holdout.py``; the RL curves are the saved per-iteration rows.  Output: ``figures/*.png``.
"""

import glob
import json
import pathlib
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

C_REF, C_A1, C_A2, C_BAD, C_GRU, C_SSP, C_OTHER = "#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b"


def finals(tag):
    f = sorted(glob.glob(str(RES / f"sweep_{tag}.jsonl")))
    if not f:
        raise FileNotFoundError(tag)
    rows = [json.loads(l) for l in open(f[0]) if l.strip()]
    rows = [r for r in rows if isinstance(r.get("summary"), dict) and "conv_id" in r["summary"]]
    fin = [r for r in rows if r.get("phase") == "final"] if rows and "phase" in rows[0] else [r for r in rows if r.get("seed", 0) >= 1000]
    return fin


def stat(tag, key="conv_id", sign=1.0):
    v = [sign * r["summary"][key] for r in finals(tag) if r["summary"].get(key) is not None]
    return (st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)) if v else (np.nan, 0.0, 0)


def n95(tag):
    v = [r["summary"]["conv_id"] for r in finals(tag)]
    return sum(x >= 0.95 for x in v), len(v)


def t90(tag):
    v = [r["summary"]["t90_id"] for r in finals(tag) if r["summary"].get("t90_id")]
    return (st.mean(v) if v else np.nan), len(v), len(finals(tag))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------------------------------------------
# Fig 1: the two-axis scoreboard, index T=32 (hold-out conv, annotated with seeds >= 0.95)
# ---------------------------------------------------------------------------------------------------------------
def fig_axes():
    cells = [  # (label, tag, colour, group)
        ("recipe (incumbent, 4 hand-set constants)", "p8_idx32_ssp_recipe", C_REF),
        ("AXIS 1 -- learnable parts", None, None),
        ("input-dependent gates", "p13_gate_input_clean_T32", C_A1),
        ("learned frequencies + codes (rung 2)", "p11_idx32_rung2", C_A1),
        ("K=5 read heads, offset-1 start (search)", "p8_idx32_ssp_k5_o1", C_A1),
        ("K=1, offset-1 start (no search)", "p8_idx32_ssp_k1_o1", C_BAD),
        ("K=5 read heads, offset-3 start", "p8_idx32_ssp_k5_o3", C_BAD),
        ("learned kernel width", "p11_idx32_lslearned", C_BAD),
        ("AXIS 2 -- architecture", None, None),
        ("multi-timescale banks x2", "p12_idx32_banks2", C_OTHER),
        ("multi-timescale banks x4", "p12_idx32_banks4", C_A2),
        ("gates + banks x2", "p12_idx32_gate_banks2", C_A2),
        ("tied addressing, NO prior", "p10_idx32_tied_scr_3k", C_A2),
        ("tied + gates, NO prior", "p11_idx32_tied_gate", C_A2),
        ("tied + prior", "p8_idx32_ssp_tied_rec", C_A2),
        ("delta-rule write", "p8_idx32_ssp_dw", C_OTHER),
        ("interpolating readout", "p11_idx32_interp3", C_OTHER),
        ("dead-zone-free (Cauchy) kernel + prior", "p8_idx32_ssp_cauchy_rec", C_BAD),
        ("PRIOR-FREE: tied + gates + banks x2", "p16_idx32_tied_gate_banks2", C_A2),
    ]
    fig, ax = plt.subplots(figsize=(11, 8))
    y, yt, yl = 0, [], []
    for label, tag, col in cells[::-1]:
        if tag is None:
            ax.text(0.30, y, label, fontsize=10, fontweight="bold", va="center", color="black")
            y += 1
            continue
        m, sd, n = stat(tag)
        k, tot = n95(tag)
        ax.barh(y, m, xerr=sd, color=col, alpha=0.85, edgecolor="k", linewidth=0.4)
        ax.text(max(m + sd, m) + 0.012, y, f"{m:.3f}   {k}/{tot} at 0.95", va="center", fontsize=8.5)
        yt.append(y)
        yl.append(label)
        y += 1
    ax.axvline(0.95, color="k", ls="--", lw=0.8)
    ax.text(0.951, y - 0.4, "0.95", fontsize=8)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=9)
    ax.set_xlim(0.3, 1.18)
    ax.set_xlabel("converged accuracy, index T=32, budget 256 floats (hold-out, n=10; bar = mean, whisker = sd)")
    ax.set_title("Fig 1.  The two axes on one task.  Green/blue = helped (accuracy, speed, or removed a constant)\nred = failed, brown = neutral, grey = incumbent")
    save(fig, "fig1_axes_scoreboard.png")


# ---------------------------------------------------------------------------------------------------------------
# Fig 2: equal aid -- the index lead is withdrawn, what survives is speed at t90, and the extrapolation cost
# ---------------------------------------------------------------------------------------------------------------
def fig_equal_aid():
    arms = [("no aid", "p14_index_recipe_def", "p14_index_fwp_delta_def"),
            ("+ gate", "p14_index_recipe_gate_def", "p14_index_fwp_gate_def"),
            ("+ gate + banks x2", "p14_index_recipe_gate_banks2_def", "p14_index_fwp_gate_banks2_def")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    x = np.arange(len(arms))
    w = 0.36
    for ax, key, title in ((axs[0], "conv_id", "(a) converged accuracy"), (axs[2], "conv_extrap_T", "(c) accuracy on a 4x longer stream")):
        s = [stat(a[1], key)[0] for a in arms]
        f = [stat(a[2], key)[0] for a in arms]
        ax.bar(x - w / 2, s, w, color=C_SSP, label="SSP (recipe)")
        ax.bar(x + w / 2, f, w, color=C_GRU, label="fast-weight (delta rule)")
        for i in range(len(arms)):
            ax.text(x[i] - w / 2, s[i] + 0.01, f"{s[i]:.2f}", ha="center", fontsize=8)
            ax.text(x[i] + w / 2, f[i] + 0.01, f"{f[i]:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([a[0] for a in arms]); ax.set_title(title); ax.set_ylim(0, 1.12)
    ax = axs[1]
    s = [t90(a[1])[0] / 1e3 for a in arms]
    f = [t90(a[2])[0] / 1e3 for a in arms]
    ax.bar(x - w / 2, s, w, color=C_SSP)
    fb = [0 if np.isnan(v) else v for v in f]
    ax.bar(x + w / 2, fb, w, color=C_GRU)
    for i in range(len(arms)):
        ax.text(x[i] - w / 2, s[i] + 1, f"{s[i]:.0f}k", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, (fb[i] if f[i] == f[i] else 0) + 1, f"{f[i]:.0f}k" if f[i] == f[i] else "never", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([a[0] for a in arms]); ax.set_title("(b) tokens to 90% (lower = faster)")
    axs[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Fig 2.  Equal aid on index T=32 (hold-out, n=10): the fast-weight arm catches up on accuracy; SSP keeps a t90 speed edge but is WORSE at extrapolation", fontsize=10)
    save(fig, "fig2_equal_aid.png")


# ---------------------------------------------------------------------------------------------------------------
# Fig 3: interactions -- single changes fail, the combination passes
# ---------------------------------------------------------------------------------------------------------------
def fig_interactions():
    pairs = [("delay 1-4, no prior\n(gates alone -> +banks)", "p11_dly4_tied_gate", "p16_dly4_tied_gate_banks2"),
             ("index T=32, offset-3 start\n(K5 alone -> +gates+banks)", "p8_idx32_ssp_k5_o3", "p16_idx32_gate_banks2_k5_o3"),
             ("index T=48\n(recipe+K5 -> prior-free tied+gate+banks)", "p16_idx48_recipe_k5", "p16_idx48_tied_gate_banks2"),
             ("index T=64\n(tied alone -> n/a, shown for scale)", "p10_idx64_tied_scr", "p10_idx64_tied_scr")]
    pairs = pairs[:3]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(pairs)); w = 0.36
    a = [stat(p[1])[0] for p in pairs]; b = [stat(p[2])[0] for p in pairs]
    ka = [n95(p[1])[0] for p in pairs]; kb = [n95(p[2])[0] for p in pairs]
    ax.bar(x - w / 2, a, w, color=C_BAD, label="single change")
    ax.bar(x + w / 2, b, w, color=C_A2, label="combination")
    for i in range(len(pairs)):
        ax.text(x[i] - w / 2, a[i] + 0.01, f"{a[i]:.2f}\n{ka[i]}/10 pass", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, b[i] + 0.01, f"{b[i]:.2f}\n{kb[i]}/10 pass", ha="center", fontsize=8)
    ax.axhline(0.95, color="k", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pairs], fontsize=8.5)
    ax.set_ylim(0, 1.2); ax.set_ylabel("converged accuracy (hold-out, n=10)"); ax.legend(loc="lower right")
    ax.set_title("Fig 3.  The axes interact: three single-change failures that the combination repairs\n(ablation contrasts only -- the mechanism was not diagnosed)")
    save(fig, "fig3_interactions.png")


# ---------------------------------------------------------------------------------------------------------------
# Fig 4: capability tasks with chance lines
# ---------------------------------------------------------------------------------------------------------------
def fig_capability():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    # (a) distractors at 3000 steps
    d = [("recipe", "p15b_ddist_recipe_3k", C_REF), ("recipe+gate\n+banks2", "p15b_ddist_recipe_gate_banks2_3k", C_SSP),
         ("tied+gate", "p15b_ddist_tied_gate_3k", C_A2), ("fast-weight\n+gate", "p15b_ddist_fwp_gate_3k", C_GRU), ("GRU", "p15b_ddist_gru_3k", C_BAD)]
    m = [stat(t)[0] for _, t, _ in d]; s = [stat(t)[1] for _, t, _ in d]
    axs[0].bar(range(len(d)), m, yerr=s, color=[c for _, _, c in d], edgecolor="k", linewidth=0.4)
    for i, v in enumerate(m): axs[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axs[0].axhline(0.125, color="k", ls=":"); axs[0].text(4.45, 0.135, "chance", fontsize=8, ha="right")
    axs[0].set_xticks(range(len(d))); axs[0].set_xticklabels([x[0] for x in d], fontsize=8); axs[0].set_ylim(0, 1.0)
    axs[0].set_title("(a) delay + distractors, 3000 steps\nnobody passes; gates lift the ceiling; GRU at chance")
    # (b) lag hidden
    d = [("recipe", "p17b_nolag_recipe", C_REF), ("recipe+gate\n+banks2", "p17b_nolag_recipe_gate_banks2", C_SSP),
         ("tied", "p17b_nolag_tied", C_A2), ("tied+gate\n+banks2", "p17b_nolag_tied_gate_banks2", C_A2), ("K5", "p17b_nolag_k5_o1", C_A1),
         ("fast-weight", "p17b_nolag_fwp_delta", C_GRU), ("GRU", "p17b_nolag_gru", C_BAD), ("LRU", "p17b_nolag_lru", C_OTHER)]
    m = [stat(t)[0] for _, t, _ in d]; s = [stat(t)[1] for _, t, _ in d]
    axs[1].bar(range(len(d)), m, yerr=s, color=[c for _, _, c in d], edgecolor="k", linewidth=0.4)
    for i, v in enumerate(m): axs[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axs[1].axhline(0.125, color="k", ls=":"); axs[1].axhline(0.95, color="k", ls="--", lw=0.8)
    axs[1].set_xticks(range(len(d))); axs[1].set_xticklabels([x[0] for x in d], fontsize=7.5); axs[1].set_ylim(0, 1.05)
    axs[1].set_title("(b) delay with the lag HIDDEN (800 steps)\nno architecture solves it; 0 of 80 seeds pass")
    # (c) mixed keys
    d = [("content+position\n(full)", "p18_amix_full", C_A2), ("no position\nkey", "p18_amix_nopos", C_BAD), ("no binding\n(additive)", "p18_amix_nobind", C_BAD),
         ("content-only\nSSP", "p18_amix_content_only_ssp", C_REF), ("GRU", "p18_amix_gru", C_OTHER), ("LRU", "p18_amix_lru", C_OTHER)]
    m = [stat(t)[0] for _, t, _ in d]; s = [stat(t)[1] for _, t, _ in d]
    axs[2].bar(range(len(d)), m, yerr=s, color=[c for _, _, c in d], edgecolor="k", linewidth=0.4)
    for i, v in enumerate(m): axs[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axs[2].axhline(0.25, color="k", ls=":"); axs[2].text(5.4, 0.27, "chance", fontsize=8, ha="right")
    axs[2].axhline(0.625, color="k", ls=":", lw=0.6); axs[2].text(2.0, 0.82, "dotted 0.625 = one mode solved,\nother mode at chance", fontsize=7.5, ha="center")
    axs[2].set_xticks(range(len(d))); axs[2].set_xticklabels([x[0] for x in d], fontsize=7.5); axs[2].set_ylim(0, 1.12)
    axs[2].set_title("(c) content-OR-index queries\nfull arm 10/10; each ablation loses exactly its mode")
    fig.suptitle("Fig 4.  Capability tasks (hold-out, n=10; bar = mean, whisker = sd)", fontsize=11)
    save(fig, "fig4_capability_tasks.png")


# ---------------------------------------------------------------------------------------------------------------
# Fig 5: spatial recall -- the reversal
# ---------------------------------------------------------------------------------------------------------------
def fig_spatial():
    arms = [("SSP", "p19_spatial_ssp", "p21_tuned_ssp", C_SSP), ("GRU", "p19_spatial_gru", "p21_tuned_gru", C_GRU),
            ("LRU", "p19_spatial_lru", "p21_tuned_lru", C_OTHER), ("pose_attn", "p19_spatial_pose_attn", "p21_tuned_pose", C_REF),
            ("fast-weight", "p19_spatial_fwp_delta", "p21_tuned_fwp", C_BAD)]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.6, 1.0, 1.3]})
    ax = axs[0]
    x = np.arange(len(arms)); w = 0.38
    un = [stat(a[1], sign=-1)[0] for a in arms]; tu = [stat(a[2], sign=-1)[0] for a in arms]
    ax.bar(x - w / 2, un, w, color="#cccccc", edgecolor="k", linewidth=0.4, label="UNTUNED (1 config, 1500 steps) -- the first run")
    ax.bar(x + w / 2, tu, w, color=[a[3] for a in arms], edgecolor="k", linewidth=0.4, label="TUNED (4 configs, 4000 steps)")
    for i in range(len(arms)):
        ax.text(x[i] - w / 2, un[i] * 1.12, f"{un[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, tu[i] * 1.12, f"{tu[i]:.3f}", ha="center", fontsize=8)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([a[0] for a in arms]); ax.set_ylim(0.012, 14)
    ax.axhline(1.64, color="k", ls=":"); ax.text(-0.45, 2.0, "no-memory floor 1.64", fontsize=8, ha="left")
    ax.text(3.5, 0.30, "fast-weight and pose_attn:\nIDENTICAL tuned and untuned\n= never trained", fontsize=8, ha="center", color="#8b0000")
    ax.set_ylabel("MSE (log scale, lower = better)"); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(a) spatial recall: tuning flips the winner")
    # gate
    ax = axs[1]
    g = [("no-memory\nMLP", "p21_easy_mlp", C_REF), ("GRU", "p21_easy_gru", C_GRU), ("SSP", "p21_easy_ssp", C_SSP)]
    v = [stat(t, sign=-1)[0] for _, t, _ in g]; sd = [stat(t, sign=-1)[1] for _, t, _ in g]
    ax.bar(range(3), v, yerr=sd, color=[c for _, _, c in g], edgecolor="k", linewidth=0.4)
    for i, val in enumerate(v): ax.text(i, val + 0.04, f"{val:.3f}", ha="center", fontsize=9)
    ax.axhline(0.1 * v[0], color="k", ls="--", lw=0.8); ax.text(2.45, 0.1 * v[0] + 0.03, "pre-registered bar\n(0.1 x floor)", fontsize=7.5, ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels([x[0] for x in g]); ax.set_ylabel("MSE")
    ax.set_title("(b) trainability gate (1 landmark,\nno motion): the GRU FAILS it")
    # splits
    ax = axs[2]
    splits = [("in-dist", "conv_id"), ("2x episode", "conv_extrap_T"), ("2x step size", "conv_extrap_motion"), ("25% pos. noise", "conv_rob_pos")]
    xs = np.arange(len(splits)); w = 0.36
    gru = [stat("p21_tuned_gru", k, -1)[0] for _, k in splits]; ssp = [stat("p21_tuned_ssp", k, -1)[0] for _, k in splits]
    ax.bar(xs - w / 2, gru, w, color=C_GRU, label="GRU (tuned)"); ax.bar(xs + w / 2, ssp, w, color=C_SSP, label="SSP (tuned)")
    for i in range(len(splits)):
        ax.text(xs[i] - w / 2, gru[i] + 0.02, f"{gru[i]:.3f}", ha="center", fontsize=7.5)
        ax.text(xs[i] + w / 2, ssp[i] + 0.02, f"{ssp[i]:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in splits], fontsize=8); ax.set_ylabel("MSE (lower = better)"); ax.legend(fontsize=8)
    ax.set_title("(c) tuned GRU vs tuned SSP, every split:\nthe GRU wins 3 of 4")
    fig.suptitle("Fig 5.  The spatial-recall result REVERSED once the baselines were given a fair chance (hold-out, n=10)", fontsize=11)
    save(fig, "fig5_spatial_reversal.png")


# ---------------------------------------------------------------------------------------------------------------
# Fig 6: RL
# ---------------------------------------------------------------------------------------------------------------
def rl_rows(pattern):
    out = {}
    for f in sorted(glob.glob(str(RES / pattern))):
        seed = int(f.rsplit("_s", 1)[1].split(".")[0])
        rows = [json.loads(l) for l in open(f) if l.strip()]
        out[seed] = [r for r in rows if r.get("row") == "iter" and r.get("episode_return_mean") is not None]
    return out


def first_cross(rows, thr):
    for r in rows:
        if r["episode_return_mean"] >= thr:
            return r["timesteps"]
    return None


def fig_rl():
    ev = json.load(open(RES / "rl_env_returns.json"))["RepeatPreviousEasy"]
    rnd, opt = ev["random_mean"], ev["optimal_mean"]
    thr50, thr90 = rnd + 0.5 * (opt - rnd), rnd + 0.9 * (opt - rnd)
    sets = [("GRU (POPGym)", "rl_pilot_RepeatPreviousEasy_gru_s*.jsonl", C_GRU, 2.4),
            ("SSP tied + lag bank (ours)", "rl_bank_RepeatPreviousEasy_tied_bank_s*.jsonl", C_SSP, 2.4),
            ("SSP tied, lag fixed to the task's (diagnostic)", "rl_tied_RepeatPreviousEasy_tied_lag4_s*.jsonl", C_A2, 1.2),
            ("SSP tied, lag 0 (no selection)", "rl_tied_RepeatPreviousEasy_tied_s*.jsonl", C_BAD, 1.2),
            ("SSP tied + bank + aux loss", "rl_bank_RepeatPreviousEasy_tied_bank_aux_s*.jsonl", C_OTHER, 1.2),
            ("POPGym fast-weight", "rl_pilot_RepeatPreviousEasy_fwp_s*.jsonl", C_REF, 1.2)]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.7, 1.0, 1.1]})
    ax = axs[0]
    for name, pat, col, lw in sets:
        runs = rl_rows(pat)
        for k, (s, rows) in enumerate(sorted(runs.items())):
            ax.plot([r["timesteps"] / 1e3 for r in rows], [r["episode_return_mean"] for r in rows], color=col, lw=lw, alpha=0.85,
                    label=name if k == 0 else None)
    ax.axhline(thr50, color="k", ls=":", lw=0.8); ax.axhline(thr90, color="k", ls=":", lw=0.8)
    ax.text(1015, thr50 + 0.02, "50% threshold", fontsize=7.5, ha="right"); ax.text(1015, thr90 + 0.02, "90% threshold", fontsize=7.5, ha="right")
    ax.axhline(rnd, color="k", ls="--", lw=0.5); ax.text(5, rnd + 0.02, "random", fontsize=7.5)
    ax.set_xlim(0, 1020); ax.set_xlabel("environment steps (thousands)"); ax.set_ylabel("episode return (RepeatPreviousEasy)")
    ax.legend(fontsize=7.5, loc="center right"); ax.set_title("(a) learning curves, 3 seeds per arm (thin = diagnostics)")
    # selector trajectory
    ax = axs[1]
    for s, rows in sorted(rl_rows("rl_bank_RepeatPreviousEasy_tied_bank_s*.jsonl").items()):
        pts = [(r["timesteps"] / 1e3, r["addr"]["lag_eff"], r["addr"]["sel_entropy"]) for r in rows if r.get("addr") and "lag_eff" in r["addr"]]
        ax.plot([p[0] for p in pts], [p[2] for p in pts], color=C_SSP, lw=1.6, label="selector entropy" if s == 0 else None)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C_A2, lw=1.6, ls="--", label="effective lag" if s == 0 else None)
    ax.axhline(3, color="k", ls=":", lw=0.8); ax.text(1015, 3.1, "true recall lag = 3 (k-1)", fontsize=7.5, ha="right")
    ax.set_xlabel("environment steps (thousands)"); ax.set_title("(b) the learned selector concentrates\n(uniform bank of 8 lags at step 0)")
    ax.legend(fontsize=8)
    # crossings
    ax = axs[2]
    gran = 32793 / 1e3
    labs, xs = [], 0
    for name, pat, col in (("GRU", "rl_pilot_RepeatPreviousEasy_gru_s*.jsonl", C_GRU), ("SSP bank", "rl_bank_RepeatPreviousEasy_tied_bank_s*.jsonl", C_SSP)):
        runs = rl_rows(pat)
        for j, (thr, lab) in enumerate(((thr50, "T50"), (thr90, "T90"))):
            v = [first_cross(r, thr) for r in runs.values()]
            v = [t / 1e3 for t in v if t]
            ax.scatter([xs] * len(v), v, color=col, s=45, zorder=3)
            ax.plot([xs - 0.25, xs + 0.25], [np.median(v)] * 2, color="k", lw=1.5)
            ax.text(xs + 0.3, np.median(v), f"{np.median(v):.0f}k", fontsize=8, va="center")
            labs.append(f"{name}\n{lab}"); xs += 1
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel("steps to threshold (thousands)")
    ax.set_title(f"(c) threshold crossings (each dot = 1 seed)\nresolution is one PPO iteration = {gran:.0f}k steps")
    ax.set_xlim(-0.6, 4.2)
    fig.suptitle("Fig 6.  RL on POPGym RepeatPreviousEasy: same PPO config for every arm (32 envs, batch 32768, lr 5e-5, 3 seeds, 1M steps)", fontsize=11)
    save(fig, "fig6_rl.png")


def main():
    for fn in (fig_axes, fig_equal_aid, fig_interactions, fig_capability, fig_spatial, fig_rl):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print("FAILED", fn.__name__, repr(e))


if __name__ == "__main__":
    main()
