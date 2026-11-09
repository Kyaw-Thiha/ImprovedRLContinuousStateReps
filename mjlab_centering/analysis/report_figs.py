import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mjlab_centering import r2
OUT = "../docs/mjlab_centering/figures/"

def rows(camp, job, f):
    try:
        b = r2.client().get_object(Bucket=r2.bucket(), Key=r2.key(f"{camp}/runs/{job}/{f}"))["Body"].read().decode()
        return [json.loads(l) for l in b.splitlines() if l.strip()]
    except Exception:
        return []

# Fig 1: throughput
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
envs, fps = [], []
for n in (1024, 2048, 4096):
    v = [np.median([r["fps"] for r in rows("b1-4d73b6d1", f"bench_n{n}_s{s}", "iter.jsonl")]) for s in (0, 1)]
    envs.append(n); fps.append(np.mean(v))
ax[0].bar([str(e) for e in envs], [f / 1000 for f in fps], color="#4C78A8")
ax[0].set_title("Harness-reported throughput (A40)"); ax[0].set_ylabel("k env-steps/s"); ax[0].set_xlabel("num_envs")
for i, f in enumerate(fps): ax[0].text(i, f / 1000 + 1, f"{f/1000:.0f}k", ha="center")

# Fig 2: reference arm learning curves (plateau runs)
for s, c in ((2000, "#E45756"), (2001, "#F58518")):
    ev = rows("b1-4d73b6d1", f"plateau_ref_s{s}", "eval.jsonl")
    if ev: ax[1].plot([r["iter"] for r in ev], [r["J_task"] for r in ev], "-o", ms=3, color=c, label=f"seed {s} (n={len(ev)} evals)")
ax[1].set_title("Reference arm (none, g=0.99, offset 0)\nincomplete runs: 275 / 224 of 800 iters")
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("J_task (eval, per-step task reward)"); ax[1].legend()

# Fig 3: eta sweep
for e, c in (("0.003", "#54A24B"), ("0.01", "#B279A2")):
    ev = rows("e1-4562778f", f"eta_g0.999_o1.0_value_s2000_eta{e}", "eval.jsonl")
    if ev:
        ax[2].plot([r["iter"] for r in ev], [r["J_task"] for r in ev], "-o", ms=3, color=c,
                   label=f"eta={e}  last5={np.mean([r['J_task'] for r in ev][-5:]):.4f}")
ax[2].set_title("eta sweep (value centering, g=0.999, offset +1.0)\none seed, 600 iters"); ax[2].set_xlabel("iteration"); ax[2].legend()
plt.tight_layout(); plt.savefig(OUT + "fig1_throughput_reference_eta.png", dpi=110); plt.close()

# Fig 4: partial A cells, none vs value, offset 0, g=0.999
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for camp, arm, c in (("u2-649ffe59", "none", "#E45756"), ("c3-37e5a443", "value", "#4C78A8"),
                     ("u4-ae4c81d5", "none", "#E45756"), ("c5-5d4ad9f5", "value", "#4C78A8"),
                     ("u3-ae4c81d5", "none", "#E45756"), ("c4-5d4ad9f5", "value", "#4C78A8")):
    for s in (2000, 2001, 2002):
        ev = rows(camp, f"A_g0.999_o0.0_{arm}_s{s}", "eval.jsonl")
        if len(ev) > 3:
            ax[0].plot([r["iter"] for r in ev], [r["J_task"] for r in ev], color=c, alpha=.6, lw=1.2)
ax[0].plot([], [], color="#E45756", label="none"); ax[0].plot([], [], color="#4C78A8", label="value")
ax[0].set_title("Partial A cells, g=0.999, offset 0\n(no run finished; all partial)"); ax[0].set_xlabel("iteration"); ax[0].set_ylabel("J_task"); ax[0].legend()

# Fig 5: wall-clock accounting (diagnostic job)
it = rows("d1-f61dbe3d", "diag_i150", "iter.jsonl")
if it:
    x = [r["iter"] for r in it]
    ax[1].stackplot(x, [r["t_collect"] for r in it], [r["t_learn"] for r in it], [r.get("t_push", 0) for r in it],
                    labels=["collect", "update", "mirror push"], colors=["#4C78A8", "#F58518", "#B279A2"])
    ax[1].set_title("Diagnostic job: where an iteration's time goes\n(sum ~1.9 s)"); ax[1].set_xlabel("iteration"); ax[1].set_ylabel("s"); ax[1].legend(loc="upper right")
plt.tight_layout(); plt.savefig(OUT + "fig2_partial_A_and_timing.png", dpi=110); plt.close()

# Fig 6: run inventory
fig, ax = plt.subplots(figsize=(9, 3.6))
inv = [("b1 bench+plateau", 8, 6), ("e1 eta sweep", 2, 1), ("u2/c3", 6, 0), ("u4/c5", 3, 0), ("v1 verify", 1, 0)]
y = np.arange(len(inv))
ax.barh(y, [i[1] for i in inv], color="#BBBBBB", label="started")
ax.barh(y, [i[2] for i in inv], color="#54A24B", label="completed (DONE)")
ax.set_yticks(y); ax.set_yticklabels([i[0] for i in inv]); ax.invert_yaxis(); ax.legend(); ax.set_xlabel("runs")
ax.set_title("Runs started vs completed (from R2)")
plt.tight_layout(); plt.savefig(OUT + "fig3_inventory.png", dpi=110); plt.close()
print("ok")
