"""
eval_v3 deployment metrics for ONE cell (rep/centering/arch), computed on the saved policies of the
policy pass. Writes a single JSON (layout.deployment_key) with three independently runnable sections:

  size          bytes of every stored component at fp32 / fp16 / int8 (network tensors, encoder constants,
                totals), plus dim / n_params / n_active.
  quantization  greedy return of each saved policy after simulated fp16 / int8 weight quantization (and an
                fp16 SSP phase matrix), PAIRED with the fp32 policy: same env seed stream for every variant.
  latency       per-call wall time of encode / q_from_phi / act on one policy, single thread, plus analytic
                operation counts (the part that does not depend on this machine or on Python).

WHY separate sections: latency is only meaningful on a quiet machine, while size/quantization are cheap and
parallel. `--sections latency` re-runs just that part and merges it into the existing deployment.json.

Usage (offline):
    python scripts/run_deployment.py --cell ssp/none/deep --local-root /tmp/eval_v3 --no-r2 --workers 4
    python scripts/run_deployment.py --cell tile_coding/value/linear --sections latency --no-r2
"""

import argparse
import concurrent.futures as cf
import copy
import gc
import json
import multiprocessing as mp
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import eval_v3_layout as layout  # noqa: E402
import parallel_jobs  # noqa: E402

BYTES_PER_ELEM = {"fp32": 4, "fp16": 2, "int8": 1}
PRECISIONS = ("fp32", "fp16", "int8")
QUANT_VARIANTS = ("fp32", "fp16_weights", "int8_weights", "fp16_encoder")
# Quantization rollouts use their own env stream, disjoint from training (seed), final eval (+1e6) and the
# policy verify replay (+2e6).
QUANT_ENV_SEED_OFFSET = 3_000_000
LATENCY_WARMUP = 200

LATENCY_NOTE = (
    "Tile-coding and discrete encoders are pure Python here (hash-table / index arithmetic loops), and the "
    "SSP encoder is NumPy; wall times are therefore implementation- and machine-specific. The analytic "
    "operation counts are the implementation-independent comparison."
)


# ---------------------------------------------------------------------------------------------
# Pure helpers (unit-tested without any trained policy)
# ---------------------------------------------------------------------------------------------
def quantize_int8_per_row(w, out_axis: int = 0):
    """Symmetric int8 quantization with one scale per OUTPUT unit.

    `out_axis` is the axis that indexes output units: 0 for nn.Linear weights (out, in), 1 for the sparse
    first layer's EmbeddingBag table (dim, width) whose output units are the columns. The scale is
    max|w| / 127 over the other axis, so every element is within scale/2 of its dequantized value.
    Returns (q int8, scale float32 with keepdims shape).
    """
    w = np.asarray(w, dtype=np.float32)
    assert w.ndim == 2 and out_axis in (0, 1), (w.shape, out_axis)
    red_axis = 1 - out_axis
    amax = np.abs(w).max(axis=red_axis, keepdims=True)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q, scale):
    return q.astype(np.float32) * scale


def fp16_roundtrip(x):
    """Cast to float16 and back to the original dtype (simulated fp16 storage)."""
    x = np.asarray(x)
    return x.astype(np.float16).astype(x.dtype)


def classify_tensor(name: str, tensor):
    """('matrix', out_axis) for a weight matrix that int8 quantization applies to, else ('bias', None).
    The sparse first layer table is `emb.weight` with shape (dim, width): output units are columns."""
    ndim = tensor.dim() if hasattr(tensor, "dim") else np.ndim(tensor)
    if ndim == 2:
        return "matrix", (1 if name == "emb.weight" else 0)
    return "bias", None


def quantize_state_dict(state_dict: dict, mode: str) -> dict:
    """New state_dict (fresh tensors, input untouched) with weights simulated at `mode`.

    fp16: every tensor round-tripped through float16. int8: every weight MATRIX quantized per output row and
    dequantized back to fp32; biases stay fp32 (they are tiny and added after the integer accumulate).
    """
    import torch
    assert mode in ("fp16", "int8"), mode
    out = {}
    for name, t in state_dict.items():
        arr = t.detach().cpu().numpy()
        if mode == "fp16":
            new = fp16_roundtrip(arr.astype(np.float32))
        else:
            kind, out_axis = classify_tensor(name, t)
            if kind == "matrix":
                q, s = quantize_int8_per_row(arr, out_axis)
                new = dequantize_int8(q, s)
            else:
                new = arr.astype(np.float32)
        out[name] = torch.from_numpy(np.ascontiguousarray(new, dtype=np.float32))
    return out


def network_size(net) -> dict:
    """Bytes of every REAL stored tensor of the QNetwork at fp32 / fp16 / int8.

    For tile/discrete the first layer is the (dim x width) embedding table `emb.weight`, counted as stored.
    int8: weight matrices are 1 byte/element plus one fp32 scale per output unit; biases stay fp32.
    """
    tensors, total = {}, {p: 0 for p in PRECISIONS}
    n_params = 0
    for name, t in net.state_dict().items():
        n = int(t.numel())
        kind, out_axis = classify_tensor(name, t)
        if kind == "matrix":
            n_out = int(t.shape[out_axis])
            int8_bytes = n + 4 * n_out
        else:
            int8_bytes = 4 * n
        entry = {"shape": list(t.shape), "n": n, "kind": kind,
                 "fp32": 4 * n, "fp16": 2 * n, "int8": int8_bytes}
        tensors[name] = entry
        n_params += n
        for p in PRECISIONS:
            total[p] += entry[p]
    return {"tensors": tensors, "n_params": n_params, **total}


def network_macs(net, n_active: int = None) -> dict:
    """Multiply-accumulates for ONE forward pass, from the actual layers.

    Dense net: sum of in*out over Linear layers. Sparse net: the EmbeddingBag first layer only sums
    `n_active` table rows of length width (n_active*width accumulates, counted as MAC-equivalents), then the
    remaining Linear layers as usual.
    """
    first = 0
    if getattr(net, "sparse", False):
        assert n_active is not None, "sparse net needs n_active"
        first = int(n_active) * int(net.emb.embedding_dim)
    per_layer = [int(l.in_features) * int(l.out_features) for l in net.layers]
    return {"first_layer_sparse_accumulates": first, "linear_layer_macs": per_layer,
            "total": first + sum(per_layer)}


def latency_stats(ns_values) -> dict:
    """median / p95 / p99 (and mean, n) of a list of nanosecond timings, reported in microseconds."""
    a = np.asarray(ns_values, dtype=np.float64) / 1000.0
    return {"median_us": float(np.median(a)), "p95_us": float(np.percentile(a, 95)),
            "p99_us": float(np.percentile(a, 99)), "mean_us": float(a.mean()), "n": int(a.size)}


def greedy_returns(make_policy, env_seed: int, n_episodes: int, max_steps: int = 500) -> list:
    """Greedy rollouts on one CartPole-v1 env: reset(seed=env_seed) on the FIRST episode only (the stream
    then continues), a fresh policy object per episode (`make_policy()`, normally policy.clone()).
    CartPole draws from the env rng only in reset(), so the initial states are identical for every variant
    given the same env_seed: that is what makes the quantization comparison paired."""
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    rets = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=env_seed) if ep == 0 else env.reset()
            pol = make_policy()
            pol.reset()
            total = 0.0
            for _ in range(max_steps):
                obs, r, term, trunc, _ = env.step(int(pol.act(obs)))
                total += float(r)
                if term or trunc:
                    break
            rets.append(total)
    finally:
        env.close()
    return rets


# ---------------------------------------------------------------------------------------------
# Policy access
# ---------------------------------------------------------------------------------------------
def _repo_imports():
    """rl.* imports are lazy so the pure helpers above work without a policy pass."""
    from rl.policy_io import load_policy
    return load_policy


def policy_local(study: str, seed: int, local_root, no_r2: bool):
    """Local path of the saved policy (pulled from R2 unless --no-r2); None if it is not available."""
    key = layout.policy_key(study, seed)
    path = layout.local_path(key, local_root)
    if os.path.exists(path):
        return path
    if no_r2:
        return None
    st = parallel_jobs.pull_file(key, path)
    return path if st == "ok" and os.path.exists(path) else None


def _encoder_of(policy):
    """The sspspace SSPEncoder held by an SSPRep (attribute `_encoder`, with `.phase_matrix`)."""
    return policy.rep._encoder


def n_active_of(policy) -> int:
    """Number of active input features: tilings for tile coding, 1 for one-hot, dim for dense SSP."""
    n = getattr(policy.rep, "n_active", None)
    return int(n) if n else int(policy.dim)


# ---------------------------------------------------------------------------------------------
# Section: size
# ---------------------------------------------------------------------------------------------
def _float_item(n: int, note: str = "") -> dict:
    d = {"n": int(n), **{p: int(n) * BYTES_PER_ELEM[p] for p in PRECISIONS}}
    if note:
        d["note"] = note
    return d


def _int_item(n_bytes: int, note: str = "") -> dict:
    # Integer tables do not shrink with weight precision.
    d = {p: int(n_bytes) for p in PRECISIONS}
    if note:
        d["note"] = note
    return d


def encoder_size(policy) -> dict:
    kind = policy.rep_kind
    items, extra = {}, {}
    if kind == "ssp":
        enc = _encoder_of(policy)
        pm = np.asarray(enc.phase_matrix)
        n_rows, n_cols = pm.shape
        item = _float_item(pm.size)
        # int8 for the phase matrix would need one fp32 scale per row
        item["int8"] = int(pm.size) + 4 * int(n_rows)
        item["shape"] = [int(n_rows), int(n_cols)]
        item["stored_dtype"] = str(pm.dtype)
        item["bytes_as_stored"] = int(pm.nbytes)
        item["note"] = ("phase matrix is conjugate-symmetric, so only (ssp_dim//2 + 1) rows are independent; "
                        "counted here in full as stored")
        items["phase_matrix"] = item
        items["length_scale"] = _float_item(int(np.size(enc.length_scale)))
    elif kind == "tile":
        rep = policy.rep
        d = rep.iht.dictionary
        n_entries = len(d)
        key_len = len(next(iter(d))) if n_entries else 1 + len(rep.state_indices)
        iht_size = int(rep.iht.size)
        items["iht_dictionary"] = _int_item(
            n_entries * (key_len + 1) * 4,
            f"{n_entries} allocated entries x ({key_len} int32 key coords + int32 index)")
        items["iht_dictionary"]["n_entries"] = n_entries
        n_dim = len(rep.state_indices)
        items["bounds_and_grid"] = _float_item(3 * n_dim, "lower, upper, tiles_per_dim")
        extra["iht_n_entries"] = n_entries
        extra["iht_size"] = iht_size
        extra["iht_dense_alternative_bytes"] = iht_size * (key_len + 1) * 4
        extra["iht_dense_alternative_assumption"] = (
            f"a preallocated open-addressing hash table with iht_size={iht_size} slots at load factor 1.0, each "
            f"slot holding {key_len} int32 key coords + an int32 index; NOT included in totals")
    elif kind == "discrete":
        rep = policy.rep
        ranges = [int(r) for r in np.asarray(rep.ranges).ravel()]
        items["bin_edges"] = _float_item(sum(r + 1 for r in ranges),
                                         "explicit equal-width edges, sum(n_bins+1) per dimension")
        items["bin_counts"] = _int_item(4 * len(ranges), "int32 per dimension")
        extra["ranges"] = ranges
    else:
        raise ValueError(f"unknown rep_kind {kind!r}")
    total = {p: sum(it[p] for it in items.values()) for p in PRECISIONS}
    return {"items": items, **total, **extra}


def size_report(policy) -> dict:
    net = network_size(policy.net)
    enc = encoder_size(policy)
    total = {p: net[p] + enc[p] for p in PRECISIONS}
    out = {"rep_kind": policy.rep_kind, "hidden_dims": policy.hidden_dims, "sparse": bool(policy.sparse),
           "dim": int(policy.dim), "n_params": int(net["n_params"]), "n_active": n_active_of(policy),
           "network": net, "encoder": enc, "total": total}
    out["total_note"] = ("total = network + encoder constants as STORED (for tile coding: the sparse IHT "
                         "dictionary of n_entries allocated tiles)")
    if "iht_dense_alternative_bytes" in enc:
        # Reported separately, never mixed into 'total': what the table would cost preallocated at iht_size.
        out["total_with_dense_iht_alternative"] = {
            p: total[p] - enc["items"]["iht_dictionary"][p] + enc["iht_dense_alternative_bytes"]
            for p in PRECISIONS}
    return out


def run_size(args, study, load_policy) -> dict:
    ref_idx = args.latency_seed_idx
    path = policy_local(study, args.seed_offset + ref_idx, args.local_root, args.no_r2)
    if path is None:
        raise FileNotFoundError(f"size: policy for seed_idx {ref_idx} not available")
    pol = load_policy(path)
    rep = size_report(pol)
    out = {"reference_seed_idx": ref_idx, "reference_seed": args.seed_offset + ref_idx, **rep}
    if pol.rep_kind == "tile":
        # The IHT fills during training and eval, so its entry count differs per seed: report the spread.
        counts = {}
        for i in range(args.n_seeds):
            p = policy_local(study, args.seed_offset + i, args.local_root, args.no_r2)
            if p is not None:
                counts[i] = len(load_policy(p).rep.iht.dictionary)
        vals = np.array(list(counts.values()), dtype=float)
        out["iht_n_entries_per_seed"] = {str(k): v for k, v in counts.items()}
        if vals.size:
            out["iht_n_entries_summary"] = {"mean": float(vals.mean()), "min": float(vals.min()),
                                            "max": float(vals.max()), "n_seeds": int(vals.size)}
    return out


# ---------------------------------------------------------------------------------------------
# Section: quantization
# ---------------------------------------------------------------------------------------------
def make_variant(policy, variant: str):
    """A policy object for `variant` that never mutates `policy` or anything it shares.

    clone() shares the (read-only) network and SSP phase matrix with the original, so the network is
    deep-copied BEFORE its weights are replaced, and the rep (+ its encoder) is shallow-copied before the
    phase matrix attribute is swapped.
    """
    pol = policy.clone()
    if variant == "fp32":
        return pol
    if variant in ("fp16_weights", "int8_weights"):
        mode = "fp16" if variant == "fp16_weights" else "int8"
        net = copy.deepcopy(policy.net)
        net.load_state_dict(quantize_state_dict(policy.net.state_dict(), mode))
        pol.net = net
        return pol
    if variant == "fp16_encoder":
        if policy.rep_kind != "ssp":
            raise ValueError("fp16_encoder applies to SSP only")
        rep = copy.copy(pol.rep)
        enc = copy.copy(_encoder_of(pol))
        enc.phase_matrix = fp16_roundtrip(np.asarray(enc.phase_matrix))
        rep._encoder = enc
        pol.rep = rep
        return pol
    raise ValueError(variant)


def _episode_factory(base, variant: str):
    """Per-episode policy factory (one clone per episode). If clone() would silently drop the fp16 encoder
    override (e.g. by rebuilding the encoder from its spec), fall back to the variant object itself: the SSP
    encoder and the net have no mutable state, so sharing across episodes is safe."""
    if variant == "fp16_encoder":
        probe = base.clone()
        if not np.array_equal(np.asarray(_encoder_of(probe).phase_matrix),
                              np.asarray(_encoder_of(base).phase_matrix)):
            return lambda: base
    return base.clone


def quant_worker(job: dict) -> dict:
    """All variants for ONE saved policy (runs in a spawned process). Never raises."""
    meta = {k: job[k] for k in ("seed_idx", "seed")}
    try:
        torch_threads()
        load_policy = _repo_imports()
        policy = load_policy(job["path"])
        variants = ["fp32", "fp16_weights", "int8_weights"]
        if policy.rep_kind == "ssp":
            variants.append("fp16_encoder")
        env_seed = job["seed"] + QUANT_ENV_SEED_OFFSET
        states = None
        if job.get("probe_path") and os.path.exists(job["probe_path"]):
            states = np.load(job["probe_path"])["states"]
        ref_q = policy.clone().q_values_batch(states) if states is not None else None
        res = {}
        for v in variants:
            base = make_variant(policy, v)
            if base.net is policy.net and v in ("fp16_weights", "int8_weights"):
                raise RuntimeError("variant network aliases the shared network")
            rets = greedy_returns(_episode_factory(base, v), env_seed, job["n_episodes"])
            entry = {"mean_return": float(np.mean(rets)), "returns": rets}
            if ref_q is not None and v != "fp32":
                # audit that the override really changed the function (0.0 would mean it was ignored)
                entry["q_max_abs_diff_probe"] = float(np.max(np.abs(base.clone().q_values_batch(states) - ref_q)))
            res[v] = entry
        return {**meta, "env_seed": env_seed, "variants": res}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {**meta, "error": repr(e) + " | " + traceback.format_exc()[-400:]}


def torch_threads():
    parallel_jobs.init_worker()


def aggregate_quant(per_seed: list) -> dict:
    """Across seeds: mean, std (sample) of the per-seed mean return, retained = mean(variant)/mean(fp32),
    and the mean paired (same seed, same episodes) difference to fp32."""
    variants = [v for v in QUANT_VARIANTS if any(v in r["variants"] for r in per_seed)]
    ref = {r["seed_idx"]: r["variants"]["fp32"]["mean_return"] for r in per_seed}
    out = {}
    for v in variants:
        rows = [r for r in per_seed if v in r["variants"]]
        means = np.array([r["variants"][v]["mean_return"] for r in rows])
        refs = np.array([ref[r["seed_idx"]] for r in rows])
        ref_mean = float(refs.mean())
        out[v] = {"n_seeds": len(rows), "mean": float(means.mean()),
                  "std": float(means.std(ddof=1)) if len(rows) > 1 else 0.0,
                  "retained": float(means.mean() / ref_mean) if ref_mean > 0 else None,
                  "mean_paired_diff_vs_fp32": float((means - refs).mean()),
                  "per_seed_mean": [float(x) for x in means]}
    return out


def run_quant(args, study, probe_path) -> dict:
    n = min(args.quant_seeds if args.quant_seeds is not None else args.n_seeds, args.n_seeds)
    jobs, missing = [], []
    for i in range(n):
        seed = args.seed_offset + i
        path = policy_local(study, seed, args.local_root, args.no_r2)
        if path is None:
            missing.append(i)
            continue
        jobs.append({"seed_idx": i, "seed": seed, "path": path, "n_episodes": args.quant_episodes,
                     "probe_path": probe_path})
    print(f"[quant] {len(jobs)} policies ({len(missing)} missing), {args.quant_episodes} episodes x "
          f"variants, {args.workers} workers", flush=True)
    records, errors = [], []
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=ctx,
                                initializer=parallel_jobs.init_worker) as ex:
        futs = [ex.submit(quant_worker, j) for j in jobs]
        for k, fut in enumerate(cf.as_completed(futs), 1):
            rec = fut.result()
            if "error" in rec:
                errors.append(rec)
                print(f"[quant] [{k}/{len(jobs)}] ERROR seed {rec['seed']}: {rec['error'][:160]}", flush=True)
            else:
                records.append(rec)
                bits = " ".join(f"{v}={r['mean_return']:.1f}" for v, r in rec["variants"].items())
                print(f"[quant] [{k}/{len(jobs)}] seed {rec['seed']} {bits}", flush=True)
    records.sort(key=lambda r: r["seed_idx"])
    out = {"n_seeds": len(records), "n_episodes": args.quant_episodes,
           "env_seed_rule": f"seed + {QUANT_ENV_SEED_OFFSET}; reset(seed=..) on episode 0 only; "
                            "identical for every variant (paired); one policy.clone() per episode",
           "int8_scheme": "symmetric per-output-row weight matrices (embedding table: per output column), "
                          "biases fp32, dequantized back to fp32 for simulation",
           "missing_seed_idx": missing, "errors": errors}
    if records:
        out["variants"] = aggregate_quant(records)
        out["per_seed"] = [{"seed_idx": r["seed_idx"], "seed": r["seed"], "variants": r["variants"]}
                           for r in records]
    return out


# ---------------------------------------------------------------------------------------------
# Section: latency
# ---------------------------------------------------------------------------------------------
def time_calls(policy, states, n_calls: int, warmup: int = LATENCY_WARMUP) -> dict:
    """Time encode / q_from_phi / act separately on a CLONE (tile coding allocates tiles while encoding)."""
    pol = policy.clone()
    n = len(states)
    pc = time.perf_counter_ns
    for i in range(warmup):
        pol.act(states[i % n])
    t_enc = np.empty(n_calls, dtype=np.int64)
    t_q = np.empty(n_calls, dtype=np.int64)
    t_act = np.empty(n_calls, dtype=np.int64)
    gc.collect()
    gc.disable()  # a collection inside a 20 us call would show up as a fake p99
    try:
        for i in range(n_calls):
            obs = states[i % n]
            a = pc()
            phi = pol.encode(obs)
            b = pc()
            pol.q_from_phi(phi)
            c = pc()
            pol.act(obs)
            d = pc()
            t_enc[i], t_q[i], t_act[i] = b - a, c - b, d - c
    finally:
        gc.enable()
    return {"encode": latency_stats(t_enc), "q_from_phi": latency_stats(t_q), "act": latency_stats(t_act)}


def operation_counts(policy) -> dict:
    kind = policy.rep_kind
    n_act = n_active_of(policy)
    macs = network_macs(policy.net, n_act if getattr(policy.net, "sparse", False) else None)
    if kind == "ssp":
        enc = _encoder_of(policy)
        ssp_dim, obs_dim = int(enc.ssp_dim), int(enc.domain_dim)
        enc_ops = {"phase_matmul_macs": (ssp_dim // 2) * obs_dim,
                   "exp_ifft_ops_estimate": ssp_dim,
                   "note": "matmul over the independent half of the conjugate-symmetric phase matrix; "
                           "exp + ifft estimated as ssp_dim operations (ifft is really O(ssp_dim log ssp_dim))"}
        enc_total = enc_ops["phase_matmul_macs"] + enc_ops["exp_ifft_ops_estimate"]
    elif kind == "tile":
        rep = policy.rep
        enc_ops = {"hash_lookups": int(rep.num_tilings),
                   "coordinate_ops": int(rep.num_tilings) * len(rep.state_indices),
                   "note": "one dictionary lookup per tiling (+ per-dimension integer coordinate arithmetic)"}
        enc_total = enc_ops["hash_lookups"]
    elif kind == "discrete":
        d = len(np.asarray(policy.rep.ranges).ravel())
        enc_ops = {"bin_index_ops": d, "table_lookups": 1,
                   "note": "one bin index per dimension, combined into a single embedding row index"}
        enc_total = d
    else:
        raise ValueError(kind)
    return {"encoder": enc_ops, "encoder_total_ops": int(enc_total), "network": macs,
            "network_macs": macs["total"], "total_ops": int(enc_total) + macs["total"],
            "n_active": n_act}


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_latency(args, study, load_policy, probe_path) -> dict:
    import torch
    torch.set_num_threads(1)
    path = policy_local(study, args.seed_offset + args.latency_seed_idx, args.local_root, args.no_r2)
    if path is None:
        raise FileNotFoundError(f"latency: policy for seed_idx {args.latency_seed_idx} not available")
    policy = load_policy(path)
    states = np.load(probe_path)["states"]
    timings = time_calls(policy, states, args.latency_calls)
    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    return {"seed_idx": args.latency_seed_idx, "seed": args.seed_offset + args.latency_seed_idx,
            "n_calls": args.latency_calls, "warmup_calls": LATENCY_WARMUP, "n_probe_states": int(len(states)),
            "timings": timings, "operation_counts": operation_counts(policy), "note": LATENCY_NOTE,
            "cpu_model": cpu_model(), "n_cpu": os.cpu_count(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "torch_num_threads": torch.get_num_threads(),
            "loadavg_at_end": load}


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell", required=True, help="rep/centering/arch, e.g. ssp/none/deep")
    ap.add_argument("--local-root", default=None, help="default <repo>/outputs/eval_v3")
    ap.add_argument("--no-r2", action="store_true", help="never touch R2; policies must exist locally")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=layout.POLICY_N_SEEDS)
    ap.add_argument("--seed-offset", type=int, default=layout.POLICY_SEED_OFFSET)
    ap.add_argument("--sections", default="size,quant,latency", help="comma list of size,quant,latency")
    ap.add_argument("--quant-seeds", type=int, default=None, help="default: all --n-seeds")
    ap.add_argument("--quant-episodes", type=int, default=20)
    ap.add_argument("--latency-calls", type=int, default=2000)
    ap.add_argument("--latency-seed-idx", type=int, default=0)
    args = ap.parse_args(argv)
    args.sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    bad = [s for s in args.sections if s not in ("size", "quant", "latency")]
    if bad:
        ap.error(f"unknown sections {bad}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.no_r2:
        # Hard guarantee for --no-r2, including inside helpers that fetch/push on their own
        # (e.g. make_probe_states.ensure_probe_file): no credentials -> every R2 call is a no-op.
        parallel_jobs.r2_client = lambda: (None, None)
    study = layout.study_name(args.cell)
    key = layout.deployment_key(study)
    out_path = layout.local_path(key, args.local_root)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    load_policy = _repo_imports()

    # Merge into an existing file so one section can be re-run alone (e.g. latency on a quiet pod).
    if not os.path.exists(out_path) and not args.no_r2:
        parallel_jobs.pull_file(key, out_path)
    result = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                result = json.load(f)
        except (OSError, ValueError):
            result = {}
    result.update({"study": study, "cell": args.cell, "n_seeds": args.n_seeds, "seed_offset": args.seed_offset})

    def save():
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=1)
        os.replace(tmp, out_path)
        if not args.no_r2:
            parallel_jobs.push_file(key, out_path)

    probe_path = None
    if "quant" in args.sections or "latency" in args.sections:
        from make_probe_states import ensure_probe_file
        probe_path = ensure_probe_file(args.local_root, push=False)

    # Order matters: latency runs BEFORE the worker pool exists so nothing else competes for the CPU.
    if "size" in args.sections:
        t0 = time.time()
        result["size"] = run_size(args, study, load_policy)
        print(f"[size] done in {time.time() - t0:.1f}s: total fp32/fp16/int8 bytes = "
              f"{result['size']['total']['fp32']}/{result['size']['total']['fp16']}/"
              f"{result['size']['total']['int8']}", flush=True)
        save()
    if "latency" in args.sections:
        result["latency"] = run_latency(args, study, load_policy, probe_path)
        t = result["latency"]["timings"]["act"]
        print(f"[latency] act median {t['median_us']:.1f} us p95 {t['p95_us']:.1f} p99 {t['p99_us']:.1f}",
              flush=True)
        save()
    if "quant" in args.sections:
        result["quantization"] = run_quant(args, study, probe_path)
        for v, s in result["quantization"].get("variants", {}).items():
            print(f"[quant] {v}: mean {s['mean']:.1f} std {s['std']:.1f} retained {s['retained']}", flush=True)
        save()
    try:
        result["pod"] = parallel_jobs.pod_info(args.workers)
    except Exception:  # noqa: BLE001
        pass
    save()
    print(f"[deployment] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
