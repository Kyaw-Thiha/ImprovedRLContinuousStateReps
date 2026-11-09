"""One training run: continuing Go1/G1 PPO with a centering arm, periodic deterministic eval, JSONL logs.

Usage (local smoke):
    python -m mjlab_centering.train --job-id smoke --robot go1 --num-envs 32 --iters 5 --eval-envs 8 \
        --center value --out results/smoke

The run writes, under --out:
    iter.jsonl    one row per iteration (see docs/mjlab_centering/00-plan.md "Logs and evals")
    eval.jsonl    one row per periodic eval
    final.json    deployment eval (nominal 5k steps + command extrapolation + 1-step action delay)
    model_final.pt, config.json
and, when --r2 is set, pushes every artefact to R2 as it is produced.
"""

from __future__ import annotations

import argparse
import json
import threading
import math
import os
import time
from dataclasses import asdict, dataclass, field

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from mjlab_centering.algo import CenteredPPO, init_critic_output_bias
from mjlab_centering.env_cfg import apply_shock, build_agent_cfg, build_env_cfg
from mjlab_centering.wrapper import ContinuingVecEnv


@dataclass
class RunSpec:
  job_id: str = "smoke"
  block: str = "smoke"
  robot: str = "go1"
  seed: int = 2000
  center: str = "none"
  gamma: float = 0.99
  offset: float = 0.0
  penalty: float = 10.0
  iters: int = 500
  learning_rate: float | None = None
  eta: float = 0.01
  beta: float = 0.01
  value_clip: bool | None = None
  ret_norm: str = "none"
  reward_scale: str = "none"
  r_bar_init: str = "zero"
  log_rew_std: bool = False  # d155: log the per-step reward std (task and raw) every iteration; pure logging
  vn_clamp: float | None = 5.0  # rl_games' +-5 clamp on normalised values; 0/None is the diagnostic no-clamp arm
  rs_clip: float = 10.0  # SB3 VecNormalize's reward clip after scaling
  # round-9 arms (REPORT_009)
  bias_lr: float | None = None  # A1: critic output bias in its own group at this fixed lr (out of the KL schedule)
  warmup: bool = False  # A2: freeze the actor until the critic's level reaches warmup_frac x c/(1-gamma)
  warmup_frac: float = 0.95
  warmup_max_iters: int = 1000
  actor_lr_scale: float | None = None  # A3: actor lr x scale for the first actor_lr_iters iterations
  actor_lr_iters: int = 0
  # round 9b: the truncated-GAE leak arms
  lam: float | None = None  # GAE lambda; None keeps the runner's default (0.95)
  adv_mode: str = "global"  # global | pos_norm | pos_center
  log_adv_pos: bool = False  # log the per-rollout-position advantage mean/std and position R^2
  shock_kind: str = "none"
  shock_at: int | None = None
  shock_factor: float = 5.0
  p_reset: float = 0.0
  num_envs: int = 4096
  eval_envs: int = 256
  eval_every: int = 25
  eval_steps: int = 300
  final_steps: int = 5000
  final_shift_steps: int = 3000
  save_at: int | None = None  # write ckpt_it<N>.pt (full state) after iteration N-1, for shock branching
  resume: str | None = None  # continue from such a checkpoint; env is freshly reset (same for every branch)
  save_every: int = 0  # periodic checkpoint every N iterations, mirrored to R2 as ckpt_latest.pt so a preempted pod resumes
  episodic: bool = False  # reference path: mjlab's own episodic env for TRAINING (eval stays the continuing env)
  falls_reset: bool = True  # decision 81 default: a fall resets the robot in place (env-level), PPO never sees it
  time_outs: bool = True  # decision 77 default: episodes still end on time (truncated + bootstrapped)
  obs_v130: bool = False  # reproduce mjlab 1.3.0's observation layout (unbiased joint_pos for actor and critic)
  upload_ckpt: bool = False  # 4.7 MB per run: only the headline cells need the policy on R2
  num_steps_per_env: int = 24
  priority: int = 9
  notes: str = ""
  extra: dict = field(default_factory=dict)


class _Watchdog:
  """Exit the process if an iteration stops making progress.

  A stalled CUDA call or a network hang leaves a training job idle forever, and on RunPod an idle pod bills. The
  watchdog writes a STALL marker and releases the claim, so a restarted pod retakes the work immediately.
  """

  def __init__(self, limit_s: float = 600.0, on_stall=None) -> None:
    self.limit_s = limit_s
    self.on_stall = on_stall
    self.last = time.time()
    self._stop = threading.Event()
    self._th = threading.Thread(target=self._run, daemon=True)

  def beat(self) -> None:
    self.last = time.time()

  def _run(self) -> None:
    while not self._stop.wait(15):
      if time.time() - self.last > self.limit_s:
        print(f"[watchdog] no progress for {self.limit_s:.0f}s: aborting", flush=True)
        if self.on_stall:
          try:
            self.on_stall()
          except Exception:
            pass
        os._exit(3)

  def start(self) -> "_Watchdog":
    self._th.start()
    return self

  def stop(self) -> None:
    self._stop.set()


def _mean(x: torch.Tensor) -> float:
  return float(x.mean())


FALL_CHECK_DEG = 70.0  # mjlab's fell_over threshold; used only to keep measuring falls when they are not terminal


def build_spec(d: dict) -> RunSpec:
  known = {f for f in RunSpec.__dataclass_fields__}
  kw = {k: v for k, v in d.items() if k in known}
  return RunSpec(**kw)


def make_env(spec: RunSpec, num_envs: int, seed_off: int = 0, train: bool = True):
  episodic = bool(spec.episodic) and train
  if episodic:
    assert spec.center == "none" and spec.offset == 0.0 and spec.shock_kind == "none", "episodic arm is the plain reference"
  ex = spec.extra or {}
  cfg = build_env_cfg(
    robot=spec.robot, num_envs=num_envs, continuing=not episodic, encoder=ex.get("encoder", "raw"),
    ssp_length_scale=float(ex.get("ssp_length_scale", 0.3)), ssp_n_rotates=int(ex.get("ssp_n_rotates", 8)),
    ssp_n_scales=int(ex.get("ssp_n_scales", 8)), ssp_rff_seed=int(ex.get("ssp_rff_seed", 0)), falls_reset=bool(spec.falls_reset),
    time_outs=bool(spec.time_outs), obs_v130=bool(spec.obs_v130),
  )
  cfg.seed = int(spec.seed) + seed_off
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
  if episodic:
    return env, RslRlVecEnvWrapper(env)
  shock_at = spec.shock_at if spec.shock_kind != "none" else None
  # Offset shock size: `extra["shock_delta"]` is an absolute per-step addition (needed when the pre-shock offset is 0,
  # where the multiplicative form offset * (factor - 1) is identically 0); otherwise the legacy multiplicative form.
  if spec.shock_kind != "offset":
    delta = 0.0
  elif "shock_delta" in ex:
    delta = float(ex["shock_delta"])
  else:
    delta = spec.offset * (spec.shock_factor - 1.0)
  vec = ContinuingVecEnv(
    env,
    penalty=spec.penalty,
    time_outs=bool(spec.time_outs),
    fall_check_deg=None if spec.falls_reset else FALL_CHECK_DEG,
    offset=spec.offset,
    shock_at=shock_at,
    shock_delta=delta,
    p_reset=spec.p_reset,
  )
  return env, vec


class JsonlSink:
  """Append-only JSONL that mirrors itself to R2 on a background thread.

  Pushing inline blocked the training loop for minutes at a time on this pod's egress (measured ~200 s per upload
  with retries), which dominated the wall clock. The main loop now only marks the file dirty; a daemon thread
  uploads the latest bytes, skips a cycle if an upload is still in flight, and swallows errors after recording them.
  """

  def __init__(self, path: str, r2_key: str | None = None, push_every: int = 25):
    self.path = path
    self.r2_key = r2_key
    self.push_every = push_every
    self.n = 0
    self.last_push_s = 0.0
    self._dirty = threading.Event()
    self._stop = threading.Event()
    self._thread = None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    self._fh = open(path, "a", buffering=1)
    if r2_key:
      self._thread = threading.Thread(target=self._loop, daemon=True)
      self._thread.start()

  def write(self, row: dict) -> None:
    self._fh.write(json.dumps(row, default=float) + "\n")
    self.n += 1
    if self.r2_key and self.n % self.push_every == 0:
      self._dirty.set()

  def _loop(self) -> None:
    while not self._stop.wait(5):
      if not self._dirty.is_set():
        continue
      self._dirty.clear()
      self.push()

  def push(self) -> None:
    """Upload the current file contents once (bounded by the client's timeouts)."""
    if not self.r2_key:
      return
    t0 = time.time()
    try:
      from mjlab_centering import r2

      r2.upload_file(self.path, self.r2_key)
    except Exception as e:  # never let a sync error kill a run
      print(f"[warn] r2 push failed for {self.r2_key}: {e!r}", flush=True)
    self.last_push_s = time.time() - t0

  def flush(self) -> None:
    if self.r2_key and self._dirty.is_set():
      self._dirty.clear()
      self.push()

  def close(self) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=30)
    self.flush()
    self._fh.close()


def evaluate(vec, policy, steps: int, delay: int = 0) -> dict:
  """Deterministic rollout of the current policy.

  All accumulation stays on the GPU and is converted once at the end: a per-step `.mean()` costs a host sync, and
  32 evals x 500 steps of syncs dominated the run time (measured: eval was ~50x the simulation cost).
  """
  obs = vec.get_observations().to(policy.device if hasattr(policy, "device") else "cuda")
  task_sum = torch.zeros((), device=vec.device)
  seen_sum = torch.zeros((), device=vec.device)
  falls = torch.zeros((), device=vec.device)
  term_sum = None
  n = 0.0
  prev = None
  with torch.no_grad():
    for _ in range(steps):
      act = policy(obs, stochastic_output=False)
      if delay and prev is not None:
        act = prev
      prev = act
      obs, rew, dones, extras = vec.step(act)
      cc = extras["cc"]
      task_sum += cc["task"].sum()
      seen_sum += cc["seen"].sum()
      falls += cc["fell"].sum()
      term_sum = cc["terms"] if term_sum is None else term_sum + cc["terms"]
      n += vec.num_envs
  task, seen, fell = float(task_sum), float(seen_sum), float(falls)
  out = {
    "J_task": task / n,
    "J_seen": seen / n,
    "fall_rate_per_1k": 1000.0 * fell / n,
    "steps": steps,
    "delay": delay,
  }
  if term_sum is not None:
    # `cc["terms"]` is already the mean over envs of weight x f (not dt-scaled), so average over steps only.
    # (Dividing by steps x envs, as before, under-reported every term by a factor num_envs: decision 43.)
    for name, v in zip(cc["term_names"], term_sum.tolist()):
      out[f"term/{name}"] = v / steps
    # tracking quality: mean f in [0, 1]. `term_rewards` is weight x f x dt (reward_manager.term_rewards is "scaled
    # like reward_buf"), so f = term / (weight x dt). Verified on a standing rollout: upright -> 0.95 (decision 45).
    rm = vec.unwrapped.reward_manager
    dt = float(vec.unwrapped.step_dt)
    for short, name in (("track_lin_frac", "track_linear_velocity"), ("track_ang_frac", "track_angular_velocity")):
      if name in rm.active_terms and f"term/{name}" in out:
        w = abs(float(rm._term_cfgs[rm.active_terms.index(name)].weight))
        out[short] = out[f"term/{name}"] / max(w * dt, 1e-12)
  return out


def run(spec: RunSpec, out_dir: str, r2_campaign: str | None = None, quiet: bool = False) -> dict:
  configure_torch_backends()
  torch.manual_seed(spec.seed)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  os.makedirs(out_dir, exist_ok=True)
  # Headless: the pod sets MUJOCO_GL=disable (no GL context is needed; mjlab-mujoco only renders for video).

  env, vec = make_env(spec, spec.num_envs)
  agent_cfg = build_agent_cfg(
    robot=spec.robot,
    max_iterations=spec.iters,
    seed=spec.seed,
    center=spec.center,
    eta=spec.eta,
    beta=spec.beta,
    value_clip=spec.value_clip,
    ret_norm=spec.ret_norm,
    reward_scale=spec.reward_scale,
    r_bar_init=spec.r_bar_init,
    learning_rate=spec.learning_rate,
    num_steps_per_env=spec.num_steps_per_env,
    log_rew_std=spec.log_rew_std,
    vn_clamp=spec.vn_clamp,
    rs_clip=spec.rs_clip,
    bias_lr=spec.bias_lr,
    warmup=spec.warmup,
    warmup_frac=spec.warmup_frac,
    warmup_max_iters=spec.warmup_max_iters,
    actor_lr_scale=spec.actor_lr_scale,
    actor_lr_iters=spec.actor_lr_iters,
    level_target=(spec.offset / (1.0 - spec.gamma)) if spec.warmup else None,
    lam=spec.lam,
    adv_mode=spec.adv_mode,
    log_adv_pos=spec.log_adv_pos,
  )
  agent_cfg["algorithm"]["gamma"] = float(spec.gamma)

  obs = vec.get_observations()
  alg = CenteredPPO.construct_algorithm(obs, vec, agent_cfg, device)
  cb = (spec.extra or {}).get("critic_bias_init")
  if cb is not None and not spec.resume:  # d114: hand the critic the analytic level of the return before any learning
    b0 = spec.offset / (1.0 - spec.gamma) if cb == "offset" else float(cb)
    init_critic_output_bias(alg.critic, b0)
    if not quiet:
      print(f"[init] critic output bias set to {b0:.3f}", flush=True)
  policy = alg.get_policy()

  # Evaluation env: separate instances so the training env's state stream is untouched.
  eval_vec = eval_env = None
  if spec.eval_envs > 0:
    try:
      eval_env, eval_vec = make_env(spec, spec.eval_envs, seed_off=7_000_000, train=False)
      eval_vec.penalty = spec.penalty  # same task; offset irrelevant for J_task
    except Exception as e:  # pragma: no cover - depends on GPU memory / driver
      if not quiet:
        print(f"[warn] eval env unavailable ({e!r}); falling back to no periodic eval", flush=True)
      eval_vec = None

  prefix = f"{r2_campaign}/runs/{spec.job_id}" if r2_campaign else None
  it_path = os.path.join(out_dir, "iter.jsonl")
  ev_path = os.path.join(out_dir, "eval.jsonl")
  it_sink = JsonlSink(it_path, f"{prefix}/iter.jsonl" if prefix else None, push_every=50)
  ev_sink = JsonlSink(ev_path, f"{prefix}/eval.jsonl" if prefix else None, push_every=1)
  if prefix:
    try:
      from mjlab_centering import r2

      r2.put_json(
        f"{prefix}/config.json",
        {**asdict(spec), "out_dir": out_dir, "gpu": os.environ.get("GPU_NAME", ""),
         "torch": torch.__version__, "cuda": torch.version.cuda},
      )
    except Exception as e:
      print(f"[warn] config push failed: {e!r}", flush=True)

  def _on_stall() -> None:
    if prefix:
      from mjlab_centering import r2 as _r2

      _r2.put_json(f"{prefix}/STALL", {"job_id": spec.job_id, "iters": spec.iters})
      _r2.release(r2_campaign, spec.job_id)

  wd = _Watchdog(on_stall=_on_stall).start()
  evals: list[dict] = []
  t_start = time.time()
  steps_done = 0
  shock_fired = False
  start_it = 0
  if spec.resume:
    ck = torch.load(spec.resume, map_location=device, weights_only=False)
    alg.import_state(ck["alg"])
    start_it = int(ck["iter"])
    torch.set_rng_state(ck["rng_cpu"].cpu())
    if torch.cuda.is_available() and ck.get("rng_cuda") is not None:
      torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
    steps_done = start_it * spec.num_steps_per_env * spec.num_envs
    if not quiet:
      print(f"[resume] from {spec.resume} at iteration {start_it}", flush=True)
  for it in range(start_it, spec.iters):
    vec.iter = it
    if spec.shock_kind == "action_rate" and not shock_fired and spec.shock_at is not None and it >= spec.shock_at:
      w = apply_shock(env, "action_rate", spec.shock_factor)
      shock_fired = True
      if not quiet:
        print(f"[shock] action_rate weight -> {w}", flush=True)
      if prefix:
        from mjlab_centering import r2

        r2.put_json(f"{prefix}/shock.json", {"iter": it, "kind": spec.shock_kind})

    t0 = time.time()
    with torch.inference_mode():
      for _ in range(spec.num_steps_per_env):
        actions = alg.act(obs)
        obs, rew, dones, extras = vec.step(actions)
        obs = obs.to(device)
        alg.process_env_step(obs, rew.to(device), dones.to(device), extras)
    t_collect = time.time() - t0
    t0 = time.time()
    alg.compute_returns(obs)
    loss = alg.update()
    t_learn = time.time() - t0

    steps_done += spec.num_steps_per_env * spec.num_envs
    if spec.save_every and (it + 1) % spec.save_every == 0:
      # d168: a preempted pod used to restart its jobs from zero. A rolling checkpoint makes the re-claim a resume.
      latest = os.path.join(out_dir, "ckpt_latest.pt")
      torch.save({"alg": alg.export_state(), "iter": it + 1, "job_id": spec.job_id, "rng_cpu": torch.get_rng_state(),
                  "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}, latest)
      if prefix:
        from mjlab_centering import r2
        try:
          r2.retry(lambda: r2.upload_file(latest, f"{prefix}/ckpt_latest.pt"), attempts=2)
        except Exception as e:  # noqa: BLE001 - a missed checkpoint must not kill the run
          if not quiet:
            print(f"[warn] ckpt_latest upload failed: {e!r}", flush=True)
    if spec.save_at is not None and it + 1 == spec.save_at:
      ck_path = os.path.join(out_dir, f"ckpt_it{spec.save_at}.pt")
      torch.save({"alg": alg.export_state(), "iter": it + 1, "job_id": spec.job_id, "rng_cpu": torch.get_rng_state(),
                  "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}, ck_path)
      if prefix and spec.upload_ckpt:  # branching checkpoint to R2 (full state: weights, optimiser, lr, r_bar, rng)
        from mjlab_centering import r2

        try:
          r2.retry(lambda: r2.upload_file(ck_path, f"{prefix}/ckpt_it{spec.save_at}.pt"), attempts=4)
        except Exception as e:
          print(f"[warn] ckpt_it{spec.save_at} upload failed: {e!r}", flush=True)
    row = {
      "job_id": spec.job_id,
      "iter": it,
      "ts": time.time(),
      "steps": steps_done,
      "fps": spec.num_steps_per_env * spec.num_envs / max(t_collect + t_learn, 1e-6),
      "t_collect": t_collect,
      "t_learn": t_learn,
      "t_push": it_sink.last_push_s,
      "lr": float(alg.learning_rate),
      "action_std": float(alg.get_policy().output_std.mean().detach()),
      "offset_now": getattr(vec, "offset_now", 0.0),
      **alg.last_diag,
    }
    wd.beat()
    it_sink.write(row)
    if not quiet and (it % 25 == 0 or it == spec.iters - 1):
      print(
        f"[{spec.job_id}] it {it} fps {row['fps']:.0f} J_task {row['r_task_rate']:.4f} "
        f"r_bar {row['r_bar']:.3f} falls/1k {row['fall_rate_per_1k']:.2f} ev {row['ev']:.3f}",
        flush=True,
      )

    if eval_vec is not None and ((it + 1) % spec.eval_every == 0 or it == spec.iters - 1):
      eval_vec.iter = it
      _t = time.time()
      ev = evaluate(eval_vec, policy, spec.eval_steps)
      ev.update({"iter": it, "steps": steps_done, "kind": "nominal", "t_eval": time.time() - _t})
      evals.append(ev)
      ev_sink.write(ev)
      if not quiet:
        print(f"    eval it {it}: J_task {ev['J_task']:.4f} falls/1k {ev['fall_rate_per_1k']:.2f}", flush=True)

  # Finalise in an order that never loses a finished training run: the periodic-eval record and DONE are written and
  # uploaded FIRST, then the (slower, optional) deployment evals are appended. On the pod, the deployment evals were
  # where the end-of-run sequence stalled, and DONE never landed as a result.
  ckpt = os.path.join(out_dir, "model_final.pt")
  torch.save({**alg.export_state(), "iter": spec.iters, "job_id": spec.job_id}, ckpt)
  wd.stop()
  final = {
    "job_id": spec.job_id,
    "iters": spec.iters,
    "steps": steps_done,
    "eval": evals[-5:],
    "n_evals": len(evals),
    "train_wall_s": time.time() - t_start,
  }
  with open(os.path.join(out_dir, "final.json"), "w") as f:
    json.dump(final, f, indent=2, default=float)
  it_sink.close()  # flushes the final rows, then stops the mirror thread
  ev_sink.close()
  if prefix:
    from mjlab_centering import r2

    try:
      _upload_final(r2, prefix, it_path, ev_path, out_dir, ckpt, spec, steps_done, final)
    except Exception as e:
      # The run is complete locally and its DONE may be the only casualty; never let that kill the worker.
      print(f"[warn] final artefact upload failed: {e!r}", flush=True)
  _maybe_deploy(spec, eval_vec, policy, final, out_dir, prefix, t_start)
  env.close()
  if eval_env is not None:
    eval_env.close()
  return final


def _upload_final(r2, prefix, it_path, ev_path, out_dir, ckpt, spec, steps_done, final) -> None:
  """Upload the training result. DONE goes last, and only after every artefact it certifies is on R2."""
  r2.retry(lambda: r2.upload_file(it_path, f"{prefix}/iter.jsonl"))
  r2.retry(lambda: r2.upload_file(ev_path, f"{prefix}/eval.jsonl"))
  r2.retry(lambda: r2.upload_file(os.path.join(out_dir, "final.json"), f"{prefix}/final.json"))
  if spec.upload_ckpt:
    r2.retry(lambda: r2.upload_file(ckpt, f"{prefix}/model_final.pt"))
  r2.retry(lambda: r2.put_json(f"{prefix}/DONE", {"job_id": spec.job_id, "steps": steps_done,
                                                 "train_wall_s": final["train_wall_s"]}))


def _maybe_deploy(spec, eval_vec, policy, final, out_dir, prefix, t_start) -> None:
  """Deployment evals: optional (`--final-steps 0`) and never allowed to invalidate a finished run."""
  if eval_vec is None or spec.final_steps <= 0:
    return
  try:
    eval_vec.iter = spec.iters
    final["nominal"] = evaluate(eval_vec, policy, spec.final_steps)
    try:  # command-range extrapolation x1.5
      term = eval_vec.unwrapped.command_manager.get_term("twist")
      r = term.cfg.ranges
      r.lin_vel_x = (r.lin_vel_x[0] * 1.5, r.lin_vel_x[1] * 1.5)
      r.ang_vel_z = (r.ang_vel_z[0] * 1.5, r.ang_vel_z[1] * 1.5)
      final["cmd_extrap"] = evaluate(eval_vec, policy, spec.final_shift_steps)
    except Exception as e:
      final["cmd_extrap_error"] = repr(e)
    final["delay1"] = evaluate(eval_vec, policy, spec.final_shift_steps, delay=1)
  except Exception as e:
    final["deploy_error"] = repr(e)
  final["wall_s"] = time.time() - t_start
  with open(os.path.join(out_dir, "final.json"), "w") as f:
    json.dump(final, f, indent=2, default=float)
  if prefix:
    from mjlab_centering import r2

    try:
      r2.retry(lambda: r2.upload_file(os.path.join(out_dir, "final.json"), f"{prefix}/final.json"), attempts=3)
    except Exception as e:
      print(f"[warn] deployment eval upload failed: {e!r}", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--job", default=None, help="job dict as JSON (overrides the flags below)")
  ap.add_argument("--job-id", default="smoke")
  ap.add_argument("--block", default="smoke")
  ap.add_argument("--robot", default="go1")
  ap.add_argument("--seed", type=int, default=2000)
  ap.add_argument("--center", default="none", choices=["none", "value", "simple", "simple_raw"])
  ap.add_argument("--gamma", type=float, default=0.99)
  ap.add_argument("--offset", type=float, default=0.0)
  ap.add_argument("--penalty", type=float, default=10.0)
  ap.add_argument("--iters", type=int, default=500)
  ap.add_argument("--eta", type=float, default=0.01)
  ap.add_argument("--beta", type=float, default=0.01)
  ap.add_argument("--lr", type=float, default=None)
  ap.add_argument("--value-clip", default=None, choices=["true", "false"])
  ap.add_argument("--ret-norm", default="none", choices=["none", "meanstd", "rlgames"])
  ap.add_argument("--reward-scale", default="none", choices=["none", "std", "ret_std"])
  ap.add_argument("--log-rew-std", action="store_true", help="log the per-step reward std (task and raw) every iteration")
  ap.add_argument("--vn-clamp", type=float, default=5.0, help="rl_games value-norm clamp; 0 disables (diagnostic arm)")
  ap.add_argument("--bias-lr", type=float, default=None, help="A1: critic output bias in its own group at this lr")
  ap.add_argument("--warmup", action="store_true", help="A2: freeze the actor until the critic's level arrives")
  ap.add_argument("--warmup-frac", type=float, default=0.95)
  ap.add_argument("--warmup-max-iters", type=int, default=1000)
  ap.add_argument("--actor-lr-scale", type=float, default=None, help="A3: actor lr x scale for the first N iterations")
  ap.add_argument("--actor-lr-iters", type=int, default=0)
  ap.add_argument("--lam", type=float, default=None, help="GAE lambda (None = the runner's 0.95)")
  ap.add_argument("--adv-mode", default="global", choices=["global", "pos_norm", "pos_center"])
  ap.add_argument("--log-adv-pos", action="store_true")
  ap.add_argument("--shock-kind", default="none", choices=["none", "offset", "action_rate"])
  ap.add_argument("--shock-at", type=int, default=None)
  ap.add_argument("--shock-factor", type=float, default=5.0)
  ap.add_argument("--p-reset", type=float, default=0.0)
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--eval-envs", type=int, default=256)
  ap.add_argument("--eval-every", type=int, default=25)
  ap.add_argument("--eval-steps", type=int, default=300)
  ap.add_argument("--final-steps", type=int, default=5000)
  ap.add_argument("--final-shift-steps", type=int, default=3000)
  ap.add_argument("--r-bar-init", default="zero", choices=["zero", "first_batch"])
  ap.add_argument("--encoder", default="raw", choices=["raw", "ssp_low", "rff_low"])
  ap.add_argument("--ssp-length-scale", type=float, default=0.3)
  ap.add_argument("--save-at", type=int, default=None)
  ap.add_argument("--resume", default=None)
  ap.add_argument("--save-every", type=int, default=0, help="periodic checkpoint every N iterations (0 = off)")
  ap.add_argument("--episodic", action="store_true", help="reference arm: train on mjlab's unmodified episodic env")
  ap.add_argument("--no-falls-reset", action="store_true",
                  help="a fallen robot is NOT reset and stays down until the time-out (diagnostic only)")
  ap.add_argument("--no-time-outs", action="store_true", help="legacy protocol: no time-outs at all (waves 1-9)")
  ap.add_argument("--obs-v130", action="store_true", help="reproduce mjlab 1.3.0's observation layout")
  ap.add_argument("--upload-ckpt", action="store_true")
  ap.add_argument("--out", default=None)
  ap.add_argument("--r2", default=None, help="R2 campaign prefix, e.g. mjlab_ssp_centering/c1")
  ap.add_argument("--quiet", action="store_true")
  a = ap.parse_args()

  if a.job:
    spec = build_spec(json.loads(a.job))
  else:
    spec = RunSpec(
      job_id=a.job_id, block=a.block, robot=a.robot, seed=a.seed, center=a.center, gamma=a.gamma,
      offset=a.offset, penalty=a.penalty, iters=a.iters, eta=a.eta, beta=a.beta, learning_rate=a.lr,
      value_clip=None if a.value_clip is None else (a.value_clip == "true"), ret_norm=a.ret_norm,
      reward_scale=a.reward_scale, shock_kind=a.shock_kind, shock_at=a.shock_at,
      log_rew_std=a.log_rew_std, vn_clamp=(a.vn_clamp or None), bias_lr=a.bias_lr, warmup=a.warmup,
      warmup_frac=a.warmup_frac, warmup_max_iters=a.warmup_max_iters, actor_lr_scale=a.actor_lr_scale,
      actor_lr_iters=a.actor_lr_iters, lam=a.lam, adv_mode=a.adv_mode, log_adv_pos=a.log_adv_pos,
      shock_factor=a.shock_factor, p_reset=a.p_reset, num_envs=a.num_envs, eval_envs=a.eval_envs,
      eval_every=a.eval_every, eval_steps=a.eval_steps, final_steps=a.final_steps,
      final_shift_steps=a.final_shift_steps, upload_ckpt=a.upload_ckpt, save_at=a.save_at, resume=a.resume, save_every=a.save_every, episodic=a.episodic, r_bar_init=a.r_bar_init, falls_reset=not a.no_falls_reset, time_outs=not a.no_time_outs, obs_v130=a.obs_v130,
      extra={"encoder": a.encoder, "ssp_length_scale": a.ssp_length_scale},
    )
  out = a.out or os.path.join("results", spec.job_id)
  res = run(spec, out, r2_campaign=a.r2, quiet=a.quiet)
  print(json.dumps({"job_id": res["job_id"], "steps": res["steps"], "wall_s": res.get("wall_s", res.get("train_wall_s"))}, default=float))


if __name__ == "__main__":
  main()
