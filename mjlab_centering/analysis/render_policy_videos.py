"""Render short videos of trained Go1 / G1 policies (centered vs uncentered, same seed) for the supervisor report.

The videos are ILLUSTRATIVE RETRAINED RUNS (block r7vid, campaign r7videoruns-*; same config and seed as the claim cell),
not the analysed claim runs, whose checkpoints were never saved (d151). Everything here is read-only against R2.

    python analysis/render_policy_videos.py all                 # fetch the 4 retrained runs, render mp4 + gif, write README
    python analysis/render_policy_videos.py clip --dir D --robot go1 --arm value --seed 3002 --logged 0.8829 --out x.mp4
    python analysis/render_policy_videos.py select              # print the median seeds (no GPU needed)

Rendering uses mjlab's own machinery at the pinned commit (8bb2bc9e): `ManagerBasedRlEnv(render_mode="rgb_array")` ->
`OffscreenRenderer` (mujoco.Renderer) with a `ViewerConfig` tracking camera on the robot root. `mjlab play --video` is not
used because it loads rsl-rl runner checkpoints with mjlab's own command resampling, while these policies are
`CenteredPPO` state dicts that must be driven by a fixed command script. The env is built with the same `build_env_cfg`
as training (observation terms and normaliser state come from config.json and the checkpoint); the reward offset and
centering do not change actions, so no continuing wrapper logic is needed beyond constructing the same agent.

Run from mjlab_centering/ with its venv. Backend: MUJOCO_GL=egl (override with the environment variable).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MJ = HERE.parent
REPO = MJ.parent
sys.path.insert(0, str(MJ / "src"))
sys.path.insert(0, str(HERE))

CAMPAIGN = "r7videoruns-90413106"
VID_DIR = REPO / "docs" / "figures" / "supervisor" / "videos"
CACHE = MJ / "results" / "video_runs"
FPS = 30
WIDTH, HEIGHT = 1280, 720

# (label, vx, vy, wz, seconds). Commands are body-frame, fixed, never resampled. "turn left" is a walking turn (vx 0.4, wz +0.5
# rad/s, the top of the trained yaw range); "sideways" is a left strafe at 0.4 m/s.
SCRIPT = (
  ("forward 0.8 m/s", 0.8, 0.0, 0.0, 4.0),
  ("turn left", 0.4, 0.0, 0.5, 3.0),
  ("sideways (left)", 0.0, 0.4, 0.0, 3.0),
  ("forward 0.8 m/s", 0.8, 0.0, 0.0, 3.0),
  ("stop", 0.0, 0.0, 0.0, 3.0),
)
ROBOTS = {  # name -> (display, camera distance m, elevation deg, azimuth deg)
  "go1": ("Go1", 2.6, -18.0, 150.0),
  "g1": ("G1", 3.6, -14.0, 150.0),
}
ARM = {"value": "centered, offset +1", "none": "uncentered, offset +1"}
JOBS = {  # (robot, arm) -> (job id, seed, claim job id used for the logged value)
  ("go1", "value"): "r7vid_go1_g0.999_o1_value_s3002",
  ("go1", "none"): "r7vid_go1_g0.999_o1_none_s3002",
  ("g1", "value"): "r7vid_g1_g0.99_o1_value_s3012",
  ("g1", "none"): "r7vid_g1_g0.99_o1_none_s3012",
}


# ------------------------------------------------------------------ seed selection (no cherry-picking)
def lower_median_seed(track: dict[int, float]) -> tuple[int, float]:
  """Seed at rank n/2 (lower median, 1-based) of the logged last-5-eval tracking; n is even (20), the two middle seeds are
  equidistant from the median, so the lower one is taken by rule."""
  s = sorted(track.items(), key=lambda kv: (kv[1], kv[0]))
  k = (len(s) - 1) // 2
  return s[k]


def logged_tracking() -> dict[str, dict[str, dict[int, float]]]:
  """{robot: {arm: {seed: last-5 tracking}}} for the claim cells, from the local R2 mirrors used by the supervisor figures."""
  import make_supervisor_figs as M

  out = {"go1": {}, "g1": {}}
  for arm in ("value", "none"):
    out["go1"][arm] = dict(M.go1_stats(arm, 1.0)["track"])
  g1 = M.g1_data()
  for arm in ("value", "none"):
    out["g1"][arm] = {s: v["track"] for s, v in g1[arm].items()}
  return out


def cmd_select(_a) -> dict:
  lt = logged_tracking()
  sel = {}
  for robot in ("go1", "g1"):
    seed, tr = lower_median_seed(lt[robot]["value"])
    sel[robot] = {"seed": seed, "centered_logged": tr, "uncentered_logged": lt[robot]["none"].get(seed),
                  "n_seeds": len(lt[robot]["value"])}
    print(robot, sel[robot])
  return sel


# ------------------------------------------------------------------ fetch (download only)
def fetch(job: str) -> Path:
  from mjlab_centering import r2

  d = CACHE / job
  d.mkdir(parents=True, exist_ok=True)
  for f in ("model_final.pt", "config.json", "eval.jsonl", "final.json"):
    if (d / f).exists():
      continue
    r2.client().download_file(r2.bucket(), r2.key(f"{CAMPAIGN}/runs/{job}/{f}"), str(d / f))
  return d


def last5_tracking(d: Path) -> float:
  rows = [json.loads(l) for l in (d / "eval.jsonl").read_text().splitlines() if l.strip()]
  starts = [i for i, r in enumerate(rows) if r.get("iter") == 0]  # newest attempt only, as analysis/report4.newest
  rows = rows[starts[-1]:] if len(starts) > 1 else rows
  return float(np.mean([r["track_lin_frac"] for r in rows[-5:]]))


# ------------------------------------------------------------------ rendering
def _font(size: int):
  from PIL import ImageFont

  for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Regular.ttf"):
    try:
      return ImageFont.truetype(name, size)
    except OSError:
      continue
  return ImageFont.load_default()


def overlay(frame: np.ndarray, header: list[str], footer: str) -> np.ndarray:
  from PIL import Image, ImageDraw

  im = Image.fromarray(frame).convert("RGBA")
  lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
  dr = ImageDraw.Draw(lay)
  f1, f2 = _font(20), _font(18)
  pad, lh = 10, 26
  w = max(dr.textlength(t, font=f1) for t in header) + 2 * pad
  dr.rectangle([8, 8, 8 + w, 8 + lh * len(header) + pad], fill=(0, 0, 0, 150))
  for i, t in enumerate(header):
    dr.text((8 + pad, 8 + pad // 2 + i * lh), t, font=f1, fill=(255, 255, 255, 255))
  fw = dr.textlength(footer, font=f2) + 2 * pad
  dr.rectangle([8, im.size[1] - 8 - 34, 8 + fw, im.size[1] - 8], fill=(0, 0, 0, 150))
  dr.text((8 + pad, im.size[1] - 8 - 29), footer, font=f2, fill=(255, 255, 255, 255))
  return np.asarray(Image.alpha_composite(im, lay).convert("RGB"))


def render_clip(d: Path, robot: str, arm: str, seed: int, logged: float | None, out: Path, retrained: float | None = None,
                quiet: bool = False) -> dict:
  import imageio.v2 as imageio
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer import ViewerConfig
  from mjlab_centering.algo import CenteredPPO
  from mjlab_centering.env_cfg import build_agent_cfg, build_env_cfg
  from mjlab_centering.train import build_spec
  from mjlab_centering.wrapper import ContinuingVecEnv

  cfgj = json.loads((d / "config.json").read_text())
  spec = build_spec(cfgj)
  assert spec.robot == robot and spec.center == arm and spec.seed == seed, (spec.robot, spec.center, spec.seed)
  ex = spec.extra or {}
  device = "cuda" if torch.cuda.is_available() else "cpu"
  torch.manual_seed(seed)

  cfg = build_env_cfg(
    robot=spec.robot, num_envs=1, continuing=True, encoder=ex.get("encoder", "raw"),
    ssp_length_scale=float(ex.get("ssp_length_scale", 0.3)), ssp_n_rotates=int(ex.get("ssp_n_rotates", 8)),
    ssp_n_scales=int(ex.get("ssp_n_scales", 8)), ssp_rff_seed=int(ex.get("ssp_rff_seed", 0)),
    falls_reset=bool(spec.falls_reset), time_outs=bool(spec.time_outs), obs_v130=bool(spec.obs_v130),
  )
  cfg.seed = int(seed)
  _, dist, elev, azim = ROBOTS[robot]
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_ROOT, entity_name="robot", width=WIDTH, height=HEIGHT,
    distance=dist, elevation=elev, azimuth=azim, max_extra_envs=0, enable_shadows=True, enable_reflections=False,
  )
  # Debug overlays (contact markers, command arrows, ray casts) are purely visual; switch them off for a clean picture.
  for c in list(cfg.commands.values()) + list(cfg.scene.sensors):
    if hasattr(c, "debug_vis"):
      c.debug_vis = False
  cmd_cfg = cfg.commands["twist"]
  cmd_cfg.resampling_time_range = (1.0e6, 1.0e6)  # never resample; the script owns the command
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  vec = ContinuingVecEnv(env, penalty=spec.penalty, time_outs=bool(spec.time_outs),
                         fall_check_deg=None if spec.falls_reset else 70.0, offset=spec.offset)

  agent_cfg = build_agent_cfg(
    robot=spec.robot, max_iterations=spec.iters, seed=spec.seed, center=spec.center, eta=spec.eta, beta=spec.beta,
    value_clip=spec.value_clip, ret_norm=spec.ret_norm, reward_scale=spec.reward_scale, r_bar_init=spec.r_bar_init,
    learning_rate=spec.learning_rate, num_steps_per_env=spec.num_steps_per_env,
  )
  agent_cfg["algorithm"]["gamma"] = float(spec.gamma)
  obs = vec.get_observations()
  alg = CenteredPPO.construct_algorithm(obs, vec, agent_cfg, device)
  ck = torch.load(d / "model_final.pt", map_location=device, weights_only=False)
  if "cc_state" in ck:
    alg.import_state(ck)  # actor, critic, observation normaliser statistics, centering state
  else:  # checkpoints written before export_state existed (dry runs only)
    alg.load(ck, None, True)
  policy = alg.get_policy()

  term = env.command_manager.get_term("twist")
  term.is_heading_env[:] = False
  term.is_world_env[:] = False
  term.is_standing_env[:] = False
  term.is_forward_env[:] = False
  robot_ent = env.scene["robot"]
  dt = float(env.step_dt)

  disp = ROBOTS[robot][0]
  tag = "retrained" if retrained is not None else "logged"
  head = [f"{disp}  |  {ARM[arm]}  |  seed {seed}",
          f"final tracking: {retrained:.3f} (this retrained run)" if retrained is not None else "",
          f"claim-cell value, same seed: {logged:.3f}" if logged is not None else "",
          "illustrative retrained run (same config and seed as the claim cell)"]
  head = [h for h in head if h]

  vec.reset()
  obs = vec.get_observations()
  frames, seg_stats, step = [], [], 0
  n_frames_total = int(round(sum(s[4] for s in SCRIPT) * FPS))
  writer = imageio.get_writer(str(out), fps=FPS, codec="libx264", quality=None, macro_block_size=None,
                              ffmpeg_params=["-crf", "22", "-preset", "medium"])
  try:
    next_frame, t_seg0 = 0, 0.0
    for label, vx, vy, wz, secs in SCRIPT:
      n_steps = int(round(secs / dt))
      vals = []
      for _ in range(n_steps):
        c = torch.tensor([vx, vy, wz], device=device)
        term.vel_command_b[:] = c
        term.vel_command_w[:] = c
        term.is_heading_env[:] = False
        term.is_world_env[:] = False
        term.is_standing_env[:] = False
        obs = vec.get_observations()  # reflects the command just written
        with torch.no_grad():
          act = policy(obs, stochastic_output=False)  # mean action
        obs, rew, dones, extras = vec.step(act)
        step += 1
        v_b = robot_ent.data.root_link_lin_vel_b[0]
        w_b = robot_ent.data.root_link_ang_vel_b[0]
        vals.append([float(v_b[0]), float(v_b[1]), float(w_b[2]), float(robot_ent.data.root_link_pos_w[0, 2])])
        t = step * dt
        while next_frame < n_frames_total and next_frame / FPS <= t:
          img = env.render()
          writer.append_data(overlay(np.asarray(img), head, f"t = {t:4.1f} s   command: {label}  (vx {vx:+.1f}, vy {vy:+.1f}, wz {wz:+.1f})"))
          next_frame += 1
      a = np.asarray(vals)
      seg_stats.append({"command": label, "cmd": [vx, vy, wz], "seconds": secs,
                        "mean_vx": float(a[:, 0].mean()), "mean_vy": float(a[:, 1].mean()), "mean_wz": float(a[:, 2].mean()),
                        "min_root_z": float(a[:, 3].min())})
      if not quiet:
        print(f"  {label:18s} cmd ({vx:+.1f},{vy:+.1f},{wz:+.1f})  realised vx {a[:,0].mean():+.2f} vy {a[:,1].mean():+.2f} wz {a[:,2].mean():+.2f}", flush=True)
  finally:
    writer.close()
    env.close()
  return {"segments": seg_stats, "frames": next_frame, "step_dt": dt, "fps": FPS, "size": [WIDTH, HEIGHT], "torch": torch.__version__,
          "mujoco_gl": os.environ["MUJOCO_GL"]}


def make_gif(mp4: Path, gif: Path, width: int = 640, fps: int = 12, max_mb: float = 10.0) -> float:
  """Palette GIF; shrinks width / fps until it is under max_mb."""
  for w, f in ((width, fps), (560, 10), (480, 10), (400, 8), (320, 8)):
    filt = f"fps={f},scale={w}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=5"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-filter_complex", filt, str(gif)], check=True)
    mb = gif.stat().st_size / 1e6
    if mb <= max_mb:
      return mb
  return mb


def cmd_clip(a) -> None:
  d = Path(a.dir)
  out = Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  info = render_clip(d, a.robot, a.arm, a.seed, a.logged, out, retrained=last5_tracking(d) if (d / "eval.jsonl").exists() else None)
  print(json.dumps(info)[:300])


def cmd_all(a) -> None:
  VID_DIR.mkdir(parents=True, exist_ok=True)
  sel = json.loads((VID_DIR / "selection.json").read_text()) if (VID_DIR / "selection.json").exists() and not a.reselect else cmd_select(a)
  (VID_DIR / "selection.json").write_text(json.dumps(sel, indent=1))
  runs = {}
  for (robot, arm), job in JOBS.items():
    seed = sel[robot]["seed"]
    assert job.endswith(f"_s{seed}"), (job, seed)
    d = fetch(job)
    retr = last5_tracking(d)
    logged = sel[robot]["centered_logged"] if arm == "value" else sel[robot]["uncentered_logged"]
    name = f"{robot}_{'centered' if arm == 'value' else 'uncentered'}_seed{seed}"
    out = VID_DIR / f"{name}.mp4"
    print(f"[{name}] retrained tracking {retr:.4f}, logged claim-cell {logged}", flush=True)
    info = render_clip(d, robot, arm, seed, logged, out, retrained=retr)
    mb = make_gif(out, VID_DIR / f"{name}.gif")
    cfgj = json.loads((d / "config.json").read_text())
    runs[name] = {"robot": robot, "arm": arm, "seed": seed, "job": job, "ckpt": f"R2 mjlab_ssp_centering/{CAMPAIGN}/runs/{job}/model_final.pt "
                  f"(local copy {d.relative_to(REPO)}/model_final.pt)", "retrained_tracking": retr, "logged_tracking_same_seed": logged,
                  "gif_mb": round(mb, 2), "mp4_mb": round(out.stat().st_size / 1e6, 2), "gpu": cfgj.get("gpu"), "train_torch": cfgj.get("torch"),
                  "render": info}
  (VID_DIR / "clips.json").write_text(json.dumps(runs, indent=1))
  write_readme(sel, runs)


def write_readme(sel: dict, runs: dict) -> None:
  script = " -> ".join(f"{s[0]} ({s[4]:g} s, cmd vx {s[1]:+.1f} vy {s[2]:+.1f} wz {s[3]:+.1f})" for s in SCRIPT)
  L = ["# Policy videos (illustrative retrained runs)", "",
       "**Every clip is an illustrative retrained run (same config and seed as the claim cell), not one of the analysed runs.** "
       "The claim runs never saved their policies (`upload_ckpt` was false; d151), so one centered/uncentered pair per robot was "
       "retrained with `jobs_r4.py video-runs` (campaign `r7videoruns-90413106`, block `r7vid`, checkpoints on R2) and rendered here. "
       "A retrained run does not reproduce its logged tracking exactly; both numbers are given.", "",
       "## Seed choice (no cherry-picking)", "",
       "Per robot: the centered claim seed at the **lower median** (rank 10 of 20) of the logged final tracking "
       "(mean of the last 5 evals, claim seeds 3000-3019, the values behind the supervisor figures). 20 is even and the two middle "
       "seeds are equidistant from the median, so the lower one is taken by rule. `selection.json` has the numbers.", ""]
  for r in ("go1", "g1"):
    s = sel[r]
    L.append(f"- {ROBOTS[r][0]}: seed **{s['seed']}**, logged centered tracking {s['centered_logged']:.4f} "
             f"(uncentered partner, same seed, logged {s['uncentered_logged']:.3f}).")
  L += ["", "## What the clips show (read from the clips and the per-segment velocities below)", "",
        "- Go1 centered walks every segment (realised vx 0.74-0.75 against 0.8 commanded, wz 0.51 against 0.5). Go1 uncentered stays on the floor (root z 0.07-0.08 m, velocities about 0): the standing-floor failure of the claim cell.",
        "- G1 centered walks upright throughout (root z 0.74-0.75 m); its second forward segment realises vx 0.49 against 0.8, so the tracking is good, not perfect (final tracking 0.81).",
        "- G1 uncentered stands upright and shuffles for about the first 10 s without following the commands, then collapses and stays down (root z 0.12 m in the last two segments; torso tilt stays under the 70 deg reset limit, so no reset fires). This is one clip of one seed; its tracking value (0.30) is the floor shared by all 20 uncentered claim seeds.", "",
        "## Clips", ""]
  for name, c in runs.items():
    L += [f"### {name}.mp4 / .gif", "",
          f"- arm: {ARM[c['arm']]}; robot {ROBOTS[c['robot']][0]}; seed {c['seed']}; job `{c['job']}`",
          f"- checkpoint: {c['ckpt']}",
          f"- final tracking (last-5 evals): **{c['retrained_tracking']:.4f}** this retrained run; {c['logged_tracking_same_seed']:.4f} logged in the claim cell (same seed)",
          f"- trained on GPU `{c['gpu']}`, torch {c['train_torch']}; rendered with torch {c['render']['torch']}, MUJOCO_GL={c['render']['mujoco_gl']}",
          f"- sizes: mp4 {c['mp4_mb']} MB, gif {c['gif_mb']} MB", "- realised velocities per command segment (body frame, mean over the segment):", ""]
    for g in c["render"]["segments"]:
      L.append(f"  - {g['command']}: command ({g['cmd'][0]:+.1f}, {g['cmd'][1]:+.1f}, {g['cmd'][2]:+.1f}) -> "
               f"vx {g['mean_vx']:+.2f}, vy {g['mean_vy']:+.2f}, wz {g['mean_wz']:+.2f}, min root z {g['min_root_z']:.2f} m")
    L.append("")
  L += ["## Command sequence (fixed, body frame; never resampled)", "", script, "",
        "Commands are (vx m/s, vy m/s, wz rad/s). 'turn left' is a walking turn (vx 0.4, wz +0.5, the top of the trained yaw range). "
        "The heading/world-frame/standing command modes are switched off so the script is exact.", "",
        "## Render settings", "",
        f"- {WIDTH}x{HEIGHT}, {FPS} fps (control step 0.02 s; frames are taken whenever simulated time passes k/30 s), "
        f"H.264 yuv420p crf 22; GIF 640 px wide, 12 fps, 96-colour palette (shrinks automatically if over 10 MB).",
        "- mjlab commit 8bb2bc9e, `ManagerBasedRlEnv(render_mode='rgb_array')` with mjlab's `OffscreenRenderer`; tracking camera on the "
        "robot root (`ViewerConfig.OriginType.ASSET_ROOT`; Go1 distance 2.6 m / G1 3.6 m, elevation -18/-14 deg, azimuth 150 deg). "
        "One env, deterministic (actor mean action, `stochastic_output=False`), observation normaliser and encoder from the "
        "checkpoint and `config.json` (raw encoder, as in the claim cells). mjlab's `fell_over` termination (torso tilt > 70 deg) stays active as in training, so a fall past that angle would reset the robot in place; a robot that collapses without crossing it (the uncentered G1) simply stays down.",
        "- Overlay (top-left): robot, arm, seed, final tracking, retrained-run label; (bottom-left): time and current command.", "",
        "## Regenerate", "",
        "```bash", "cd mjlab_centering", ".venv/bin/python analysis/render_policy_videos.py all", "```", "",
        "This downloads the four checkpoints (read-only) into `results/video_runs/`, re-derives the seeds from the local mirrors "
        "(`selection.json` is reused unless `--reselect`), renders the mp4 and gif files and rewrites this README. "
        "To retrain the checkpoints: `python jobs_r4.py video-runs` then launch the campaign with `orchestrator.py` (see decision 151)."]
  (VID_DIR / "README.md").write_text("\n".join(L) + "\n")


def main() -> None:
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  sub.add_parser("select").set_defaults(f=cmd_select)
  p = sub.add_parser("all")
  p.add_argument("--reselect", action="store_true")
  p.set_defaults(f=cmd_all)
  p = sub.add_parser("clip")
  p.add_argument("--dir", required=True)
  p.add_argument("--robot", required=True, choices=list(ROBOTS))
  p.add_argument("--arm", required=True, choices=list(ARM))
  p.add_argument("--seed", type=int, required=True)
  p.add_argument("--logged", type=float, default=None)
  p.add_argument("--out", required=True)
  p.set_defaults(f=cmd_clip)
  a = ap.parse_args()
  a.f(a)


if __name__ == "__main__":
  main()
