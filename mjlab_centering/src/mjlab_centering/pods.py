"""RunPod pod helpers (thin wrapper over `runpodctl`, which is version-pinned and authoritative for flags)."""

from __future__ import annotations

import json
import re
import subprocess
import time

UA_ENV = {"RUNPOD_API_KEY": ""}  # filled from the repo .env by the orchestrator


def _key() -> str:
  from mjlab_centering.r2 import load_env, _repo_root

  env = load_env()
  if "RUNPOD_API_KEY" in env:
    return env["RUNPOD_API_KEY"]
  for line in (_repo_root() / ".env").read_text().splitlines():
    if line.startswith("RUNPOD_API_KEY="):
      return line.split("=", 1)[1].strip()
  raise RuntimeError("RUNPOD_API_KEY not found")


_SECRET_PAT = re.compile(r"(rpa_[A-Za-z0-9]+|[A-Fa-f0-9]{32,})")


def redact(text: str) -> str:
  """Never let secrets reach a log: RUNPOD_API_KEY and R2 keys are 32+ hex chars or rpa_* tokens."""
  return _SECRET_PAT.sub("<redacted>", text)


TRANSIENT = ("network_error", "dial tcp", "server misbehaving", "temporary failure", "timeout", "TLS")


def _run(args: list[str]) -> str:
  """Run runpodctl, retrying transient network failures.

  A single DNS blip killed the whole width-stage hunter for six hours on 2026-11-06 (`rest.runpod.io` lookup failed,
  systemd then gave up restarting it). Retrying here means the caller never sees those.
  """
  env = {"PATH": "/usr/bin:/bin:/home/kevin/.local/bin", "RUNPOD_API_KEY": _key(), "HOME": "/home/kevin"}
  last = ""
  for attempt, wait in enumerate((0, 5, 15, 45)):
    if wait:
      time.sleep(wait)
    r = subprocess.run(["runpodctl", *args], capture_output=True, text=True, env=env, timeout=300)
    if r.returncode == 0:
      return r.stdout
    last = (r.stderr or r.stdout).strip()
    if not any(t.lower() in last.lower() for t in TRANSIENT):
      break
  raise RuntimeError(f"runpodctl {args[0]} {args[1]} failed: {last[:400]}")


def list_pods() -> list[dict]:
  out = _run(["pod", "list", "--all"])
  try:
    data = json.loads(out)
  except json.JSONDecodeError:
    return []
  return data if isinstance(data, list) else data.get("pods", [])


def create_pod(name: str, image: str, gpu_id: str, docker_args: str, env: dict, cloud: str = "COMMUNITY",
               disk_gb: int = 40, min_cuda: str | None = None, gpu_count: int = 1,
               data_center_ids: str | None = None, ssh: bool = False) -> str:
  args = [
    "pod", "create",
    "--name", name,
    "--image", image,
    "--gpu-id", gpu_id,
    "--gpu-count", str(gpu_count),
    "--cloud-type", cloud,
    "--container-disk-in-gb", str(disk_gb),
    "--env", json.dumps(env),
    "--docker-args", docker_args,
  ]
  # 40 GB container disk: the image is ~13 GB and the venv adds ~5 GB, so 20 GB leaves no room for the lab stack (the
  # 2026-11-06 20 GB pod never pushed a boot log in 20 min). The ssh/ports line below is what actually blocked creates.
  # `--ssh` asks the host for a port-22 mapping. Community hosts almost never publish one (the last four pods reported
  # "port 22 is declared but the host has not published a mapping", and the sshd in the bootstrap never started), and
  # asking for it made every create refuse with "no longer any instances available with the requested specifications"
  # while a 20 GB disk without it was accepted immediately (measured 2026-11-06, d170). Off unless explicitly wanted.
  if ssh:
    args += ["--ssh", "--ports", "22/tcp"]
  if min_cuda:
    args += ["--min-cuda-version", min_cuda]
  if data_center_ids:
    args += ["--data-center-ids", data_center_ids]
  out = _run(args)
  try:
    pod = json.loads(out)
    return pod.get("id") or pod.get("podId")
  except json.JSONDecodeError:
    for tok in out.split():
      if len(tok) > 8 and all(ch.isalnum() for ch in tok):
        return tok
  raise RuntimeError(f"could not parse pod id from: {out[:300]}")


def delete_pod(pod_id: str, api_key: str | None = None) -> None:
  import urllib.request

  key = api_key or _key()
  req = urllib.request.Request(
    f"https://api.runpod.io/v2/pods/{pod_id}", method="DELETE",
    headers={"Authorization": "Bearer " + key, "User-Agent": "curl/8.5.0"},
  )
  with urllib.request.urlopen(req, timeout=60):
    pass


def pod_created_ts(pod: dict) -> float | None:
  """Best-effort creation time from a pod record (field names have moved between API versions)."""
  for k in ("createdAt", "lastStatusChange", "created_at"):
    v = pod.get(k)
    if isinstance(v, str):
      try:
        return time.mktime(time.strptime(v[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
      except Exception:
        continue
    if isinstance(v, (int, float)) and v > 1e9:
      return float(v)
  rt = pod.get("runtime") or {}
  up = rt.get("uptimeInSeconds")
  if isinstance(up, (int, float)) and up > 0:
    return time.time() - float(up)
  return None


def community_stock() -> list[dict]:
  """Community GPU SKUs with stock in at least one datacentre, cheapest first.

  `gpu list --include-unavailable` reports per-DC stock; the top-level flag hides no-stock SKUs, so this is the only read
  that says which SKU is worth trying right now (d122: community stock flashes for seconds, and a blind cycle of every
  candidate spends create calls on SKUs that are empty).
  """
  out = _run(["gpu", "list", "--include-unavailable"])
  try:
    gpus = json.loads(out)
  except json.JSONDecodeError:
    return []
  rows = []
  for g in gpus if isinstance(gpus, list) else gpus.get("gpus", []):
    if not g.get("communityCloud"):
      continue
    price = g.get("communityPricePerHr")
    dcs = [d.get("dataCenterId") for d in (g.get("dataCenterAvailability") or []) if d.get("stockStatus") not in (None, "none")]
    if price is None or not dcs:
      continue
    rows.append({"gpuId": g.get("gpuId") or g.get("displayName"), "price": float(price), "dcs": dcs})
  return sorted(rows, key=lambda r: r["price"])
