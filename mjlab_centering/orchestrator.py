"""Campaign orchestrator (runs on the laptop): ship code, launch pods, monitor R2, enforce the cost cap.

Completion is decided from R2 `DONE` markers, never from the pod list (finished pods self-delete, and inferring
"outstanding" from pods relaunches completed work). Spend is pod-hours x price, accumulated per pod from the
orchestrator's own state file and reconciled against each pod's uptime when the API reports it.

    python orchestrator.py --campaign c1-abc123 --plan            # dry run: jobs, estimate, no pods
    python orchestrator.py --campaign c1-abc123 --launch 3        # 3 pods, until the queue drains
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

from mjlab_centering import pods, r2

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "src"
DOCS = ROOT.parent / "docs" / "mjlab_centering"

BOOTSTRAP = r"""
set -uo pipefail
mkdir -p /project /tmp
exec > >(tee -a /tmp/boot.log) 2>&1
fail() {
  echo "[boot] FAILED: $*"
  python - <<'PY' || true
import os, boto3
from botocore.config import Config as _C
e = os.environ
s3 = boto3.client('s3', endpoint_url='https://%s.r2.cloudflarestorage.com' % e['R2_ACCOUNT_ID'],
                  aws_access_key_id=e['R2_ACCESS_KEY_ID'], aws_secret_access_key=e['R2_SECRET_ACCESS_KEY'],
                  config=_C(connect_timeout=10, read_timeout=60, retries={'max_attempts': 3}))
pod = e.get('RUNPOD_POD_ID', 'pod')
s3.upload_file('/tmp/boot.log', e['R2_BUCKET'], 'mjlab_ssp_centering/%s/logs/boot_%s.log' % (e['CAMPAIGN'], pod))
PY
  python - <<'PY' || true
import os, urllib.request
pod, key = os.environ.get('RUNPOD_POD_ID'), os.environ.get('RUNPOD_API_KEY')
if pod and key:
    req = urllib.request.Request('https://api.runpod.io/v2/pods/%s' % pod, method='DELETE',
                                 headers={'Authorization': 'Bearer ' + key, 'User-Agent': 'curl/8.5.0'})
    urllib.request.urlopen(req, timeout=60)
    print('[boot] pod deleted after failure')
PY
  sleep 3600
}
boot_log() {
  # Push the current boot log. Called at every stage, so a pod that hangs is diagnosable from the laptop without SSH
  # (2026-11-06: a pod sat RUNNING for hours with no log at all, because the log was only pushed on failure).
  # BOUNDED: this call sits on the boot's critical path, and an unbounded R2 upload here hung five pods at stage=queue.
  timeout 60 python - <<'PY' || true
import os, boto3
from botocore.config import Config as _C
e = os.environ
if not e.get('CAMPAIGN'):
    raise SystemExit(0)
s3 = boto3.client('s3', endpoint_url='https://%s.r2.cloudflarestorage.com' % e['R2_ACCOUNT_ID'],
                  aws_access_key_id=e['R2_ACCESS_KEY_ID'], aws_secret_access_key=e['R2_SECRET_ACCESS_KEY'],
                  config=_C(connect_timeout=10, read_timeout=60, retries={'max_attempts': 3}))
s3.upload_file('/tmp/boot.log', e['R2_BUCKET'], 'mjlab_ssp_centering/%s/logs/boot_%s.log' % (e['CAMPAIGN'], e.get('RUNPOD_POD_ID', 'pod')))
PY
}
retry() {
  # Transient package-index failures must not cost a pod (2026-11-06: one died downloading nvidia-cufft).
  for i in 1 2 3; do
    "$@" && return 0
    echo "[boot] attempt $i failed: $*"
    sleep 20
  done
  return 1
}
watchdog() {
  # A boot that hangs is invisible to the orchestrator (the pod is RUNNING and spending money). Cap it.
  sleep "${BOOT_TIMEOUT_S:-2700}"
  [ -f /tmp/boot_ok ] && return 0
  echo "[boot] WATCHDOG: boot did not reach the queue within ${BOOT_TIMEOUT_S:-2700}s"
  boot_log
  python - <<'PY' || true
import os, urllib.request
pod, key = os.environ.get('RUNPOD_POD_ID'), os.environ.get('RUNPOD_API_KEY')
if pod and key:
    req = urllib.request.Request('https://api.runpod.io/v2/pods/%s' % pod, method='DELETE',
                                 headers={'Authorization': 'Bearer ' + key, 'User-Agent': 'curl/8.5.0'})
    urllib.request.urlopen(req, timeout=60)
    print('[boot] pod deleted by watchdog')
PY
}
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
watchdog &
boot_log
# Preflight: a host that passed the API's CUDA filter can still have a driver too old for torch's cu130 wheels, and
# discovering that after the installs costs ~40 min of a $0.69/h pod (seen 2026-11-05: "cuda check (cu130 fallback too)").
# Check the driver before spending anything on it.
nvidia-smi >/dev/null 2>&1 || fail "no nvidia-smi on this host"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*\.[0-9]*' | head -1 | awk '{print $3}')
echo "[boot] driver=$DRV cuda=$CUDA_VER"
# Drivers measured tonight to be unable to load the cu130 wheel (d180): rejecting them here costs about a minute instead
# of the five it takes to install torch and discover the same thing. This is a measured blacklist, not a rule about
# CUDA versions - 580.95.05 works and 580.159.04 does not.
case "$DRV" in
  580.159.*|580.178.*) fail "driver $DRV is on the known-bad list for torch cu130 (d180)";;
esac
CHANNEL=$(python - "$CUDA_VER" <<'PY'
import sys
def maj_min(v):
    try:
        a, b = (v.split('.') + ['0'])[:2]
        return int(a), int(b)
    except Exception:
        return (0, 0)
v = maj_min(sys.argv[1] if len(sys.argv) > 1 else '')
# A CUDA-13.0 driver is REQUIRED, not merely preferred: warp/mujoco_warp refuses anything older ("minimum required CUDA
# driver version is 13.0"), so a 12.x host cannot run this stack even though torch's cu128 build imports fine there
# (measured on an L40, driver 570.124.04/12.8, 2026-11-06: torch ok, warp IndexError in get_device). d150/d162.
print('cu130' if v >= (13, 0) else 'none')
PY
)
echo "[boot] torch channel=$CHANNEL"
[ "$CHANNEL" != "none" ] || fail "driver CUDA < 13.0: warp/mujoco_warp requires 13.0, so this host cannot run this stack"
export TORCH_CHANNEL="$CHANNEL"
mkdir -p /run/sshd /root/.ssh
[ -f /root/.ssh/authorized_keys ] || true
/usr/sbin/sshd -p 22 2>/dev/null || echo "[boot] sshd not started"
echo "[boot] stage=boto3"
retry pip install -q --no-input boto3 || { fail "pip install boto3"; exit 1; }
boot_log
cd /project || { fail "no /project"; exit 1; }
boot_log; echo "[boot] stage=code"
retry python - <<'PY' || { fail "code download"; exit 1; }
import os, tarfile, boto3
from botocore.config import Config as _C
e = os.environ
s3 = boto3.client('s3', endpoint_url='https://%s.r2.cloudflarestorage.com' % e['R2_ACCOUNT_ID'],
                  aws_access_key_id=e['R2_ACCESS_KEY_ID'], aws_secret_access_key=e['R2_SECRET_ACCESS_KEY'],
                  config=_C(connect_timeout=10, read_timeout=60, retries={'max_attempts': 3}))
s3.download_file(e['R2_BUCKET'], e['CODE_KEY'], '/tmp/code.tgz')
with tarfile.open('/tmp/code.tgz') as t:
    t.extractall('/project')
print('[boot] code extracted', flush=True)
PY
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; } || { fail "git"; exit 1; }
# Ubuntu's system python is PEP-668 externally managed, so install into a venv (not --system).
python3 -m venv /opt/venv || { fail "venv"; exit 1; }
boot_log; echo "[boot] stage=venv"
retry /opt/venv/bin/pip install -q --no-input --upgrade pip uv || { fail "pip install uv"; exit 1; }
# Install ONLY torch first and test it, before pulling mjlab and the rest of the stack. A host whose driver cannot load
# the cu130 wheel fails here in about five minutes (~$0.05) instead of after the whole dependency tree (45 min, $0.63 on
# 2026-11-05). The build tested here is the one the fallback below would use anyway, so a passing host pays nothing extra.
if [ "$TORCH_CHANNEL" = "cu130" ]; then
  TORCH_PIN="torch==2.14.1"; TORCH_IDX="https://download.pytorch.org/whl/cu130"
elif [ "$TORCH_CHANNEL" = "cu128" ]; then
  TORCH_PIN="torch==2.11.0"; TORCH_IDX="https://download.pytorch.org/whl/cu128"
else
  # CUDA 12.x hosts cannot load the cu130 build. torch 2.11 is the newest cu128 wheel; mjlab only branches on >=2.9
  # (mjlab/utils/torch.py), so it is expected to run, and mjlab_centering/pyproject's torch>=2.14 floor is bypassed with
  # a uv override. This pod's build is recorded in every run's config.json, and a campaign is launched with --launch 1 so
  # one build covers all of its cells (d152).
  TORCH_PIN="torch==2.11.0"; TORCH_IDX="https://download.pytorch.org/whl/cu128"
fi
printf '%s\n' "$TORCH_PIN" > /tmp/torch_override.txt
boot_log; echo "[boot] stage=torch $TORCH_PIN"
retry /opt/venv/bin/uv pip install --python /opt/venv/bin/python -q "$TORCH_PIN" \
    --index-url "$TORCH_IDX" --extra-index-url https://pypi.org/simple \
    || { fail "torch install ($TORCH_PIN)"; exit 1; }
echo "[boot] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; ls -l /dev/nvidia* 2>&1 | head -6
# Device nodes can arrive numbered from 1 on multi-GPU hosts while only that one node is passed into the container
# (seen 2026-11-06: /dev/nvidia1 present, /dev/nvidia0 absent, nvidia-smi fine, torch "CUDA unknown error" on every
# wheel). CUDA enumerates from 0, so give it the node it expects before blaming the driver.
if [ ! -e /dev/nvidia0 ] && [ -e /dev/nvidia1 ]; then
  echo "[boot] /dev/nvidia0 missing, /dev/nvidia1 present: linking"
  ln -sf /dev/nvidia1 /dev/nvidia0 && echo "[boot] linked /dev/nvidia0 -> /dev/nvidia1"
fi
torch_ok=0
/opt/venv/bin/python -c "import torch,sys; print('[boot] early torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True); sys.exit(0 if torch.cuda.is_available() else 1)" && torch_ok=1
if [ "$torch_ok" = "0" ]; then
  # "CUDA unknown error" with a working nvidia-smi usually means the device environment, not the wheel: a stale or
  # unmatched CUDA_VISIBLE_DEVICES is the commonest cause on rented hosts (seen 2026-11-06, driver 580.178 / CUDA 13.0,
  # both cu130 and cu128 affected). Try clearing it before touching the install.
  echo "[boot] torch cannot see the device; retrying with CUDA_VISIBLE_DEVICES cleared"
  unset CUDA_VISIBLE_DEVICES
  /opt/venv/bin/python -c "import torch,sys; print('[boot] after unset', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count(), flush=True); sys.exit(0 if torch.cuda.is_available() else 1)" && torch_ok=1
fi
# A host that cannot run the cu130 build is unusable: warp/mujoco_warp needs a CUDA-13.0 driver and there is no other
# torch build that can drive it (d162). Fail now and let the orchestrator take another host - do NOT retry with cu128.
[ "$torch_ok" = "1" ] || { nvidia-smi 2>&1 | head -5; fail "cuda check (cu130 torch does not run on this host)"; exit 1; }
/opt/venv/bin/python -c "import torch; torch.zeros(1, device='cuda'); print('[boot] early cuda smoke ok', flush=True)" \
    || { fail "cuda smoke (early)"; exit 1; }
boot_log; echo "[boot] stage=deps"
retry /opt/venv/bin/uv pip install --python /opt/venv/bin/python -q --override /tmp/torch_override.txt -e ./mjlab_centering \
    || { fail "uv install (override $TORCH_PIN)"; exit 1; }
# mjlab at our pinned commit needs torch>=2.14, and the resolver then picks the cu132 build (2.15), which requires a
# CUDA 13.2 driver. Round 4's hosts had one; a community host with an older driver does not. Rather than burn the pod,
# fall back to the newest build that a CUDA 13.0 driver can run (torch 2.14.1+cu130) and record which build is in use.
cu() { /opt/venv/bin/python -c "import torch,sys; print('[boot] torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True); sys.exit(0 if torch.cuda.is_available() else 1)"; }
if ! cu; then
  # Nothing to fall back to: warp needs CUDA >= 13.0 and cu130 is the only build that can drive it (d162).
  nvidia-smi 2>&1 | head -5; fail "cuda check (cu130 build unusable on this host)"; exit 1
fi
/opt/venv/bin/python - <<'PY' || { fail "cuda smoke"; exit 1; }
import torch
torch.zeros(1, device='cuda')
print('[boot] cuda smoke ok', flush=True)
PY
# Encoder self-test (d1012): the SSP/RFF arms need sspspace for the phase matrix, and a missing dependency there fails every
# encoder job at env construction while raw jobs pass (2026-11-06). Catch it here, in seconds, not in a campaign.
/opt/venv/bin/python - <<'PY' || { fail "encoder self-test (sspspace / HexSSP)"; exit 1; }
from mjlab_centering.ssp_torch import HexSSP
import torch
for n in (2, 4):
    e = HexSSP(3, n_rotates=n, n_scales=n)
    out = e.encode(torch.zeros(4, 3))
    assert tuple(out.shape) == (4, e.out_dim), out.shape
    r = HexSSP(3, n_rotates=n, n_scales=n, rff=True, rff_dim=e.out_dim)
    assert tuple(r.encode(torch.zeros(4, 3)).shape) == (4, r.out_dim), r.out_dim
    print(f'[boot] encoder self-test ok: {n}x{n} hex {e.out_dim}, rff {r.out_dim}', flush=True)
PY
# Warp's JIT kernel cache (~/.cache/warp, ~50 MB) is restored from R2 before the warm-up. The key includes the torch
# build, so a pod that fell back to cu130 never inherits cu132 kernels. A cold warm-up costs ~20 min of billed GPU time
# per pod, and preemptible community pods would pay it again on every replacement.
echo "[boot] $(date -u +%H:%M:%S) step=warp-restore-begin"
/opt/venv/bin/python - <<'PYX' || echo "[boot] warp cache restore skipped"
import os, tarfile, io, boto3, torch
from botocore.config import Config as _C  # d1025: the missing import made every restore fail silently (2>/dev/null)
e = os.environ
key = "mjlab_ssp_centering/_cache/warp_%s.tgz" % torch.__version__.replace("+", "_")
s3 = boto3.client('s3', endpoint_url='https://%s.r2.cloudflarestorage.com' % e['R2_ACCOUNT_ID'],
                  aws_access_key_id=e['R2_ACCESS_KEY_ID'], aws_secret_access_key=e['R2_SECRET_ACCESS_KEY'],
                  config=_C(connect_timeout=10, read_timeout=60, retries={'max_attempts': 3}))
buf = io.BytesIO()
s3.download_fileobj(e['R2_BUCKET'], key, buf)
buf.seek(0)
with tarfile.open(fileobj=buf) as t:
    t.extractall(os.path.expanduser("~/.cache"))
open("/tmp/warp_cache_restored", "w").close()
print('[boot] warp cache restored from', key, flush=True)
PYX
# Warm the mujoco-warp JIT kernel cache once per pod: the first run on a cold cache pays ~20 min of compilation,
# every later run on the same pod is ~1 min per 25 iterations. Shapes must match the campaign settings.
boot_log; echo "[boot] stage=warmup"
echo "[boot] $(date -u +%H:%M:%S) step=warmup-begin"
/opt/venv/bin/python -m mjlab_centering.train --job-id warmup --center none --iters 2 --num-envs 2048 \
    --eval-envs 256 --eval-every 1 --eval-steps 5 --final-steps 5 --final-shift-steps 5 \
    --out /tmp/warmup --quiet || echo "[boot] warmup failed (continuing)"
echo "[boot] $(date -u +%H:%M:%S) step=warp-push-begin"
/opt/venv/bin/python - <<'PYX' || echo "[boot] warp cache push skipped"
import os, tarfile, io, boto3, torch
e = os.environ
key = "mjlab_ssp_centering/_cache/warp_%s.tgz" % torch.__version__.replace("+", "_")
root = os.path.expanduser("~/.cache/warp")
if os.path.isdir(root) and not os.path.exists("/tmp/warp_cache_restored"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(root, arcname="warp")
    buf.seek(0)
    boto3.client('s3', endpoint_url='https://%s.r2.cloudflarestorage.com' % e['R2_ACCOUNT_ID'],
                 aws_access_key_id=e['R2_ACCESS_KEY_ID'], aws_secret_access_key=e['R2_SECRET_ACCESS_KEY']
                 ).upload_fileobj(buf, e['R2_BUCKET'], key)
    print('[boot] warp cache pushed to', key, flush=True)
PYX
echo "[boot] $(date -u +%H:%M:%S) step=queue-begin"
touch /tmp/boot_ok
boot_log; echo "[boot] stage=queue"
if [ -n "${PODRUN_PROCS:-}" ]; then
  # podrun mode: N concurrent training processes on this one GPU (mothers/branches need it), artefacts mirrored to R2.
  boot_log; echo "[boot] stage=manifest"
  retry timeout 240 /opt/venv/bin/python - <<'PY'   # bounded and retried: a stalled download used to hang the boot for ever
import os, boto3
from botocore.config import Config as _C
e = os.environ
s3 = boto3.client('s3', endpoint_url='https://%s.r2.cloudflarestorage.com' % e['R2_ACCOUNT_ID'],
                  aws_access_key_id=e['R2_ACCESS_KEY_ID'], aws_secret_access_key=e['R2_SECRET_ACCESS_KEY'],
                  config=_C(connect_timeout=10, read_timeout=60, retries={'max_attempts': 3}))
s3.download_file(e['R2_BUCKET'], 'mjlab_ssp_centering/%s/manifest.json' % e['CAMPAIGN'], '/project/manifest.json')
PY
  echo "[boot] $(date -u +%H:%M:%S) step=podrun-pilot-begin"
  cd /project/mjlab_centering && /opt/venv/bin/python podrun.py --manifest /project/manifest.json \
      --out "/project/results/$CAMPAIGN" --procs "$PODRUN_PROCS" --r2 "$CAMPAIGN"
  # Keep-alive (2026-11-06, user's instruction): a captured community pod is the scarcest resource in this project, so a pod
  # that drains its queue does NOT delete itself. It polls `<campaign>/queue.json`; when a new queue appears it runs it on
  # the same warm installation (torch, mjlab, warp kernels, all paid for already). It deletes itself on a `<campaign>/TERMINATE`
  # marker or after KEEP_MAX_H hours (default 3), so a forgotten pod cannot spend indefinitely.
  if [ "${KEEP_ALIVE:-1}" = "1" ]; then   # default ON: a pod does not delete itself unless KEEP_ALIVE=0
    touch /tmp/boot_ok
    KEEP_MAX_H="${KEEP_MAX_H:-3}"
    END_TS=$(( $(date +%s) + KEEP_MAX_H * 3600 ))
    LAST_ETAG=""
    echo "[boot] queue drained; keeping the pod warm for up to ${KEEP_MAX_H}h (idle cost ${PODRUN_PRICE:-?} $/h), polling $CAMPAIGN/queue.json"
    boot_log
    while [ "$(date +%s)" -lt "$END_TS" ]; do
      boot_log   # one small push per minute: a warm pod must never look silent (d177's lesson)
      NEXT=$(timeout 120 /opt/venv/bin/python - "$CAMPAIGN" <<'PYQ' 2>/dev/null
import json, os, sys, boto3
from botocore.config import Config as _C  # was missing: the poller raised NameError and 2>/dev/null hid it
e = os.environ
c = sys.argv[1]
s3 = boto3.client('s3', endpoint_url='https://%s.r2.cloudflarestorage.com' % e['R2_ACCOUNT_ID'],
                  aws_access_key_id=e['R2_ACCESS_KEY_ID'], aws_secret_access_key=e['R2_SECRET_ACCESS_KEY'],
                  config=_C(connect_timeout=10, read_timeout=60, retries={'max_attempts': 3}))
def body(key):
    try:
        return s3.get_object(Bucket=e['R2_BUCKET'], Key=key)['Body'].read()
    except Exception:
        return None
if body('mjlab_ssp_centering/%s/TERMINATE' % c) is not None:
    print('TERMINATE'); raise SystemExit(0)
b = body('mjlab_ssp_centering/%s/queue.json' % c)
print('' if b is None else 'QUEUE ' + b.decode())
PYQ
)
      case "$NEXT" in
        TERMINATE) echo "[boot] TERMINATE marker; deleting pod"; break ;;
        QUEUE\ *)
          echo "[boot] new queue received; running it on this pod"
          printf '%s' "${NEXT#QUEUE }" > /project/manifest_next.json
          /opt/venv/bin/python - <<'PYM' || echo "[boot] queue manifest rejected"
import json, sys
d = json.load(open('/project/manifest_next.json'))
assert d.get('jobs'), 'queue has no jobs'
print('[boot] queue jobs:', len(d['jobs']), 'campaign:', d.get('campaign'))
PYM
          /opt/venv/bin/python podrun.py --manifest /project/manifest_next.json \
              --out "/project/results/$CAMPAIGN" --procs "$PODRUN_PROCS" --r2 auto
          echo "[boot] queue finished; back to waiting"
          boot_log
          ;;
        *) sleep 60 ;;
      esac
    done
  fi
  echo "[boot] deleting pod"
  /opt/venv/bin/python - <<'PY' || true
import os, urllib.request
pod, key = os.environ.get('RUNPOD_POD_ID'), os.environ.get('RUNPOD_API_KEY')
req = urllib.request.Request('https://api.runpod.io/v2/pods/%s' % pod, method='DELETE',
                             headers={'Authorization': 'Bearer ' + key, 'User-Agent': 'curl/8.5.0'})
urllib.request.urlopen(req, timeout=60)
PY
  sleep 3600
fi
/opt/venv/bin/python -m mjlab_centering.worker --campaign "$CAMPAIGN" --self-delete --results-dir /project/results
echo "[boot] worker exited"
"""


CUDA_OK_SRC = """def ok(v):
    try:
        maj, mi = (v.split('.') + ['0'])[:2]
        return (int(maj), int(mi)) >= (13, 0)
    except Exception:
        return False"""

BOOTSTRAP = BOOTSTRAP.replace("__CUDA_OK__", CUDA_OK_SRC)


def cuda_ok(v: str) -> bool:
  """Driver-side CUDA a torch cu130 wheel can use. Same rule the pod bootstrap applies before installing anything."""
  try:
    maj, mi = (v.split(".") + ["0"])[:2]
    return (int(maj), int(mi)) >= (13, 0)
  except Exception:  # noqa: BLE001
    return False


def make_code_tarball() -> bytes:
  buf = io.BytesIO()

  def keep(ti):
    n = ti.name
    if "__pycache__" in n or n.endswith((".pyc", ".pt", ".jsonl")) or "/.venv/" in n or "/results/" in n:
      return None
    return ti

  with tarfile.open(fileobj=buf, mode="w:gz") as t:
    t.add(str(ROOT), arcname="mjlab_centering", filter=keep)
    if DOCS.exists():
      t.add(str(DOCS), arcname="docs/mjlab_centering", filter=keep)
  return buf.getvalue()


def ship(campaign: str, jobs: list[dict]) -> str:
  data = make_code_tarball()
  code_key = f"{campaign}/_code/mjlab_centering.tar.gz"
  r2.client().put_object(Bucket=r2.bucket(), Key=r2.key(code_key), Body=data)
  r2.put_json(f"{campaign}/manifest.json", {"campaign": campaign, "jobs": jobs})
  return code_key


def _pending(campaign: str, jobs: list[dict]) -> list[dict]:
  done = r2.done_set(campaign)
  return [j for j in jobs if j["job_id"] not in done]


def _our_pods(prefix: str) -> list[dict]:
  """Pods of this campaign that are actually *up*. An exited/terminated pod stays in the list until it is removed, and
  counting it as live stops the supervisor from ever capturing a replacement (measured 2026-11-06: `live_pods=3` with
  nothing running, and no create attempted for half an hour)."""
  return [p for p in pods.list_pods()
          if str(p.get("name", "")).startswith(prefix) and str(p.get("desiredStatus", "RUNNING")).upper() == "RUNNING"
          and str(p.get("runtimeStatus", "running")).lower() not in ("exited", "terminated", "failed")]


def _state_path(campaign: str) -> Path:
  p = ROOT / "results" / f"orch_{campaign}.json"
  p.parent.mkdir(parents=True, exist_ok=True)
  return p


def _load_state(campaign: str) -> dict:
  p = _state_path(campaign)
  return json.loads(p.read_text()) if p.exists() else {"pods": {}, "spend": 0.0}


def _save_state(campaign: str, st: dict) -> None:
  _state_path(campaign).write_text(json.dumps(st, indent=1))


def _account(st: dict, live: list[dict], default_price: float, now: float, absent_min: float = 15.0) -> float:
  """Fold live pods' hours into cumulative spend, at each pod's own $/hour. Disappeared pods are closed out."""
  live_ids = {p.get("id") for p in live}
  for p in live:
    pid = p.get("id")
    rec = st["pods"].setdefault(pid, {"first": now, "last": now, "price": default_price})
    rec["last"] = now
  for pid, rec in st["pods"].items():
    if pid not in live_ids and "closed_ts" not in rec:
      rec.setdefault("missing_since", now)
      if now - rec["missing_since"] > 60.0 * absent_min:  # the pod list lags; only believe a disappearance after 15 min
        rec["closed_ts"] = now
    elif pid in live_ids:
      rec.pop("missing_since", None)
  total = 0.0
  for pid, rec in st["pods"].items():
    end = rec.get("closed_ts") or rec["last"]
    h = max(end - rec["first"], 0.0) / 3600.0
    total += h * float(rec.get("price", default_price))
  return total


SECURE_GPUS = "NVIDIA GeForce RTX 4090:0.89:SECURE,NVIDIA A40:0.59:SECURE"  # only with --allow-secure (d113)


def choose_candidates(gpus: list[tuple[str, float, str]], rounds: int, community_rounds: int, no_stock_s: float,
                      no_stock_limit_s: float, allow_secure: bool) -> tuple[list[tuple[str, float, str]], bool]:
  """Which GPUs to try next, and whether the user must be asked about secure cloud.

  Community is always tried first. Once `community_rounds` full passes failed, or no pod has been alive for
  `no_stock_limit_s` seconds, secure is *needed*; it is only returned when `allow_secure` (granted by the user), else the
  caller raises the NEEDS_SECURE_APPROVAL flag and keeps trying community.
  """
  comm = [g for g in gpus if g[2].upper() == "COMMUNITY"]
  sec = [g for g in gpus if g[2].upper() != "COMMUNITY"]
  needs = rounds >= community_rounds or (no_stock_s >= no_stock_limit_s > 0)
  if needs and allow_secure and sec:
    return sec, True
  return comm or gpus, needs


def stock_candidates(gpus: list[tuple[str, float, str]], stock: list[dict],
                     max_price: float | None = None) -> list[tuple[str, float, str, str]]:
  """Restrict the configured candidates to SKUs that `gpu list` reports in stock, pinned to a datacentre that has it.

  Community stock flashes for seconds (d122); trying only what is actually available turns each pass into one cheap stock
  read instead of a blind cycle of six doomed creates.
  """
  by_name = {g[0]: g for g in gpus if g[2].upper() == "COMMUNITY"}
  out = []
  for r in stock:  # already cheapest-first
    g = by_name.get(r["gpuId"])
    if g is None:
      continue
    if max_price is not None and float(r["price"]) > max_price:
      continue  # never let an opportunistic SKU (A100/H200) into a campaign budgeted for a $0.34-0.79 GPU
    out.append((g[0], g[1], g[2], r["dcs"][0]))
  return out


def _parse_gpus(spec: str) -> list[tuple[str, float, str]]:
  out = []
  for part in spec.split(","):
    part = part.strip()
    if not part:
      continue
    name, price, cloud = part.rsplit(":", 2)
    out.append((name.strip(), float(price), cloud.strip()))
  return out


def plan(campaign: str, jobs: list[dict], price: float, sec_per_iter: float | None) -> dict:
  by = {}
  for j in jobs:
    by[j["block"]] = by.get(j["block"], 0) + 1
  out = {"campaign": campaign, "jobs": len(jobs), "by_block": by}
  if sec_per_iter:
    # per-run seconds: iterations x steps x envs / fps; fps measured by the benchmark
    pass
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--manifest", default=None, help="local manifest json (from jobs.py)")
  ap.add_argument("--launch", type=int, default=0, help="number of concurrent pods (0 = plan only)")
  ap.add_argument(
    "--gpus",
    default=(
      # names exactly as `runpodctl gpu list` prints them, cheapest-with-stock first
      # canonical gpuIds from `runpodctl gpu list` (its displayName column is not accepted by create)
      # community first (d101): a lost community pod costs at most one run, because every job mirrors to R2
      "NVIDIA GeForce RTX 4090:0.34:COMMUNITY,"
      "NVIDIA GeForce RTX 3090:0.22:COMMUNITY,"
      "NVIDIA RTX 4000 Ada Generation:0.20:COMMUNITY,"
      "NVIDIA RTX A6000:0.33:COMMUNITY,"
      "NVIDIA L40:0.69:COMMUNITY,"
      "NVIDIA L40S:0.79:COMMUNITY,"
      "NVIDIA GeForce RTX 5090:0.69:COMMUNITY,"
      # SECURE entries are no longer in the default list (d113): they are only added by --allow-secure, which the user
      # grants explicitly. Measured when they were used: 4090 SECURE $0.075/1000 it, A40 SECURE $0.091/1000 it.
    ),
    help="comma-separated name:price:cloud fallbacks, tried in order when stock is unavailable",
  )
  ap.add_argument("--price", type=float, default=0.34, help="$/GPU-hour used for pods with no recorded price")
  ap.add_argument("--image", default="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404")
  ap.add_argument("--min-cuda", default=None, help="optional driver-side CUDA floor")
  ap.add_argument("--hard-cap", type=float, default=25.0)
  ap.add_argument("--soft-cap", type=float, default=20.0)
  ap.add_argument("--poll", type=float, default=60.0)
  ap.add_argument("--absent-min", type=float, default=15.0,
                  help="minutes a pod must be missing from the pod list before it is closed out and replaced")
  ap.add_argument("--max-pods-created", type=int, default=24)
  ap.add_argument("--community-rounds", type=int, default=3,
                  help="full passes over the COMMUNITY candidates before SECURE is even attempted (d101)")
  ap.add_argument("--podrun-procs", type=int, default=0,
                  help="run `podrun --procs N` on each pod instead of the claim-based worker (needed for shock branching)")
  ap.add_argument("--keep-pods", action="store_true",
                  help="do not delete pods when the queue drains: they go warm-idle and poll <campaign>/queue.json for the "
                       "next manifest (a captured community pod is the scarcest resource here; see analysis/r7_publish_queue.py)")
  ap.add_argument("--keep-hours", type=float, default=3.0,
                  help="with --keep-pods, how long a warm pod waits for the next queue before deleting itself")
  ap.add_argument("--allow-secure", action="store_true",
                  help="append SECURE candidates and use them once the community rounds / no-stock window are exhausted; "
                       "NEVER pass this without the user's go-ahead (d113)")
  ap.add_argument("--no-stock-min", type=float, default=60.0,
                  help="minutes without any live pod after which a NEEDS_SECURE_APPROVAL flag is raised")
  ap.add_argument("--max-price", type=float, default=1.05,
                  help="with --stock-first, ignore community SKUs above this $/h (keeps a $0.34-budgeted campaign off a $1.39 GPU)")
  ap.add_argument("--stock-first", action="store_true",
                  help="only try community SKUs that `gpu list` currently reports in stock, pinned to that datacentre (d122)")
  ap.add_argument("--retry-wait", type=float, default=300.0,
                  help="seconds between failed community passes: a 'round' (d113) is one pass over the candidates")
  a = ap.parse_args()

  mpath = a.manifest or str(ROOT / "manifests" / f"{a.campaign}.json")
  jobs = json.loads(Path(mpath).read_text())["jobs"]
  gpus = _parse_gpus(a.gpus + (("," + SECURE_GPUS) if a.allow_secure else ""))
  print(json.dumps(plan(a.campaign, jobs, a.price, None), indent=1), flush=True)
  print(f"[orch] gpu preference: {gpus}", flush=True)
  if a.launch <= 0:
    return

  code_key = ship(a.campaign, jobs)
  print(f"[orch] code+manifest shipped ({len(jobs)} jobs) to {code_key}", flush=True)

  prefix = f"mjc-{a.campaign}"
  env = {
    "CAMPAIGN": a.campaign,
    "R2_ACCOUNT_ID": r2.load_env()["R2_ACCOUNT_ID"],
    "R2_ACCESS_KEY_ID": r2.load_env()["R2_ACCESS_KEY_ID"],
    "R2_SECRET_ACCESS_KEY": r2.load_env()["R2_SECRET_ACCESS_KEY"],
    "R2_BUCKET": r2.bucket(),
    "RUNPOD_API_KEY": pods._key(),
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "MUJOCO_GL": "disable",
  }
  if a.podrun_procs:
    env["PODRUN_PROCS"] = str(a.podrun_procs)
  if a.keep_pods:
    env["KEEP_ALIVE"] = "1"
    env["KEEP_MAX_H"] = str(a.keep_hours)
    env["PODRUN_PRICE"] = f"{a.price:g}"
  # Deliver the bootstrap through an env var (base64) so nested quoting cannot break the docker args string.
  import base64

  env["CODE_KEY"] = r2.key(code_key)
  env["BOOT_B64"] = base64.b64encode(BOOTSTRAP.encode()).decode()
  docker_args = (
    "bash -lc 'echo $BOOT_B64 | base64 -d > /tmp/boot.sh && bash /tmp/boot.sh 2>&1 | tee /tmp/boot.log'"
  )

  st = _load_state(a.campaign)
  created = 0
  attempt = int(st.get("attempt", 0))
  while True:
    # Decision 91: re-read the job list every loop. Holding it from startup made the orchestrator declare a campaign
    # drained (and delete its pods mid-run) after jobs had been appended to the manifest by another process.
    try:
        jobs = json.loads(Path(mpath).read_text())["jobs"]
    except Exception as e:
      print(f"[orch] manifest reload failed ({e!r}); keeping the previous list", flush=True)
    live = _our_pods(prefix)
    pending = _pending(a.campaign, jobs)
    spend = _account(st, live, a.price, time.time(), a.absent_min)
    st["spend"] = spend
    _save_state(a.campaign, st)
    print(
      f"[orch] {time.strftime('%H:%M:%S')} pending={len(pending)} live_pods={len(live)} spend=${spend:.2f}",
      flush=True,
    )
    if not pending and a.keep_pods and len(live) >= a.launch:
      # Warm pod(s) exist and we are at the wanted concurrency: leave them on their queue poll (they self-delete on
      # TERMINATE or after --keep-hours). With fewer than --launch pods the code below keeps capturing, so a published
      # queue is worked by several warm pods, not one.
      print(f"[orch] queue drained; keeping {len(live)} pod(s) warm "
            f"(polling <campaign>/queue.json; they self-delete after {a.keep_hours:g} h or on a TERMINATE marker)", flush=True)
      time.sleep(a.poll)
      continue
    if not pending and a.keep_pods:
      # Queue drained and nothing is live: still capture one pod, otherwise a published queue would have no consumer.
      print("[orch] queue drained and no pod is live; capturing one warm pod for the next queue", flush=True)
    if not pending and not a.keep_pods:
      print("[orch] queue drained; deleting pods", flush=True)
      for p in live:
        try:
          pods.delete_pod(p["id"])
          r2.release_claims_of(a.campaign, p["id"])
        except Exception as e:
          print(f"[orch] delete {p.get('id')} failed: {e!r}", flush=True)
      break
    if spend >= a.hard_cap:
      print(f"[orch] HARD CAP ${a.hard_cap} reached; deleting pods", flush=True)
      for p in live:
        try:
          pods.delete_pod(p["id"])
          r2.release_claims_of(a.campaign, p["id"])
        except Exception:
          pass
      break
    min_pri = min((j.get("priority", 9) for j in pending), default=9)
    stop_for_soft = spend >= a.soft_cap and min_pri >= 3
    # the API is the truth for concurrency: a state entry whose pod is gone must not hold a slot (d172/d176). The
    # 15-minute grace in _account is for *billing*, not for deciding whether a replacement is needed.
    live_ids = {p["id"] for p in live}
    live_state = [pid for pid, rec in st["pods"].items() if "closed_ts" not in rec and pid in live_ids]
    failed_pass = False
    if not stop_for_soft and len(live_state) < a.launch and created < a.max_pods_created:
      name = f"{prefix}-{created}"
      rounds = int(st.get("comm_rounds", 0))
      if not live_state:  # stock timer: how long have we wanted a pod and had none (d113: ask after ~60 min)
        st.setdefault("no_stock_since", time.time())
      else:
        st.pop("no_stock_since", None)
      no_stock_s = time.time() - st["no_stock_since"] if "no_stock_since" in st else 0.0
      cands, needs_secure = choose_candidates(gpus, rounds, a.community_rounds, no_stock_s, a.no_stock_min * 60.0,
                                              a.allow_secure)
      if needs_secure and not a.allow_secure:
        flag = _state_path(a.campaign).with_suffix(".NEEDS_SECURE_APPROVAL")
        if not flag.exists():
          flag.write_text(json.dumps({"campaign": a.campaign, "rounds": rounds, "no_stock_min": no_stock_s / 60.0,
                                      "ts": time.time()}))
        print("[orch] NEEDS_SECURE_APPROVAL: community stock unavailable; waiting, secure NOT used (ask the user)",
              flush=True)
      if a.stock_first:
        try:
          live4 = stock_candidates(gpus, pods.community_stock(), a.max_price)
        except Exception as e:
          print(f"[orch] stock read failed ({e!r}); falling back to the configured list", flush=True)
          live4 = []
        cheap_all = [g for g in gpus if g[2].upper() == "COMMUNITY" and g[1] <= a.max_price] or \
                    [g for g in gpus if g[2].upper() == "COMMUNITY"]
        extra = [(*cheap_all[attempt % len(cheap_all)], None)] if cheap_all else []
        if live4:
          # the listing is optimistic: it advertises SKUs that then refuse a create, so every pass also gets one
          # blind attempt on a rotating cheap SKU whose listing said "none" (d123)
          cands = live4 + [e for e in extra if e[0] not in {c[0] for c in live4}]
        else:
          # The stock listing lags reality in both directions (it advertises SKUs that refuse a create, and it can be
          # empty while a host is briefly free). So an empty read still gets ONE cheap blind attempt, rotating through
          # the configured community SKUs, rather than skipping the pass (d123).
          cheap = [g for g in gpus if g[2].upper() == "COMMUNITY" and g[1] <= a.max_price] or \
                  [g for g in gpus if g[2].upper() == "COMMUNITY"]
          cands = [(*cheap[attempt % len(cheap)], None)] if cheap else []
          if cands:
            print(f"[orch] stock listing empty; one blind attempt on {cands[0][0]}", flush=True)
          else:
            failed_pass = True
      for _ in range(len(cands)):
        gpu_name, gpu_price, gpu_cloud, dc = cands[attempt % len(cands)] if len(cands[0]) == 4 else (*cands[attempt % len(cands)], None)
        try:
          env["GPU_NAME"] = gpu_name
          pid = pods.create_pod(
            name, a.image, gpu_name, docker_args, env,
            cloud=gpu_cloud, min_cuda=a.min_cuda, gpu_count=1, data_center_ids=None,  # no DC pin: a stale "in stock"
            # listing pins the create to a datacentre that then refuses it (d171). The API is authoritative at create time.
          )
          created += 1
          print(f"[orch] launched {pid} ({name}) on {gpu_name} {gpu_cloud}{' ' + dc if dc else ''} ${gpu_price}/h", flush=True)
          st["pods"][pid] = {"first": time.time(), "last": time.time(), "price": gpu_price,
                             "gpu": gpu_name, "cloud": gpu_cloud}
          _save_state(a.campaign, st)
          failed_pass = False
          st["comm_rounds"] = 0  # a successful launch ends the failure streak
          st.pop("no_stock_since", None)
          break
        except Exception as e:
          attempt += 1
          st["attempt"] = attempt
          if cands and (attempt % len(cands)) == 0 and rounds < a.community_rounds:
            rounds += 1
            st["comm_rounds"] = rounds
            if rounds >= a.community_rounds:
              print(f"[orch] {rounds} community rounds exhausted", flush=True)
          failed_pass = True
          _save_state(a.campaign, st)
          print(f"[orch] create failed on {gpu_name} ({gpu_cloud}): {e}", flush=True)
    if stop_for_soft:
      print("[orch] soft cap reached and only tier>=3 work remains; waiting for running jobs", flush=True)
    time.sleep(a.retry_wait if failed_pass else a.poll)  # a community "round" is one pass, spaced by --retry-wait


if __name__ == "__main__":
  sys.exit(main())
