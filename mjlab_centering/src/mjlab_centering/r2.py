"""R2 helpers. Every key this project writes lives under PREFIX = "mjlab_ssp_centering/"."""

from __future__ import annotations

import json
import os
import uuid
import time
from pathlib import Path

import boto3
import botocore
from botocore.config import Config

PREFIX = "mjlab_ssp_centering"
_ENV_KEYS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[3]


def load_env() -> dict:
  env = {k: os.environ[k] for k in _ENV_KEYS if k in os.environ}
  if len(env) == len(_ENV_KEYS):
    return env
  p = _repo_root() / ".env"
  if p.exists():
    for line in p.read_text().splitlines():
      line = line.strip()
      if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        if k in _ENV_KEYS:
          env.setdefault(k, v.strip().strip('"').strip("'"))
  return env


_CLIENT = None


def client():
  global _CLIENT
  if _CLIENT is not None:
    return _CLIENT
  e = load_env()
  _CLIENT = boto3.client(
    "s3",
    endpoint_url=f"https://{e['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=e["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=e["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
    # Aggressive timeouts: an R2 upload must never stall a training loop. The sink pushes asynchronously anyway.
    config=Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 1, "mode": "standard"},
                  max_pool_connections=4),
  )
  return _CLIENT


def bucket() -> str:
  return load_env()["R2_BUCKET"]


def key(rel: str) -> str:
  rel = rel.lstrip("/")
  return f"{PREFIX}/{rel}"


def put_json(rel: str, obj) -> None:
  client().put_object(Bucket=bucket(), Key=key(rel), Body=json.dumps(obj, default=float).encode())


def get_json(rel: str, default=None):
  try:
    r = client().get_object(Bucket=bucket(), Key=key(rel))
    return json.loads(r["Body"].read())
  except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
      return default
    raise


def retry(fn, attempts: int = 8, delay: float = 5.0):
  """Call fn until it succeeds; used for the end-of-run artefacts, where a lost upload means a lost run."""
  last = None
  for i in range(attempts):
    try:
      return fn()
    except Exception as e:  # noqa: BLE001 - the caller decides what a failure means
      last = e
      time.sleep(delay * (1 + i))
  raise last


def upload_file(local: str, rel: str) -> None:
  """Single PUT of the file's current bytes (no multipart machinery, no local-path uploader retries)."""
  with open(local, "rb") as f:
    body = f.read()
  client().put_object(Bucket=bucket(), Key=key(rel), Body=body)


def download_file(rel: str, local: str) -> bool:
  try:
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    client().download_file(bucket(), key(rel), local)
    return True
  except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
      return False
    raise


def list_keys(rel: str = "") -> list[str]:
  out, tok = [], None
  while True:
    kw = dict(Bucket=bucket(), Prefix=key(rel))
    if tok:
      kw["ContinuationToken"] = tok
    r = client().list_objects_v2(**kw)
    out += [o["Key"] for o in r.get("Contents", [])]
    if not r.get("IsTruncated"):
      return out
    tok = r.get("NextContinuationToken")


def delete_key(rel: str) -> None:
  """Delete one object. Idempotent: deleting a key that is already gone is not an error."""
  client().delete_object(Bucket=bucket(), Key=key(rel))


def exists(rel: str) -> bool:
  try:
    client().head_object(Bucket=bucket(), Key=key(rel))
    return True
  except botocore.exceptions.ClientError:
    return False


# ---------------------------------------------------------------- campaign state
def claim(campaign: str, job_id: str, pod: str, ttl_s: int = 600, verify_s: float = 1.5) -> bool:
  """Claim a job, then *verify* the claim by reading it back.

  R2's conditional put is not linearizable under concurrency: two simultaneous `IfNoneMatch: *` writers can both be
  accepted (observed in tests/test_r2_claims.py, where 8 concurrent claimers produced 2 winners). Left unhandled
  that double-assigns a job, and two workers then run and overwrite each other's artefacts. So the winner is decided
  by the read-back (last write wins): after writing, wait briefly and keep the claim only if the stored record is
  still ours.
  """
  c = client()
  k = key(f"{campaign}/claims/{job_id}.json")
  nonce = uuid.uuid4().hex
  body = json.dumps({"pod": pod, "ts": time.time(), "nonce": nonce}).encode()

  def _verify() -> bool:
    time.sleep(verify_s)
    cur = get_json(f"{campaign}/claims/{job_id}.json") or {}
    return cur.get("nonce") == nonce

  try:
    c.put_object(Bucket=bucket(), Key=k, Body=body, IfNoneMatch="*")
    return _verify()
  except botocore.exceptions.ClientError as e:
    if e.response["Error"]["Code"] not in ("PreconditionFailed", "412"):
      raise
  cur = get_json(f"{campaign}/claims/{job_id}.json") or {}
  if time.time() - float(cur.get("ts", 0)) < ttl_s:
    return False
  # stale claim: compare-and-swap on the ETag, then verify the same way
  try:
    head = c.head_object(Bucket=bucket(), Key=k)
    c.put_object(Bucket=bucket(), Key=k, Body=body, IfMatch=head["ETag"])
    return _verify()
  except botocore.exceptions.ClientError:
    return False


def heartbeat(campaign: str, job_id: str, pod: str) -> None:
  try:
    put_json(f"{campaign}/claims/{job_id}.json", {"pod": pod, "ts": time.time()})
  except Exception:
    pass


def release(campaign: str, job_id: str) -> None:
  try:
    client().delete_object(Bucket=bucket(), Key=key(f"{campaign}/claims/{job_id}.json"))
  except Exception:
    pass


def release_claims_of(campaign: str, pod: str) -> int:
  """Drop every claim held by a pod we are deleting, so its work is retaken immediately instead of after the TTL."""
  n = 0
  for k in list_keys(f"{campaign}/claims/"):
    rel = k.replace(f"{PREFIX}/", "", 1)
    cur = get_json(rel) or {}
    # podrun writes {"ts", "owner"}; the orchestrator's older claims used "pod". Accept both, or a dead pod's claim
    # blocks its jobs for the whole 4 h TTL (found 2026-11-06: `release_claims_of` silently released nothing).
    if pod in (cur.get("owner"), cur.get("pod")):
      client().delete_object(Bucket=bucket(), Key=k)
      n += 1
  return n


def done_set(campaign: str) -> set[str]:
  out = set()
  for k in list_keys(f"{campaign}/runs/"):
    if k.endswith("/DONE"):
      out.add(k.split("/runs/")[1].rsplit("/DONE", 1)[0])
  return out
