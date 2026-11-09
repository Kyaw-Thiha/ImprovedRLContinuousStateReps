"""Count `DONE` markers for a campaign's runs. Used by the keepers, where inline python quoting broke silently.

    python analysis/count_done.py --campaign r6g1queue-0ede5636      # prints an integer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def count_done(campaign: str) -> int:
  from mjlab_centering import r2
  c, b = r2.client(), r2.bucket()
  n, tok = 0, None
  while True:
    kw = dict(Bucket=b, Prefix=r2.key(f"{campaign}/runs/"), MaxKeys=1000)
    if tok:
      kw["ContinuationToken"] = tok
    resp = c.list_objects_v2(**kw)
    n += sum(1 for o in resp.get("Contents", []) if o["Key"].endswith("/DONE"))
    tok = resp.get("NextContinuationToken")
    if not tok:
      return n


def main(argv: list[str]) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  a = ap.parse_args()
  print(count_done(a.campaign))


if __name__ == "__main__":
  main(sys.argv[1:])
