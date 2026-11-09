"""Exit when every job of a block is DONE (or the timeout passes), then print the block's per-cell summary.

    python analysis/wait_for_block.py --campaign night1-3e71b4 --block w2main --timeout 3300
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mjlab_centering import r2


def rows(rel: str) -> list[dict]:
  with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "x.jsonl"
    if not r2.download_file(rel, str(p)):
      return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--block", required=True)
  ap.add_argument("--timeout", type=float, default=3300)
  a = ap.parse_args()
  man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
  want = [j for j in man["jobs"] if j["block"] == a.block]
  t0 = time.time()
  while time.time() - t0 < a.timeout:
    done = [j for j in want if r2.exists(f"{a.campaign}/runs/{j['job_id']}/DONE")]
    if len(done) == len(want):
      break
    time.sleep(150)
  cells: dict[tuple, dict[int, tuple[float, float]]] = defaultdict(dict)
  for j in want:
    ev = rows(f"{a.campaign}/runs/{j['job_id']}/eval.jsonl")
    if not ev:
      continue
    tr = float(np.mean([e["track_lin_frac"] for e in ev[-5:]]))
    jt = float(np.mean([e["J_task"] for e in ev[-5:]]))
    cells[(j["center"], j["gamma"], j["offset"], j["penalty"])][j["seed"]] = (tr, jt, r2.exists(f"{a.campaign}/runs/{j['job_id']}/DONE"))
  print(f"block {a.block}: {len(want)} jobs, {sum(1 for j in want if r2.exists(f'{a.campaign}/runs/' + j['job_id'] + '/DONE'))} done")
  for k, seeds in sorted(cells.items()):
    tr = [v[0] for v in seeds.values()]
    jt = [v[1] for v in seeds.values()]
    print(f"  center={k[0]:5s} gamma={k[1]:g} offset={k[2]:g} p{k[3]:g}: n={len(seeds)} "
          f"track {np.mean(tr):.3f} J {np.mean(jt):.4f} walks {sum(t >= 0.5 for t in tr)}/{len(tr)}")
  for off in sorted({k[2] for k in cells}):
    v = cells.get(("value", 0.999, off, 10.0), {})
    n0 = cells.get(("none", 0.999, off, 10.0), {})
    shared = sorted(set(v) & set(n0))
    if shared:
      d = np.array([v[s][1] - n0[s][1] for s in shared])
      dt = np.array([v[s][0] - n0[s][0] for s in shared])
      print(f"  paired offset {off:g}: n={len(shared)} dJ {d.mean():+.4f} (se {d.std(ddof=1)/np.sqrt(len(d)):.4f}, "
            f"95% CI [{d.mean()-1.96*d.std(ddof=1)/np.sqrt(len(d)):+.4f}, {d.mean()+1.96*d.std(ddof=1)/np.sqrt(len(d)):+.4f}]) "
            f"dtrack {dt.mean():+.3f}")


if __name__ == "__main__":
  main()
