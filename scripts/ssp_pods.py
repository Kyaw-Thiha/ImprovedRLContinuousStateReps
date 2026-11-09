"""List and delete ``ssp-*`` RunPod pods (the durable form of the old /tmp/reap_pods.py).

Only pods whose name starts with ``ssp-`` are ever touched; the account's ``eval-v3-*`` pods are left alone.

    .venv/bin/python scripts/ssp_pods.py list
    .venv/bin/python scripts/ssp_pods.py kill <name|id> [<name|id> ...]
    .venv/bin/python scripts/ssp_pods.py kill --prefix p14-delay
"""

import argparse
import json
import pathlib
import re
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import launch_pilot_pod as lp  # noqa: E402


def api_key() -> str:
    env = dict(re.findall(r'^([A-Z_]+)\s*=\s*"?([^"\n]+)"?', (ROOT / ".env").read_text(), re.M))
    return env["RUNPOD_API_KEY"]


def _req(url, key, method="GET"):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "User-Agent": lp.UA,
                                               "Accept": "application/json"}, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=30).read() or b"[]")


def pods(key):
    data = _req(lp.API, key)
    data = data if isinstance(data, list) else data.get("pods", [])
    return [p for p in data if str(p.get("name", "")).startswith("ssp-")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "kill"])
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--prefix")
    a = ap.parse_args()
    key = api_key()
    live = pods(key)
    if a.cmd == "list":
        print(f"{len(live)} ssp-* pods")
        for p in live:
            print(f"  {p['id']:<20} {p['name']:<42} {p.get('desiredStatus', '?')}")
        return
    wanted = set(a.targets)
    victims = [p for p in live if (p["id"] in wanted or p["name"] in wanted
                                   or (a.prefix and p["name"].startswith(a.prefix)))]
    if not victims:
        sys.exit(f"no ssp-* pod matched {sorted(wanted) or a.prefix}")
    for p in victims:
        try:
            _req(f"{lp.API}/{p['id']}", key, method="DELETE")
            print(f"deleted {p['name']} ({p['id']})")
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {p['name']} ({p['id']}): {e}")
    print("remaining:", [p["name"] for p in pods(key)])


if __name__ == "__main__":
    main()
