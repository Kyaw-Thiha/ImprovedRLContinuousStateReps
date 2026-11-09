# mjlab_centering

Reward centering (and, deferred, SSP input encoding) for PPO on a **continuing** GPU-parallel mjlab robotics
task. Design, matrix, gates and budget: [`../docs/mjlab_centering/00-plan.md`](../docs/mjlab_centering/00-plan.md).
Frozen predictions: [`../docs/mjlab_centering/01-preregistration.md`](../docs/mjlab_centering/01-preregistration.md).

## Environment

mjlab 1.6.0 needs torch >= 2.14 and a Python 3.10-3.13 interpreter, so this package has its own venv. Nothing is
imported from the parent repo's `rl/` or `scripts/`.

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -p .venv -e . "sspspace @ git+https://github.com/ctn-waterloo/sspspace@f3f68570e9488d32ce3e5dfaca6de5b0ccf5e91b"
pytest tests -q
```

## One run (local)

```bash
.venv/bin/python -m mjlab_centering.train --center value --gamma 0.999 --offset 1.0 \
    --iters 5 --num-envs 32 --eval-envs 8 --eval-every 2 --out results/smoke
```

Arms: `--center none|value|simple`, `--value-clip false`, `--ret-norm meanstd`, `--reward-scale std`,
`--shock-kind offset|action_rate --shock-at N`, `--penalty`, `--p-reset`.

## Campaign

```bash
# matrix -> manifest (iters comes from the plateau gate; --blocks/--seeds implement the cut order)
.venv/bin/python -m mjlab_centering.jobs --eta <winner> --iters <plateau> --tag c1

# pods: ship code+manifest, launch, monitor, enforce the caps
.venv/bin/python orchestrator.py --campaign c1-xxxxxxxx --manifest manifests/c1-xxxxxxxx.json \
    --launch 3 --soft-cap 20 --hard-cap 25

# results
.venv/bin/python analysis/derive.py --campaign c1-xxxxxxxx --pull
```

R2 layout (all under `mjlab_ssp_centering/`): `<campaign>/manifest.json`, `<campaign>/_code/`,
`<campaign>/claims/<job>.json`, `<campaign>/runs/<job>/{iter.jsonl,eval.jsonl,final.json,model_final.pt,DONE}`.

Every claim is an atomic conditional put; a dead pod's claim is taken over once its heartbeat is 10 minutes old;
completion is decided from `DONE` markers, never from the pod list.
