"""Branching state: a checkpoint written mid-run restores weights, optimizer, lr and centering state exactly."""

from __future__ import annotations

import torch

from mjlab_centering.train import RunSpec, run


def _spec(**kw):
  base = dict(job_id="br", num_envs=16, eval_envs=0, final_steps=0, final_shift_steps=0, center="value",
              gamma=0.999, offset=1.0, seed=1000)
  base.update(kw)
  return RunSpec(**base)


def test_checkpoint_round_trip_restores_all_state(tmp_path):
  a = tmp_path / "a"
  run(_spec(iters=3, save_at=2), str(a), quiet=True)
  ck = torch.load(a / "ckpt_it2.pt", weights_only=False)
  assert ck["iter"] == 2 and "cc_state" in ck["alg"]
  # resume with iters == the checkpoint iteration: zero training steps, so model_final must equal the checkpoint
  b = tmp_path / "b"
  run(_spec(iters=2, resume=str(a / "ckpt_it2.pt")), str(b), quiet=True)
  fin = torch.load(b / "model_final.pt", weights_only=False)
  for k in ("actor_state_dict", "critic_state_dict"):
    for name, t in ck["alg"][k].items():
      assert torch.equal(t.cpu(), fin[k][name].cpu()), (k, name)
  assert fin["cc_state"]["r_bar"] == ck["alg"]["cc_state"]["r_bar"] != 0.0
  assert fin["cc_state"]["learning_rate"] == ck["alg"]["cc_state"]["learning_rate"]
  o1 = ck["alg"]["optimizer_state_dict"]["state"]
  o2 = fin["optimizer_state_dict"]["state"]
  assert o1.keys() == o2.keys() and len(o1) > 0
  for i in o1:
    assert torch.equal(o1[i]["exp_avg"].cpu(), o2[i]["exp_avg"].cpu())


def test_resumed_run_continues_the_iteration_counter(tmp_path):
  import json

  a = tmp_path / "a"
  run(_spec(iters=2, save_at=2), str(a), quiet=True)
  b = tmp_path / "b"
  run(_spec(iters=4, resume=str(a / "ckpt_it2.pt")), str(b), quiet=True)
  rows = [json.loads(l) for l in open(b / "iter.jsonl")]
  assert [r["iter"] for r in rows] == [2, 3]
