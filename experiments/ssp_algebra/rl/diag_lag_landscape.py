"""Retrieval landscape of the tied read at init: score of the wanted token (the card 4 steps back) as a function of the lag
the head asks for.  This is the quantity every learning signal about the lag must pass through (RUN_011's 'dead zone')."""
import json
import numpy as np
import torch
import test_ssp_tied as T
from test_ssp_model import make, toks

m = make("tied")
tk = toks(64, 51, 123)
k = 4
rows = []
for lag in np.arange(-2.0, 10.01, 0.25):
    T._set_lag(m, float(lag))
    s = T._scores(m, tk)
    tgt = tk[:, :-k]
    sc = s[:, k:]
    wanted = sc.gather(-1, tgt.unsqueeze(-1))[..., 0].mean().item()
    other = (sc.sum(-1) - sc.gather(-1, tgt.unsqueeze(-1))[..., 0]).mean().item() / 3
    ce = torch.nn.functional.cross_entropy((sc / 0.1).reshape(-1, 4), tgt.reshape(-1)).item()
    rows.append({"lag": float(lag), "wanted_score": wanted, "other_score": other, "ce_tau0.1": ce})
    print(f"lag {lag:+5.2f}  wanted {wanted:+.3f}  other {other:+.3f}  CE {ce:.3f}")
json.dump(rows, open("../results/rl_tied_lag_landscape.json", "w"), indent=1)
