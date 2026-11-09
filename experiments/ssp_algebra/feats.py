"""Fair, episode-length-independent time / absolute-index features (handoff 09 step B).

RUN_007 gave every arm a *different* view of the same quantity: the generic arms received the query as
``j / T`` and ``t / T``, while the SSP arm multiplied by ``T`` internally, so the arm that was scored on
absolute index recall was the only one handed the de-normalisation (RUN_007 section 8, bug 7.13/9).
A ``t / T`` clock is worse than unfair, it is *wrong at extrapolation*: at ``T = 64`` the value 0.25 means step 16,
where at ``T = 16`` it meant step 4, so an arm cannot reuse the clock it learned (RUN_007 section 5.4).

This module is the single source of those features.  Every constant is fixed, so the meaning of a feature does not
change when the episode length does:

    * ``t / 32``      -- a monotone absolute-time ramp at a fixed scale
    * ``sin/cos(2 pi t / p)`` for ``p`` in ``(2, 4, 8, 16, 32, 64)`` -- so parity (``p = 2``) and coarse magnitude
      are both discoverable, and a windowed / relative-position arm can match a query index to the step it means
    * ``NORM = 16``  -- the fixed de-normalisation for the query (absolute index or lag)

``TIME_DIM`` (= 13) is appended to every task's ``features`` and to ``task.in_dim``, so all arms see them.
"""

import math

import torch

NORM = 16.0                                  # fixed de-normalisation for absolute index / lag: never tied to T
PERIODS = (2, 4, 8, 16, 32, 64)
RAMP_SCALE = 32.0
TIME_DIM = 1 + 2 * len(PERIODS)              # 13


def abs_time_feats(t):
    """``t`` (...,) or (B, T) long/float -> (..., TIME_DIM).

    Includes period 2, so ``t % 2`` (the key/value parity of variant B's stream) is representable as a function of
    position alone, and the whole vector is invariant to the episode length.
    """
    tf = t.float()
    parts = [tf / RAMP_SCALE]
    for p in PERIODS:
        ang = 2.0 * math.pi * tf / float(p)
        parts.append(torch.sin(ang))
        parts.append(torch.cos(ang))
    return torch.stack(parts, dim=-1)
