# Transcribed console output (no result JSON exists for these two runs)

Both were launched with `--workers 4 --envs-per-worker 8` (RepeatPreviousEasy, 32 envs, ~32.8k steps/iteration) and
`--workers 2 --envs-per-worker 8` (CountRecallEasy, 16 envs, ~16.4k steps/iteration), POPGym's GRU, its own config,
`Antialias(PreviousAction)` wrapper, one seed each. Killed by `timeout` (SIGTERM), not by a crash.

## RepeatPreviousEasy, GRU: killed at 296k of a 400k request

| iter | steps | episode_return_mean |
|---|---|---|
| 1 | 32,844 | -0.502 |
| 2 | 65,688 | -0.454 |
| 3 | 98,532 | -0.314 |
| 4 | 131,376 | -0.107 |
| 5 | 164,220 | +0.065 |
| 6 | 197,064 | +0.191 |
| 7 | 229,908 | +0.290 |
| 8 | 262,752 | +0.359 |
| 9 | 295,596 | +0.418 |

## CountRecallEasy, GRU: killed at 98.5k of a 500k request

| iter | steps | episode_return_mean |
|---|---|---|
| 1 | 16,422 | -0.929 |
| 2 | 32,844 | -0.919 |
| 3 | 49,266 | -0.918 |
| 4 | 65,688 | -0.914 |
| 5 | 82,110 | -0.910 |
| 6 | 98,532 | -0.906 |

Limits: one seed, partial budgets, different machine/loop from POPGym's published runs (up to 15e6 steps).
