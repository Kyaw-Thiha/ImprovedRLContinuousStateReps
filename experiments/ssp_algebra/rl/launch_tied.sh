#!/bin/bash
# Tied-relative-addressing round: 4 arms x 3 seeds, 1M steps (same protocol/PPO settings as launch_pilot.sh).
# Usage: STEPS=1000000 PAR=6 ./launch_tied.sh ; results/rl_tied_RepeatPreviousEasy_<arm>_s<seed>.jsonl ; checkpoints in /tmp/rl_ckpt_tied
cd "$(dirname "$0")"
PAR=${PAR:-6}
STEPS=${STEPS:-1000000}
OUT=../results
mkdir -p $OUT /tmp/rl_logs
jobs=()
for s in 0 1 2; do for a in tied tied_aux tied_lag4 untied_scaled; do jobs+=("$a $s $STEPS"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P $PAR -L 1 bash -c '
  a=$0; s=$1; n=$2
  f='$OUT'/rl_tied_RepeatPreviousEasy_${a}_s${s}.jsonl
  OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python run_pilot.py --arm $a --seed $s --steps $n --workers 0 --envs-per-worker 32 --out $f --ckpt /tmp/rl_ckpt_tied/${a}_s${s} > /tmp/rl_logs/tied_${a}_s${s}.log 2>&1'
