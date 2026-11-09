#!/bin/bash
# Continue the 1M-step pilot runs from their RLlib checkpoints (same hyperparameters, only a later stopping point).
# Rows are appended to the same results/rl_pilot_*.jsonl (a "restored" row marks the join).
# Usage: PAR=7 ./launch_extend.sh      (fwp -> 2M, ssp / ssp_aux -> 3M)
cd "$(dirname "$0")"
PAR=${PAR:-7}
OUT=../results
jobs=()
for s in 0 1 2; do jobs+=("fwp $s 2000000"); done
for s in 0 1 2; do for a in ssp ssp_aux; do jobs+=("$a $s 3000000"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P $PAR -L 1 bash -c '
  a=$0; s=$1; n=$2
  f='$OUT'/rl_pilot_RepeatPreviousEasy_${a}_s${s}.jsonl
  OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python run_pilot.py --arm $a --seed $s --steps $n --workers 0 --envs-per-worker 32 --out $f --restore /tmp/rl_ckpt/${a}_s${s} --ckpt /tmp/rl_ckpt2/${a}_s${s} > /tmp/rl_logs/ext_${a}_s${s}.log 2>&1'
