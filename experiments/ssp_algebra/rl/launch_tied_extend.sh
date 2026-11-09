#!/bin/bash
# Resume the tied-round 1M runs from their checkpoints: tied -> 3M; tied_aux and untied_scaled -> 2M.
cd "$(dirname "$0")"
PAR=${PAR:-7}
OUT=../results
jobs=()
for s in 0 1 2; do jobs+=("tied $s 3000000"); done
for s in 0 1 2; do jobs+=("tied_aux $s 2000000"); jobs+=("untied_scaled $s 2000000"); done
printf '%s\n' "${jobs[@]}" | xargs -P $PAR -L 1 bash -c '
  a=$0; s=$1; n=$2
  f='$OUT'/rl_tied_RepeatPreviousEasy_${a}_s${s}.jsonl
  OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python run_pilot.py --arm $a --seed $s --steps $n --workers 0 --envs-per-worker 32 --out $f --restore /tmp/rl_ckpt_tied/${a}_s${s} --ckpt /tmp/rl_ckpt_tied2/${a}_s${s} > /tmp/rl_logs/tiedext_${a}_s${s}.log 2>&1'
