#!/bin/bash
# Lag-bank round (handoffs/11 step 4): 2 arms x 3 seeds, 1M steps, same protocol/PPO settings as launch_tied.sh
# (so the rows are comparable with rl_tied_* / rl_pilot_* on the same machine).
#   STEPS=1000000 PAR=4 ./launch_bank.sh
# results/rl_bank_RepeatPreviousEasy_<arm>_s<seed>.jsonl ; checkpoints in /tmp/rl_ckpt_bank
cd "$(dirname "$0")"
PAR=${PAR:-4}
STEPS=${STEPS:-1000000}
ENV=${ENV:-RepeatPreviousEasy}
export ENV
OUT=../results
mkdir -p $OUT /tmp/rl_logs
jobs=()
for s in 0 1 2; do for a in tied_bank tied_bank_aux; do jobs+=("$a $s $STEPS"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P $PAR -L 1 bash -c '
  a=$0; s=$1; n=$2
  f='$OUT'/rl_bank_${ENV}_${a}_s${s}.jsonl
  OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python run_pilot.py --env '$ENV' --arm $a --seed $s --steps $n \
    --workers 0 --envs-per-worker 32 --out $f --ckpt /tmp/rl_ckpt_bank/${a}_s${s} > /tmp/rl_logs/bank_${a}_s${s}.log 2>&1'
