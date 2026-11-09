#!/bin/bash
# Pilot launcher: 4 arms x 3 seeds on RepeatPreviousEasy, 1M steps each, POPGym PPO config, local sampler: 0 remote workers x 32 envs
# (train_batch_size = 1024*1*32 = 32768 steps/iteration), at most $PAR runs at once, 1 thread each.
# Usage: PAR=6 ./launch_pilot.sh   (results/rl_pilot_*.jsonl are written incrementally)
cd "$(dirname "$0")"
PAR=${PAR:-6}
STEPS=${STEPS:-1000000}
OUT=../results
mkdir -p $OUT /tmp/rl_logs
jobs=()
jobs+=("ssp_oracle4 0 300000")
for s in 0 1 2; do for a in gru ssp ssp_aux fwp; do jobs+=("$a $s $STEPS"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P $PAR -L 1 bash -c '
  a=$0; s=$1; n=$2
  f='$OUT'/rl_pilot_RepeatPreviousEasy_${a}_s${s}.jsonl
  OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python run_pilot.py --arm $a --seed $s --steps $n --workers 0 --envs-per-worker 32 --out $f --ckpt /tmp/rl_ckpt/${a}_s${s} > /tmp/rl_logs/${a}_s${s}.log 2>&1'
