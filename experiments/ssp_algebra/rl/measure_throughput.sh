#!/bin/bash
# Uncontended single-run throughput: one run at a time, 3 PPO iterations each (32 768 steps/iter), local sampler, 1 thread.
cd "$(dirname "$0")"
for a in gru fwp ssp ssp_aux; do
  OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python run_pilot.py --arm $a --seed 0 --steps 100000 --max-iters 3 --workers 0 \
     --envs-per-worker 32 --out ../results/rl_throughput_solo_$a.jsonl > /tmp/rl_logs/tp_$a.log 2>&1
done
