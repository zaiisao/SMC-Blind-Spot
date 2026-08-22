#!/bin/bash
# E1 sweep: one process per 1/T value, CPU only, 2 threads each.
cd /home/sogang/jaehoon/Analyze-SMC
PY=/home/sogang/mnt/db_2/anaconda3/envs/analyze-smc/bin/python
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
MODE=$1; shift
for T in "$@"; do
  tag=${MODE}_$(echo "$T" | tr '.' 'p')
  $PY scripts/e1_soft_tempo_prior.py --invT "$T" --mode $MODE --out /tmp/e1_res_${tag}.json \
      > /tmp/e1_log_${tag}.txt 2>&1 &
done
wait
echo ALLDONE
