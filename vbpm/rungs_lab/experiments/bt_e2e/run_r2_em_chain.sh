#!/bin/sh
# Self-sufficient chain: wait for the lambda-only run, then launch --learn-obs.
# nohup'd so it survives VSCode/session disconnects.
cd /home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab/experiments/bt_e2e || exit 1
PY=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
export PYTHONPATH=/home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab:/home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab/external/beat_transformer/code
while pgrep -f "train_r2_em.py$" > /dev/null; do sleep 20; done
$PY train_r2_em.py --learn-obs > r2_em_learn_obs.log 2>&1
echo "learn-obs exit: $?" >> r2_em_learn_obs.log
