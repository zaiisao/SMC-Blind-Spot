#!/bin/sh
# Wait for all demix shards to finish, then launch both training arms.
cd /home/sogang/jaehoon/VBPM
while pgrep -f "beat_transformer_demix.py.*--batch" > /dev/null; do sleep 30; done
echo "demix done: $(ls cache/bt_demix/*/*.npz | wc -l) songs cached" 
P=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
PYTHONPATH=. nohup $P experiments/bt_e2e/train_bt.py --arm vanilla --device cuda:0 --epochs 30 \
    > experiments/bt_e2e/vanilla.log 2>&1 &
PYTHONPATH=. nohup $P experiments/bt_e2e/train_bt.py --arm r2 --device cuda:1 --epochs 30 \
    > experiments/bt_e2e/r2.log 2>&1 &
wait
