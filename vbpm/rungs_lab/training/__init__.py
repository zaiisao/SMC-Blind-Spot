"""Training harness for the rung ladder.

The reusable algorithms live here -- em (exact Baum-Welch), gradient (frozen-frontend CRF
backprop), e2e (joint frontend + rung) -- alongside the shared data/frontend/evaluation pieces.
Each rung declares its training contract (see rungs/base.py); train.py routes configs/train.yaml
to the right rung's fit().
"""
