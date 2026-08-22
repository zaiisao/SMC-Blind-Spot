"""R2 (EM DBN) training driver -- replaces train.py, whose import of rungs.r4_neural_hmm
(never committed; lost in the v2 cleanup) breaks at module load. Same dispatch path."""
import sys, yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rungs.r2_em_dbn import R2GenerativeLambda

config = yaml.safe_load(open(sys.argv[1]))
R2GenerativeLambda.fit(config)
