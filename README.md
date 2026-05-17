# SMC-Blind-Spot

Code and pre-computed results for:

> Ahn, J.; Hwang, T. G.; Jung, M.-R. **"The SMC Blind Spot: A Failure Mode Analysis of State-of-the-Art Beat Tracking."** Submitted to ISMIR 2026. [arXiv:2605.12287](https://arxiv.org/abs/2605.12287)

**Repository:** [github.com/zaiisao/SMC-Blind-Spot](https://github.com/zaiisao/SMC-Blind-Spot)

The paper asks why deep beat trackers that score >0.9 F-measure on mainstream datasets collapse on the [SMC dataset](https://zenodo.org/record/3553592), and traces the failures back to (i) activation-level breakdowns rooted in training-data distribution mismatch and (ii) a DBN tempo prior that is wrong for ~20% of SMC.

The companion artifact is a single Jupyter notebook, [SMC_Beat_Tracking_Analysis.ipynb](SMC_Beat_Tracking_Analysis.ipynb), that reproduces every number and figure in the paper.

## Repository contents

```
SMC-Blind-Spot/
├── SMC_Beat_Tracking_Analysis.ipynb  # paper-companion notebook (all results + figures)
├── data/                             # 12 CSVs the notebook reads / writes — see "Data files"
├── smc_metadata/                     # non-audio portion of the SMC dataset (tags + corrected
│                                     # beat annotations + provenance README), redistributed
│                                     # because the original metadata is hard to find
├── scripts/                          # producer scripts for the CSVs in data/
├── beat_this/                        # submodule: github.com/CPJKU/beat_this
├── beat_this_annotations/            # submodule: github.com/CPJKU/beat_this_annotations
└── Beat-Transformer/                 # submodule: github.com/zhaojw1998/Beat-Transformer
```

The SMC audio and pretrained checkpoints are **not** in this repo (size + license); the
notebook's setup cell handles downloading them on Colab, and instructions below cover local
setup. The SMC difficulty descriptors and ground-truth beat annotations *are* included in
[smc_metadata/](smc_metadata/) — see that folder's README for details and attribution.

## Reproducing the paper

### Quickest path: read the analysis without re-running inference

The committed CSVs already contain every per-track result. Open the notebook and run cells from Section 2 onward — they only read the CSVs and produce the paper's tables and figures.

```bash
git clone --recurse-submodules https://github.com/zaiisao/SMC-Blind-Spot.git
cd SMC-Blind-Spot
pip install pandas numpy scipy matplotlib mir_eval madmom jupyter
jupyter notebook SMC_Beat_Tracking_Analysis.ipynb
```

### Full re-run: regenerate the CSVs from scratch

This requires the SMC audio, the Beat Transformer 34 GB demixed-spectrogram archive, and a GPU.

```bash
# Environment
conda create -n analyze-smc python=3.10 -y
conda activate analyze-smc
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install madmom mir_eval librosa pandas scipy tqdm einops soxr rotary-embedding-torch mido
pip install -e beat_this

# Data
#   1. SMC_MIREX dataset:           https://zenodo.org/record/3553592  ->  ./SMC_MIREX/
#   2. Beat Transformer artifacts:  see Beat-Transformer/README.md     ->  ./Beat-Transformer/data/demix_spectrogram_data.npz
#   3. Pretrained checkpoints:      beat_this auto-downloads;  Beat-Transformer/checkpoint/ comes with the submodule

# Run inference
python scripts/run_folds.py                       # beat_this, no DBN
python scripts/run_folds_dbn_confidence.py        # beat_this + DBN + confidence
python scripts/run_beat_transformer.py            # Beat Transformer, 8-fold
python scripts/run_madmom_tcn_baseline.py         # madmom TCN baseline + tempo-constrained DBN

# Build the master per-track table
python scripts/build_results_csv.py               # writes data/results.csv

# Diagnostics & follow-up experiments
python scripts/activation_analysis.py             # writes data/activation_diagnostics.csv (+ activation_plots/)
python scripts/octave_correction.py               # writes data/octave_correction_results.csv
python scripts/ibi_cleanup.py                     # writes data/ibi_cleanup_results.csv
python scripts/threshold_sweep.py                 # writes data/threshold_sweep_results.csv
python scripts/gt_tempo_lambda_sweep.py           # writes data/gt_tempo_lambda_sweep_results.csv
python scripts/analyze_tags.py                    # prints SMC tag/confidence breakdown
```

## Data files

All CSVs live in [data/](data/) and are loaded via `DATA_DIR = ROOT / "data"` in both the notebook and the scripts.

| CSV | Rows × Cols | Source | Contents |
|---|---|---|---|
| `data/results.csv` | 217 × 64 | `build_results_csv.py` | Master table: per-track metadata, tag one-hots, mir_eval metrics under no-DBN and DBN, plus DBN log-prob and tempo stats. |
| `data/activation_diagnostics.csv` | 217 × 32 | `activation_analysis.py` | Activation-function statistics for beat_this and Beat Transformer (mean, max, peak prominence, periodicity, entropy, mean activation at GT-beat positions, failure-mode label per model). |
| `data/octave_correction_results.csv` | 217 × 15 | `octave_correction.py` | ACF-based octave error detection/correction. Orig F vs. corrected F (subsample / interpolate / oracle). |
| `data/ibi_cleanup_results.csv` | 217 × 13 | `ibi_cleanup.py` | Removal of beats whose inter-beat interval deviates from local median. Orig vs. clean F/CMLt/AMLt. |
| `data/threshold_sweep_results.csv` | 15 × 7 | `threshold_sweep.py` | Peak-pick threshold sweep, aggregate F/CMLt/AMLt per threshold. |
| `data/transition_lambda_sweep_results.csv` | 217 × 24 | *(producer lost — see note)* | Per-track DBN `transition_lambda` sweep across 13 values, optimal-λ and default-λ scores. |
| `data/gt_tempo_lambda_sweep_results.csv` | 217 × 10 | `gt_tempo_lambda_sweep.py` | GT-tempo constraint combined with per-track λ sweep; isolates whether the two levers compound. |
| `data/tcn_tempo_constrained_results.csv` | 217 × 20 | `run_madmom_tcn_baseline.py` | madmom TCN baseline, plus its tempo estimate used to constrain the DBN on beat_this activations. |
| `data/ibi_variability.csv` | 2304 × 6 | *(producer lost — see note)* | Per-track inter-beat-interval variability (mean, std, coefficient of variation) across SMC and several comparator datasets. |
| `data/training_data_bpm_summary.csv` | 6 × 10 | hand-curated | Tempo distribution summary for beat_this's 6 main training datasets (% below 55 BPM, etc.). |
| `data/gt_activation_dbn_per_track.csv` | 217 × 4 | written by the notebook | Upper-bound experiment: GT activations → DBN. |
| `data/bt_transformer_tempo_constrained_results.csv` | 217 × 44 | written by the notebook | Beat Transformer's full per-track table (paper-default DBN, wide DBN, top-K tempo, oracle, plus beat_this counterparts). |

**Note on lost producer scripts.** `transition_lambda_sweep_results.csv` and `ibi_variability.csv` were produced by scripts that are no longer in `scripts/`. The CSVs are committed as the canonical record of those experiments; reproducing them from scratch would require rewriting the producer scripts.

## Citing this work

If you build on this analysis, please cite the preprint:

```bibtex
@misc{ahn2026smcblindspot,
  title         = {The {SMC} Blind Spot: A Failure Mode Analysis of State-of-the-Art Beat Tracking},
  author        = {Ahn, Jaehoon and Hwang, Tae Gum and Jung, Moon-Ryul},
  year          = {2026},
  eprint        = {2605.12287},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2605.12287},
  note          = {Submitted to ISMIR 2026}
}
```

Plain text:
> Ahn, J.; Hwang, T. G.; Jung, M.-R. "The SMC Blind Spot: A Failure Mode Analysis of State-of-the-Art Beat Tracking." arXiv preprint arXiv:2605.12287, 2026.

## Citing related work

If you use the SMC dataset:
> Holzapfel, A.; Davies, M. E. P.; Zapata, J. R.; Oliveira, J. L.; Gouyon, F. "Selective Sampling for Beat Tracking Evaluation," *IEEE Trans. Audio, Speech, Lang. Proc.*, 20(9), 2012.

If you use beat_this:
> Foscarin, F.; Schlüter, J.; Widmer, G. "Beat This! Accurate Beat Tracking Without DBN Postprocessing," ISMIR 2024.

If you use Beat Transformer:
> Zhao, J.; Xia, G.; Wang, Y. "Beat Transformer: Demixed Beat and Downbeat Tracking with Dilated Self-Attention," ISMIR 2022.
