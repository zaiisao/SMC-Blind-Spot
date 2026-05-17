# SMC dataset metadata

A redistribution of the **non-audio** portion of the SMC (Sound and Music Computing) MIREX dataset.

This folder exists because the difficulty descriptors and annotator metadata that make SMC distinctive are not packaged with most beat-tracking benchmarks, and the original Zenodo distribution has historically been intermittent. The folder is included in this repository so the paper's analysis is fully self-contained — no audio is bundled.

## Contents

| Path | Count | Description |
|---|---|---|
| `tags/SMC_XXX.tag` | 217 | Per-track free-text **difficulty descriptors** (e.g., `expressive timing`, `ternary meter`, `lack of transient sounds`), followed by a final line of the form `a1` / `m3` / `o2` indicating annotator letter and annotator confidence (1 = easy to annotate, 4 = very hard). |
| `annotations/SMC_XXX_*_*_*_*.txt` | 217 | Ground-truth **beat times** in seconds, one beat per line. These are the corrected annotations from `SMC_MIREX_Annotations_05_08_2014` (excerpts 056, 137, 153, 203, and 257 have updated final beats vs. the original release; see the parent `SMC_MIREX_Readme.txt` for details). Filename suffix encodes metrical interpretation and annotator letter — for beat tracking evaluation only the beat times matter. |
| `SMC_MIREX_Readme.txt` | 1 | The original README distributed with the dataset, preserved verbatim for provenance. |

The audio (217 mono `.wav` files at 44.1 kHz) is **not** included here. Obtain it from the [SMC Group's data page](http://smc.inescporto.pt/research/data-2/) at INESC TEC Porto, the dataset's canonical distribution site.

## Source and attribution

All files in this folder are redistributed from:

> Holzapfel, A.; Davies, M. E. P.; Zapata, J. R.; Oliveira, J. L.; Gouyon, F. "Selective Sampling for Beat Tracking Evaluation," *IEEE Transactions on Audio, Speech, and Language Processing*, 20(9), 2539–2548, Nov. 2012. doi: [10.1109/TASL.2012.2205244](https://doi.org/10.1109/TASL.2012.2205244)

Please cite the paper above if you use these annotations or tags in your work.

## Track index convention

Files are numbered `SMC_001` through `SMC_289`, with gaps — there are 217 tracks in total. Tracks `SMC_271`–`SMC_289` are designated by the original authors as "easy" reference excerpts; the remaining 198 are the "hard" set that SMC is benchmarked on.
