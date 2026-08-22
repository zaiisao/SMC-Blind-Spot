# Prior-art check: "differentiable madmom DBN, trained end-to-end / by EM"

Target claim under audit (ICASSP 2027):

> (D) The field abandoned the madmom DBN because end-to-end models win on F-measure;
> (E) that is a mistake, because those models lose on CMLt/AMLt;
> (M) the mechanism is that frame-wise BCE factorizes over time and cannot express a preference over global beat trajectories;
> (R) the repair is a differentiable DBN in PyTorch, trained end-to-end through the forward algorithm / by EM with partially observed tempo states.

Verdict categories: **PRE-EMPTS** (this claim is already published, do not claim it) /
**OVERLAPS-MUST-CITE** / **ADJACENT** / **NOT-RELEVANT**.

Compiled 2026-08-20. `repositorio-aberto.up.pt` unreachable, so the Porto (Davies/Oliveira)
thesis lineage is unchecked — a negative there is unconfirmed. CNKI / non-English venues not searched.

---

## 0. THE HEADLINE: the paper we are attacking already published our diagnosis

**Foscarin, Schlüter, Widmer. "Beat this! Accurate beat tracking without DBN postprocessing." ISMIR 2024. arXiv:2407.21658.**

What it does: ~20M-param spectrogram transformer (partial frontend transformers + main transformer + Sum Head),
trained with a **weighted shift-tolerant BCE** (max-pool predictions over 7 frames before comparison; negatives
ignored ±6 frames of a label), plus masking / pitch / tempo augmentation. Decoding is minimal peak picking, no DBN.
Evaluated with **F1, CMLt, AMLt** via `mir_eval`, 8-fold CV over 18 datasets, plus GTZAN test-only, plus ablations
including a **"– with DBN"** row.

They report (Table 2, means over 3 seeds; beat | downbeat):

| system | beat F1 | beat CMLt | beat AMLt | db F1 | db CMLt |
|---|---|---|---|---|---|
| Hung et al. (with DBN) | 88.7 | **81.2** | **92.0** | 75.6 | **71.5** |
| Beat This! (no DBN) | **89.1** | 79.8 | 89.8 | **78.3** | 67.3 |
| Beat This! + DBN | 88.1 | 80.5 | 91.1 | 77.4 | — |

Verbatim, from their own text:

- Abstract: *"…surpasses the current state of the art in F1 score despite using no DBN. However, it can still fail, especially for difficult and underrepresented genres, and **performs worse on continuity metrics**…"*
- §4.3: *"**Disappointingly**, we notice that the continuity metrics (CMLt and AMLt) are lower than those of Hung et al. … our network introduces non-periodic beats, which drastically lower the continuity metrics. We are then brought to wonder why our network cannot learn a supposedly obvious behaviour, such as only producing periodic-like output, and we can propose two explanations. **Firstly, our loss does not specifically penalise non-periodic predictions, but treats each beat individually. This results in a discrepancy between what is preferred by continuity metrics and what the network learns to predict in difficult parts to minimise the loss.**"*
- §4.3: *"Using a DBN increases our CMLt downbeat performance by correcting some of the (wrongly) non-periodic outputs, but it reduces our F1 performance… The AMLt score does not increase since our network is overconfident in its predictions, and the DBN cannot easily switch to another metrical level."*
- §4.2: *"…even with our modifications, the BCE loss is not a good indicator of the F1 score, and **further research into alternative losses may be valuable**."*
- Conclusion: *"Remarkably, we do not use the DBN postprocessing… **However, removing the DBN hurts the CMLt and AMLt metrics.** A study on how this trade-off affects human perception, alternative metrics, and a direct comparison with DBN-based models on complex pieces is left for future work."* … *"Potential directions are: … **developing new losses that enforce periodicity during training**…"*

**VERDICT: PRE-EMPTS (D), (E) and (M) essentially in full.**
- (D) they name the abandonment; (E) they measure the CMLt/AMLt loss themselves and run the +DBN ablation showing the DBN buys continuity at F1 cost; (M) *"our loss … treats each beat individually … discrepancy between what is preferred by continuity metrics and what the network learns"* **is our factorized-loss argument, stated by them, in the paper we intended to critique.**
- They also pre-register the fix direction ("new losses that enforce periodicity during training") as explicit future work.
- Residual room: they attribute the deficit partly to **annotation non-periodicity** and explicitly *question the generality of AMLt*. They do not formalize the factorization argument, do not connect it to structured/sequence-level losses or to the ASR literature, and do not implement any fix. That formalization + implementation is the only surviving slice.

**Consequence: the framing "the field has not noticed the continuity deficit" must be abandoned outright.** It is in the abstract and conclusion of the very paper. Any submission implying otherwise will be desk-rejected by a reviewer who has read the abstract.

---

## 1. Training a beat/downbeat network end-to-end through a structured decoder

### 1.1 Böck, Krebs, Widmer. "Joint Beat and Downbeat Tracking with Recurrent Neural Networks." ISMIR 2016.
BLSTM frame activations + madmom bar-pointer DBN at inference. BCE training; DBN parameters hand-set.
Related: Krebs, Böck, Widmer, "An efficient state-space model for joint tempo and meter tracking," ISMIR 2015;
Krebs et al., "Rhythmic pattern modeling…," ISMIR 2013; Whiteley, Cemgil, Godsill, "Bayesian modelling of temporal structure in musical audio," ISMIR 2006 (the bar-pointer origin).
**VERDICT: OVERLAPS-MUST-CITE** — this is the baseline we differentiate, not a pre-emption. No gradient crosses the decoder.

### 1.2 Durand & Essid. "Downbeat Detection with Conditional Random Fields and Deep Learned Features." ISMIR 2016.
Four CNNs (harmony/rhythm/melody/bass) trained separately; their penultimate-layer features feed a CRF fitted with
**Pycrfsuite + L-BFGS** on precomputed features — no gradient into the CNNs. Their own closing sentence:
*"Future work will focus on **learning the deep networks and the conditional random field models jointly** and on refining the initial temporal segmentation."*
**VERDICT: OVERLAPS-MUST-CITE — and this is the single most dangerous citation for us.** A 2016 ISMIR paper on this exact task
names joint NN+CRF training as future work. Our contribution can be positioned as "the joint training Durand & Essid
called for, ten years later, on the bar-pointer state space" — but we cannot present the *idea* as new.

### 1.3 Fuentes, McFee, Crayencour, Essid, Bello. "A Music Structure Informed Downbeat Tracking System Using Skip-Chain Conditional Random Fields and Deep Learning." ICASSP 2019.
CRNN trained in Keras with Adam on beat/downbeat likelihood; skip-chain CRF built with `factorgraph`, coupling
constant α hand-chosen, run as inference on frozen outputs. Two stages, no joint objective.
**VERDICT: OVERLAPS-MUST-CITE.** Strongest published "structured decoder for downbeats", still does not close the loop.

### 1.4 Heydari, Cwitkowitz, Duan. "BeatNet: CRNN and Particle Filtering for Online Joint Beat, Downbeat and Meter Tracking." ISMIR 2021. arXiv:2108.03576.
CRNN trained frame-wise; two sequential Monte Carlo particle filters **at inference only**. Sampling-based, non-differentiable.
**VERDICT: ADJACENT.** Cite as an inference-side alternative to the DBN.

### 1.5 Zhao, Xia, Wang. "Beat Transformer: Demixed Beat and Downbeat Tracking with Dilated Self-Attention." ISMIR 2022. arXiv:2209.07140. And Hung et al. (ISMIR 2022 / SpecTNT-line), BEAST (arXiv:2312.17156), Chiu et al. (TASLP 2023).
All: frame-wise loss + madmom DBN stacked at inference.
**VERDICT: OVERLAPS-MUST-CITE** (as the "DBN-still-used" side of the field, and Hung et al. is Beat This!'s continuity-metric superior).

### 1.6 Chen & Su. "Toward Postprocessing-Free Neural Networks for Joint Beat and Downbeat Estimation." ISMIR 2022.
Explicitly removes the DBN by architectural means; reports **F1 only** — no CMLt/AMLt.
**VERDICT: OVERLAPS-MUST-CITE.** Supports (D) *and* is an example of the F-measure-only reporting our (E) criticises. Do not overclaim: their omission of continuity metrics is a reporting gap, not a denial.

### 1.7 Steinmetz & Reiss. "WaveBeat: End-to-end beat and downbeat tracking in the time domain." AES 151, 2021. arXiv:2110.01436.
"End-to-end" = waveform-in, TCN, no spectrogram. Not end-to-end through a decoder.
**VERDICT: NOT-RELEVANT** — but a citation trap; do not let a reviewer think it pre-empts us, and do not miscite it ourselves.

### 1.8 Cheng & Goto. "Transformer-Based Beat Tracking with Low-Resolution Encoder and High-Resolution Decoder." ISMIR 2023.
"Decoder" = neural temporal-upsampling decoder, not a probabilistic decoder.
**VERDICT: NOT-RELEVANT** (citation trap).

### 1.9 "Beat Tracking as Object Detection." arXiv:2510.14391 (2025).
Replaces the DBN with non-maximum suppression. Heuristic, not a structured decoder.
**VERDICT: ADJACENT.**

### 1.10 "An Efficient Hidden Markov Model with Periodic Recurrent Neural Network Observer for Music Beat Tracking." Electronics 11(24):4186, 2022.
Full text not retrievable (MDPI 403). Framing ("RNN observer") indicates the standard observation-model/decoder split.
**VERDICT: ADJACENT, UNVERIFIED.** Verify before citing or before asserting novelty in print.

### 1.11 Adjacent-MIR counterexample we must pre-empt
Segmental/semi-CRF chord recognition (e.g. arXiv:1810.10002; the CNN+CRF chord line) reports that although
*"combining neural networks with CRFs gives a fully differentiable model that can be learned jointly,"* they found
*"separate training was found to be advantageous both in terms of convergence time and performance."*
**VERDICT: ADJACENT — but it is the reviewer's ready-made objection.** Someone in MIR tried joint training on a
neighbouring task and reported it did not help.

### 1.12 Overall search result for item 1
**No published work trains a beat/downbeat network end-to-end through an HMM/DBN/CRF/differentiable-DP decoder.**
The gap is real as of Aug 2026. But see §4 — it is not novel *in-house*, and see §3 — it is not novel *as a method*.

---

## 2. Davies & Böck, and the metric literature

### 2.1 Davies, Degara, Plumbley. "Evaluation methods for musical audio beat tracking algorithms." Tech. Rep. C4DM-TR-09-06, QMUL, 2009.
The definitional source for CMLc/CMLt/AMLc/AMLt: continuity-based measures require a beat *and its predecessor* to be
correct, and AML additionally admits half/double/offbeat metrical levels. The sequence-level character of the
continuity family is definitional here, not a novel observation of ours.
**VERDICT: OVERLAPS-MUST-CITE** (definitional).

### 2.2 Davies & Böck. "Evaluating the Evaluation Measures for Beat Tracking." ISMIR 2014, pp. 637–642 (archives.ismir.net/ismir2014/paper/000238.pdf).
Correlates objective beat-tracking scores against **subjective human ratings** on 48 15-s excerpts. Findings, verbatim:
- *"The highest overall correlation between objective accuracy scores and subjective ratings (0.89) occurs for **Continuity-T** for a tolerance window of ±9% of the IAI rather than the default value of ±17.5%."*
- *"If we are to extrapolate these results to all existing work in the beat tracking literature this would imply that **any papers reporting only performance for the Annotated condition using F-measure and PScore may not be as representative of subjective ratings (and hence true performance) as they could be** by incorporating additional evaluation conditions."*
- Recommends *"either the use of Continuity-T for the Annotated+Offbeat+D/H condition with a smaller tolerance window, or Information Gain."*
- Caveats its own scope: 15-s excerpts only, *"we cannot yet claim that our subjective vs. objective correlations will hold for evaluating longer excerpts."*

**VERDICT: OVERLAPS-MUST-CITE — and this is our best *supporting* citation, not a threat.** It supplies the missing
normative premise for (E): continuity-T is the measure that best tracks perceived quality, and F-measure-only
reporting understates true performance. Use it to argue *why* the CMLt/AMLt loss matters. It does **not** attribute
the dissociation to a factorized training loss — that link is (M), and (M) is pre-empted by Beat This! §4.3, not by
this paper. Note also that it recommends **Information Gain**, which we should therefore report too or explain away.

### 2.3 Anyone else already making the factorized-loss argument?
Beyond Beat This! §4.3 (§0 above), the closest is the Carnatic meter-tracking work
(*"Revisiting Meter Tracking in Carnatic Music using Deep Learning Approaches"*, arXiv:2509.11241, 2025), which reports
the same signature — many beats correct but low CMLt, high AMLt from the post-processing stage predicting
rhythmically related tempi — as an empirical observation, without the loss-factorization explanation.
The knowledge-driven multiple-hypothesis contrastive SSL beat work (arXiv:2510.25560, 2025) is also in this space.
**VERDICT: ADJACENT** (both), **PRE-EMPTS** via Beat This! §4.3 for the argument itself.

---

## 3. General ML prior art: differentiable DP, neural HMMs, and the ASR analogue

This is where "our method" turns out to be a fifty-year-old standard technique. **All of these must be cited honestly.**

### 3.1 The generic mechanism
- **Mensch & Blondel. "Differentiable Dynamic Programming for Structured Prediction and Attention." ICML 2018, PMLR 80:3462–3471. arXiv:1802.03676.** Smooth the `max` in a DP recursion with a strongly convex regularizer (negentropy → softmax/marginals; ℓ2 → sparsemax), making a broad class of DPs differentiable; instantiated as smoothed Viterbi and smoothed DTW. **Negentropy-smoothed Viterbi *is* the forward algorithm and its gradient *is* the backward pass.** **VERDICT: PRE-EMPTS the mechanism.** Our "differentiable DBN" is a named special case.
- **Kim, Denton, Hoang, Rush. "Structured Attention Networks." ICLR 2017. arXiv:1702.00887.** Linear-chain CRF forward-backward (and inside-outside) as an internal differentiable layer, trained end-to-end by backprop through inference, latent states unobserved. **VERDICT: PRE-EMPTS the mechanism.**
- **Cuturi & Blondel. "Soft-DTW: a Differentiable Loss Function for Time-Series." ICML 2017. arXiv:1703.01541.** **OVERLAPS-MUST-CITE.**
- **Rush. "Torch-Struct: Deep Structured Prediction Library." ACL 2020 demos. arXiv:2002.00876.** Batched GPU differentiable HMM/CRF/CKY. **VERDICT: OVERLAPS-MUST-CITE — and evidence that this is commodity infrastructure, not a contribution, since 2020.** (Our in-house `structured_dp.py` is itself certified against torch-struct.)
- **Hannun, Pratap, Kahn, Hsu. "Differentiable Weighted Finite-State Transducers." arXiv:2010.01003 (GTN).** **OVERLAPS-MUST-CITE.**
- Belanger & McCallum, SPENs, ICML 2016 (arXiv:1511.06350) + Belanger, Yang, McCallum, ICML 2017 (arXiv:1703.05667): **ADJACENT** (relaxes inference rather than differentiating exact DP).

### 3.2 Neural HMMs
- **Tran, Bisk, Vaswani, Marcu, Knight. "Unsupervised Neural Hidden Markov Models." EMNLP 2016 Workshop on Structured Prediction for NLP, 63–71 (W16-5907). arXiv:1609.09007.** Neural emissions/transitions, **unsupervised** marginal-likelihood training via the forward algorithm. Verbally the closest match to our sentence. **PRE-EMPTS the mechanism.**
- **Wang, Zhu, Alkhouli, Gan, Ney. "Neural Hidden Markov Model for Machine Translation." ACL 2018 short, 377–382 (P18-2060).** **OVERLAPS-MUST-CITE.**
- **Mehta, Székely, Beskow, Henter. "Neural HMMs are all you need (for high-quality attention-free TTS)." ICASSP 2022, 7457–7461. arXiv:2108.13320.** Left-to-right neural HMM trained on the **exact** sequence likelihood via the forward algorithm, replacing attention. Extension: OverFlow, Interspeech 2023 (arXiv:2211.06892). **PRE-EMPTS the mechanism in an audio domain.**
- **Raissi et al. "End-to-End Training of a Neural HMM with Label and Transition Probabilities." ASRU 2023. arXiv:2310.02724.** GPU forward-backward, joint emission+transition learning. **PRE-EMPTS.**
- **Krishnan, Shalit, Sontag. "Structured Inference Networks for Nonlinear State Space Models." AAAI 2017.** **OVERLAPS-MUST-CITE.**
- **Ebbers et al. "Hidden Markov Model Variational Autoencoder for Acoustic Unit Discovery." Interspeech 2017** (+ Glarner et al., Interspeech 2018). **OVERLAPS-MUST-CITE** — this is the "VAE with an HMM prior" slot our in-house VBPM work occupies.

### 3.3 The ASR analogue (cite densely and honestly)
- **Bourlard & Wellekens. "Links Between Markov Models and Multilayer Perceptrons." IEEE TPAMI 12(12):1167–1178, 1990.** / **Bourlard & Morgan, *Connectionist Speech Recognition: A Hybrid Approach*, Kluwer 1994.** The hybrid NN-emission/HMM-decoder template — architecturally identical to activation-net + madmom DBN.
- **LeCun, Bottou, Bengio, Haffner. "Gradient-Based Learning Applied to Document Recognition." Proc. IEEE 86(11):2278–2324, 1998.** **Graph Transformer Networks: globally train a ConvNet front-end jointly with a Viterbi/forward-scoring graph module by backprop through the alignment graph. Arguably the earliest full statement of our idea (1998).**
- **Bahl, Brown, de Souza, Mercer. "Maximum Mutual Information Estimation of HMM Parameters for Speech Recognition." ICASSP 1986, 49–52.** Sequence-level objective requiring numerator + denominator forward-backward: gradient of a marginalized-HMM objective, **1986**.
- **Veselý, Ghoshal, Burget, Povey. "Sequence-discriminative Training of Deep Neural Networks." Interspeech 2013, 2345–2349.** MMI/bMMI/MPE/sMBR on DNN acoustic models; 8–9% relative WER gain over frame cross-entropy. **The canonical "frame-factorized loss is the wrong loss; train at sequence level" result.**
- **Povey et al. "Purely Sequence-Trained Neural Networks for ASR Based on Lattice-Free MMI." Interspeech 2016, 2751–2755.** Exact GPU forward-backward over an HMM, no lattices, no CE pre-training. **The literal instantiation of "train the network end-to-end through a differentiable forward algorithm of an HMM."**
- **Hadian, Sameti, Povey, Khudanpur. "End-to-end Speech Recognition Using Lattice-free MMI." Interspeech 2018, 12–16.** Flat-start, no forced alignments, **HMM states never observed — only the label sequence**. Tightest match to our "partially observed states."
- **Graves, Fernández, Gomez, Schmidhuber. "Connectionist Temporal Classification." ICML 2006, 369–376.** / **Graves. "Sequence Transduction with Recurrent Neural Networks." arXiv:1211.3711 (RNN-T).** The two most-deployed instances of a differentiable forward-backward with partially observed states.
- **Hinton et al. "Deep Neural Networks for Acoustic Modeling in Speech Recognition." IEEE SPM 29(6):82–97, 2012.**
- Supporting: Collobert, Puhrsch, Synnaeve, "Wav2Letter" (ASG), arXiv:1609.03193; Rastogi, Cotterell, Eisner, NAACL-HLT 2016.

**VERDICT for §3.3 as a block: PRE-EMPTS the method entirely.** "Neural front-end trained through a differentiable
HMM forward algorithm with partially observed states" is standard ASR practice since 1986 (MMI) / 2016 (LF-MMI),
and is CTC's core mechanism. **We must not present the algorithm as a contribution.**
Note the in-house evidence that the ASR *pathology* transfers too: our R4.5 experiments reproduced
**emission domination / the acoustic-scaling disease** verbatim (per-frame emission gaps ~22,600 nats vs transition
scores 0.2–272; the transition became inert).

---

## 4. IN-HOUSE PRIOR WORK — `/home/sogang/jaehoon/Analyze-SMC/vbpm/` (read-only inspection)

**This is the most consequential section. Much of the proposed work is already built, run, and negative.**

### 4.1 The differentiable madmom DBN already exists
`rungs_lab/rungs/bar_pointer/structured_dp.py` — an exact structured DP for the Krebs bar-pointer state space in
PyTorch, exploiting the topology (deterministic +1 advance inside a tempo block = a gather; tempo mixing only at
beat boundaries = one `[num_tempi, num_tempi]` logsumexp), so `[num_states, num_states]` is never materialized.
7,605× fewer ops/frame than dense at num_states=16,756. Its own docstring: *"Speed is NOT why this file exists…
**GRADIENTS are why.** … madmom's Cython HMM runs this topology fast but gives no gradient, which R2–R4 need."*
Certified against a dense DP, which is itself certified against hmmlearn and torch-struct.
Supporting rungs: `r0_madmom_dbn.py`, `r1_2016_dbn.py`, `r2_em_dbn.py`, `r3_conditioned_dbn.py`, `r4_5_rich_emission.py`.

### 4.2 End-to-end training through it — ALREADY RUN, AND IT LOST
`rungs_lab/experiments/bt_e2e/` + `training/e2e.py` + `experiments/bt_e2e/crf_baseline.py`.
The CRF objective is exactly the proposed one: `nll = logZ − score(annotated path)`, logZ by exact forward through
the structured DP, differentiable w.r.t. both `transition_lambda` and the activations; the annotated bar-pointer
path is constructed exactly in the Krebs state space. Trained jointly with a **Beat Transformer** frontend from
scratch, fold-0, 1,020 train / 146 val songs (`RESULTS.md`, 2026-07-18):

| activations | decode λ | beat F | downbeat F |
|---|---|---|---|
| vanilla (BCE + fixed madmom DBN) | 100 | **0.9506** | **0.9053** |
| r2 e2e (BCE + CRF NLL through exact forward) | 100 | 0.9446 | 0.8899 |
| r2 e2e | 14.2 (learned) | 0.9374 | 0.8637 |

*"VANILLA WINS. The e2e CRF arm does not beat BCE + fixed madmom DBN."* Known failure mode recorded:
**pure CRF saturates** (no calibration pressure; logits hit |55|, 100% of frames sigmoid-saturated within one epoch,
CRF gradient dies) — fixed only by a hybrid CRF+BCE loss. Three harness asymmetries were later found and fixed
(LR annealing keyed to the wrong loss; observation-λ train/decode mismatch; epoch/data edge) and a rematch was
launched — **the rematch result is not recorded in `RESULTS.md`. SUPERSEDED: see the ADDENDUM at the end of this
file. The rematch was run, the history files survive, and the CRF arm lost harder (0.9658 vs 0.9115). But see
`docs/vbpm/VBPM_ASSETS_FOR_ARGUMENT.md` — that negative is off-target: it tests CRF-NLL as a discriminative frontend
objective (which the record itself calls "NOT a rung... off-program"), on a fold that is 69% ballroom + hjdb,
where a constant-period grid already ceilings at 0.918-0.999. The *generative* rung on the same chassis
(R2 exact EM) WINS on that same fold, 0.9132 vs 0.9033.**

> ⚠️ **RETRACTED 2026-08-20 by forensic audit — do NOT put "the generative rung wins" in the
> paper.** `results_r2_val.json` loads a DIFFERENT frontend (`checkpoints/bt_fold0_repacked.pt`,
> the pretrained BT, not the from-scratch e2e model) and forces `correct=False` — **peak-snap
> OFF** — in every row, so it is not comparable to the bt_e2e experiment at all. At the full
> deployment config the record REVERSES it: RESULTS.md:110 gives R0 madmom 0.9506 = R1 0.9506 =
> CRF(98.6) 0.9506 > R2-generative(λ=40.3) 0.9488, and on continuity RESULTS.md:212 has
> `em-single λ=40.3` at 0.9494 / 0.9144 / 0.9340 | 0.8924 / 0.8671 / 0.9154, losing to hand-set
> `λ=100` at 0.9504 / 0.9191 / 0.9410 | 0.9057 / 0.8842 / 0.9366 — **on all six metrics.**
> The 0.9132-vs-0.9033 numbers are real but measure a different thing; they are not a
> head-to-head and not a corroboration of the CRF experiment.

### 4.3 EM / Baum-Welch training of the DBN — ALSO ALREADY RUN
`training/em.py` + `rungs/r2_em_dbn.py`: exact Baum-Welch, whole-dataset E-step, closed-form M-step, verified
against brute-force path enumeration (4.8e-7) and Fisher's identity (EM λ=40.27 == 250 Adam steps λ=39.05).
Findings already in hand:
- **Integer-interval dithering corrupts the generative MLE of `transition_lambda`** (a metronomically perfect
  synthetic signal reproduces the real-corpus λ≈40 to 1.5%; a 2× fps rescaling doubles λ as predicted). Repaired by
  a two-component dither+exponential mixture kernel learned by the same EM: w=0.370, λ=93.1 — which **agrees with
  hand-set 100 and with the discriminatively-learned CRF λ=98.6**. Three estimators, one answer.
- In-house prior-art check (2026-07-20) judged the dithering→corrupted-MLE→likelihood/task-divergence chain
  **unpublished**, with Krebs's thesis §4.3.2 (Viterbi training chosen over Baum-Welch) as the nearest anticipation.
- Deployment-level outcome: hand-set λ=100, CRF λ=98.6 and mixture (w=.37, λ=93.1) are a **dead tie within ±0.0004
  on all six metrics** (beat F/CMLt/AMLt and downbeat F/CMLt/AMLt). *"the transition axis is now closed at parity."*
- The continuity tiebreaker table (`tiebreak_continuity.py`) does contain a genuinely publishable-shaped finding:
  **artifact-corrupted kernels (λ=40.3, learn-obs λ=30.9) lose up to −0.021/−0.028 on downbeat AMLt while F moves by
  only −0.001/−0.005 — "F's −0.001/−0.005 understated the real cost by an order of magnitude. Loose kernels break
  bars, not beats."** This is an in-house, quantified instance of the F-vs-continuity dissociation.

### 4.4 The "98.6" the brief referred to
`transition_lambda = 98.6`, the value the **discriminative CRF** objective converged to on frozen Beat-Transformer
activations. Its significance is a **triangulation** result (98.6 ≈ hand-set 100 ≈ artifact-corrected generative
93.1), not a performance number. It decodes to exactly the same six metrics as the hand-set default.

### 4.5 The ELBO-for-DBN theory
- `vbpm/docs/NeuralDBN.pdf` — Moon-Ryul Jung (Sogang), *"The Observation Distribution and the Posterior Distribution
  for the Bar-Pointer Model, and the Derivation of the EM and Viterbi Algorithms."* Unpublished. Normalizes the
  Böck/Krebs unnormalized scores, derives EM and Viterbi for each BPM variant, and — items (6) and (7) of its own
  abstract — proposes exactly our two extensions: **input-conditioned parameters λ = g_θ(x), λ_v = f_θ(x)**, and
  **a trained neural emission p_θ(b_k|z_k,x) fit jointly with the transition — "a genuine neural DBN in the sense of
  the neural-HMM/state-space literature [Tran et al. 2016; Krishnan et al. 2017]" — trained by generalized EM**
  (M-step has no closed form, so it takes a gradient-ascent step), with a posterior-collapse analysis.
- `ELBO_for_DBN.md` (Jaehoon Ahn, March 2026) — cited as authoritative in `vbpm/docs/SPEC.md` and referenced by
  commit `43ecf34` of a repo not present here. Not recoverable from this checkout; retrieve before writing.
- The VBPM mainline (`tchain`) is a structured chain posterior with an exact forward recursion, tempo in the state —
  i.e. an amortized/VAE treatment of the same object, already at gtzan F 0.750/0.753, CMLt 0.70 / AMLt 0.85.

### 4.6 What in-house work means for novelty
1. The **artifact ("differentiable madmom DBN in PyTorch")** is built and certified. Not novel in-house.
2. The **experiment ("train the frontend end-to-end through it")** has been run against Beat Transformer and lost
   by −0.006 beat F / −0.015 downbeat F, with the caveat in §4.2.
3. The **EM route** has been run and produced a parity result plus a separate (unpublished, defensible) finding about
   grid dithering corrupting the MLE.
4. The **theory** exists as two unpublished in-house documents, one by the professor.
5. **None of this has been run in the regime the paper actually needs** — the target of the Blind Spot line is **SMC**,
   out of domain, tempo-anomalous, where the standing bar is beat F ≥ 0.7 and where *"structured inference has NEVER
   deployably beaten peak-pick beat-F on SMC (0.589 < 0.620)"* (`vbpm/docs/project_state.md`). The bt_e2e results
   above are all **in-domain, fold-0, ~0.95 F**, i.e. the saturated regime where the DBN has nothing left to buy.

---

## 5. VAE / latent-variable approaches to beat or rhythm tracking

Searches returned only music **generation** work (MusicVAE, RAVE, rhythm-generator VAEs, chord-sequence VAEs).
Classical latent-variable beat trackers (Whiteley/Cemgil bar-pointer 2006; Klapuri 2006; Ellis DP 2007) predate deep
learning and are not amortized/trained. The nearest trained analogue is the speech HMM-VAE line (Ebbers 2017;
Glarner 2018).
**VERDICT: NOT-RELEVANT / open field.** *There is no published "deep variational beat tracker."* The in-house VBPM
line is the only occupant of that slot known to us, and it is unpublished.

---

## 6. BOTTOM LINE

### Framings we must ABANDON
1. **"The field has not noticed that dropping the DBN costs continuity."** Dead. Beat This!'s abstract and conclusion
   say it; their Table 2 quantifies it; their §4.3 runs the +DBN ablation.
2. **"The dissociation is caused by a loss that treats beats individually" as our diagnosis.** Dead. Beat This! §4.3
   says exactly this in plain words. We may *formalize* and *test* it; we may not *claim* it.
3. **"Training a neural network through a differentiable HMM forward algorithm is new."** Dead many times over:
   GTN 1998, MMI 1986, CTC 2006, Veselý 2013, LF-MMI 2016, Hadian 2018, Tran 2016, Mensch & Blondel 2018,
   Mehta 2022, torch-struct 2020. Presenting this as a methodological contribution is the fastest possible reject.
4. **"Making the madmom DBN differentiable is the contribution."** Weak on the field's terms (torch-struct/pomegranate
   would do it) and false in-house (already built and certified).
5. **"F-measure vs CMLt/AMLt dissociation is unrecognized."** Dead: Davies, Degara & Plumbley 2009 define it;
   Davies & Böck 2014 characterize it.
6. **Any framing in which the headline result is an in-domain F-measure win.** The in-house run already says the
   e2e CRF arm loses in-domain at ~0.95 F, and Beat This!'s own +DBN row loses F too. Do not fight on that axis.

### What is still defensibly novel
Ranked by strength.

**(A) STRONGEST — the empirical result nobody has: does end-to-end structured training close the SMC blind spot?**
No published system trains a beat tracker through its decoder (§1.12), and the specific question — whether joint
training recovers CMLt/AMLt on *hard, tempo-anomalous, out-of-domain* material (SMC) without the F-measure loss that
post-hoc DBN application incurs — is open. Beat This! themselves defer it: *"a direct comparison with DBN-based
models on complex pieces is left for future work."* The contribution is the **measurement and the regime**, not the
algorithm. This requires a positive SMC number; the standing bar is beat F ≥ 0.7 versus peak-pick 0.620.

**(B) STRONG — "beat tracking has an unfixed 2013-era train/decode mismatch that ASR fixed with LF-MMI."**
A position/analysis contribution: name the mismatch, cite the ASR resolution honestly, and demonstrate the transfer
(including that the ASR *pathologies* transfer — our measured emission domination, and the CRF-saturation failure
that forces a BCE calibration anchor). This is defensible precisely *because* we concede the method is old; the
novelty is that MIR has not applied it and we show what happens when it does.

**(C) MODERATE — the negative/mechanistic result, honestly reported.**
"Training through the DBN does not help in-domain (measured), and here is why: emission domination, CRF saturation
under an uncalibrated frontend, and a likelihood optimum that is not the task optimum." Publishable as analysis if
paired with (A) or (B); not alone at ICASSP.

**(D) MODERATE, INDEPENDENT — the dithering artifact.**
Integer-frame grid quantization corrupts the MLE of `transition_lambda` (λ≈40 reproduced by a perfect metronome;
exact 2× scaling under fps doubling; mixture-kernel repair recovers λ≈93 ≈ the hand-set 100 and the CRF's 98.6).
In-house prior-art check found this unpublished; nearest anticipation is Krebs's thesis §4.3.2. This is a genuine,
self-contained, verifiable finding — arguably the cleanest publishable unit currently in hand — but it is a *different
paper* from the one described in the brief.

**(E) WEAK BUT USABLE AS SUPPORT — the metric-sensitivity point.**
That F-measure under-reports transition-model damage by an order of magnitude relative to downbeat AMLt (in-house
tiebreaker: −0.005 F vs −0.028 dbAMLt) sharpens Davies & Böck 2014 with a concrete mechanism ("loose kernels break
bars, not beats"). Good as a subsection; not a paper.

### Recommended framing sentence
> *Neither the diagnosis nor the algorithm is ours. Foscarin et al. name the continuity deficit and its factorized-loss
> cause; ASR solved the corresponding train/decode mismatch with LF-MMI a decade ago. What is missing is the
> experiment: nobody has trained a beat tracker through its bar-pointer decoder, and nobody has asked whether doing so
> repairs continuity on the hard, out-of-domain material where beat tracking actually fails.*

### Must-verify before submission
- The rematch result of the fixed-harness bt_e2e run (§4.2) — the current negative rests on a harness the authors
  themselves declared confounded.
- `ELBO_for_DBN.md` (commit 43ecf34, another repo) — retrieve; it may itself constrain what we can claim.
- Electronics 11(24):4186 (§1.10) — unverified.
- Porto thesis lineage (repositorio-aberto unreachable).
- Whether Hung et al. (Beat This!'s continuity-metric superior) report anything about their DBN's continuity role.

---

# ADDENDUM (verified 2026-08-20, main session): the unrecorded rematch was run, and it lost by more

`RESULTS.md` records that the first end-to-end CRF comparison had three harness
confounds, that the "likelihood-optimal but F-suboptimal lambda" interpretation was
"CONFOUNDED AND WITHDRAWN pending the obs=6-consistent rematch," and that a
"Rematch launched 2026-07-18 with all three fixes." The document then moves on to the
dithering artifact on 2026-07-20 and **never records the rematch outcome**.

The outcome is on disk, in the non-`_prelim` history files. Best validation
checkpoints (`best` field), Beat Transformer frontend:

| run | file | best beat F | best downbeat F | epoch | learned lambda |
|---|---|---|---|---|---|
| vanilla, first attempt | `vanilla_history_prelim.json` | 0.9613 | 0.9396 | 21 | 100 (fixed) |
| r2 e2e, first attempt | `r2_history_prelim.json` | 0.9471 | 0.9200 | 23 | **14.23** |
| vanilla, rematch | `vanilla_history.json` | **0.9658** | 0.9281 | 29 | 100 (fixed) |
| r2 e2e, rematch | `r2_history.json` | **0.9115** | 0.8573 | 19 | **101.04** |

Identification: these are the rematch runs, inferred from two signatures rather than
an explicit label — the r2 arm's learned lambda converged to **101.04** instead of
14.2 (exactly what the addendum predicted would happen once the CRF trained at
`observation_lambda=6` to match the decode), and the learning rate stays 1.00e-03
through epoch 29 for **both** arms, i.e. the asymmetric annealing was removed.

**The gap widened from 0.014 to 0.054 beat F.** The first attempt's deficit was
plausibly the harness; the rematch's is not. Two independent runs, the second with
the known confounds fixed, both say vanilla BCE + fixed madmom DBN beats end-to-end
CRF training of the frontend.

Three caveats, none of which rescue the result:
1. These are validation figures logged during training, not the `final_eval.py`
   protocol that produced the 0.9506 / 0.9446 numbers in `RESULTS.md`. The direction
   is unambiguous but the magnitudes are not directly comparable to that table.
2. Still a single seed and, per `RESULTS.md`'s own "Next" list, not yet run over
   multi-seed / all 8 folds — which that list explicitly required "before any strong
   claim."
3. **Regime.** This is an easy fold (F ≈ 0.95). Our thesis was always about hard,
   tempo-anomalous, out-of-domain material. The negative therefore does not *refute*
   the SMC hypothesis — it was measured where we predicted no gain. But it does mean
   the proposal has now failed twice in the only regime where we have data.

Also relevant from `RESULTS.md`: the known failure mode is that "pure CRF saturates
the frontend in one epoch, forcing a BCE anchor" — so the e2e arm is not a clean
structured objective but a BCE/CRF blend, and the CRF term's contribution is small
(`crf 0.055` against `loss 0.365`).
