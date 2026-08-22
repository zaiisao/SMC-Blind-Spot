# Working notes

Research notes written alongside *The SMC Blind Spot* and the follow-up work. These are
internal working documents, not part of the paper artifact — the reproducible results live in
the root notebook, [data/](../data/) and [scripts/](../scripts/).

## [spec/](spec/)
- [SPECIFICATION.md](spec/SPECIFICATION.md) — tempo observation model for the bar-pointer DBN;
  every claim tagged PROVEN / MEASURED / CONJECTURE.

## [findings/](findings/) — experiment results
- [FOLLOWUP_FINDINGS.md](findings/FOLLOWUP_FINDINGS.md) — blind spots and inference-time gains
  on SMC, all recomputed through the shared harness (`scripts/_harness.py`).
- [OVERNIGHT_BATCH.md](findings/OVERNIGHT_BATCH.md) — six batched experiments and their outputs.
- [STICKY_REGIME.md](findings/STICKY_REGIME.md) — pre-registered falsification test of a sticky
  two-regime tempo transition kernel.
- [TCHAIN_RESCORE.md](findings/TCHAIN_RESCORE.md) — re-score of the GTZAN downbeat result.
- [ADVERSARIAL_OBS_LAMBDA.md](findings/ADVERSARIAL_OBS_LAMBDA.md) — adversarial review of
  `scripts/experiments/exp_obs_lambda.py`.
- [KVAE_GENERATIONS.md](findings/KVAE_GENERATIONS.md) — what the two KVAE generations encoded
  and what survives.

## [audits/](audits/) — claim verification
- [CLAIM_AUDIT.md](audits/CLAIM_AUDIT.md) — the ledger; every number marked CONFIRMED /
  CONTRADICTED / REQUIRES-RERUN.
- [PHASE0_RECERT.md](audits/PHASE0_RECERT.md) — re-certification of the VBPM machinery after the
  move into this repo.
- [LATE_CORRECTIONS.md](audits/LATE_CORRECTIONS.md) — facts established after the audit workflow
  launched; pending merge into the ledger.

## [related/](related/) — prior art
- [PRIOR_ART_NEURAL_DBN.md](related/PRIOR_ART_NEURAL_DBN.md) — prior-art check on the
  differentiable-DBN claim.
- [RELATED_WORK_BRIEF.md](related/RELATED_WORK_BRIEF.md) — related work the ISMIR 2026 reviewers
  asked for.

## [vbpm/](vbpm/) — carried over from the VBPM project
- [HANDOVER_FROM_VBPM.md](vbpm/HANDOVER_FROM_VBPM.md) — turning the analysis into a solution paper.
- [VBPM_ASSETS_FOR_ARGUMENT.md](vbpm/VBPM_ASSETS_FOR_ARGUMENT.md) — assets supporting the
  ICASSP 2027 argument.
