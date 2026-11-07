# Changelog

## 0.4.0 — 2026-11-07

- Shared final-context transfer feature contract for calibration and KD.
- Real HDF5 recording/time metadata and explicit unknown quality provenance.
- Versioned, fingerprint-verified, memory-mapped frozen feature caches.
- COSD target calibration with separate basis/discovery/audit patients, repeated OOF probes,
  conditional patient bootstrap, original/PCA/random bases and rank-zero abstention.
- Direct low-dimensional ECG targets, independent weighting, soft direction-loss weights,
  vanilla statistics-only calibration and shared warm-up SSL continuation.
- Fixed-step checkpoint option and sampled encoder-gradient diagnostics.
- Dataset-level pairing controls, matched coverage/reliability and deterministic patient sampling.
- Raw quality diagnostics, native-rate reference proxies and masked physiological supervision.
- Provenance-rich features, frozen probes, full fine-tuning, group label fractions,
  patient-cluster intervals, paired comparisons and strict source-held-out guards.
- CPU tests for identity projection equivalence, patient isolation, cache relocation,
  exact resume and PPG-only export; core and downstream synthetic CLI verification.
- Frozen representation audits for identity, source, quality, physiology and within-record variation.
- Reviewable experiment plans, common budgets/warm-up, actual per-scale rank matching,
  independent reliability axes, direction-weight matching, source-only views and paired/unpaired subsets.
- Native teacher layer/pool/output normalization controls and explicitly marked random-teacher diagnostics.
- Per-layer gradient/update norms, supervision totals by source/quality, seed summaries and paired comparisons.
- Quality/availability/case/device evaluation strata, reproducible PPG noise perturbations,
  stage wall-clock/RSS and inference/teacher latency with partial profiler FLOP counts.
- Converter v3 records shared-grid timing status without asserting verified interchannel delay.
- Content-verified resume and interrupted-stage retention, experiment CLI smoke and repository relocation.

Engineering coverage and unperformed real-data work are distinguished in `docs/REQUIREMENT_AUDIT.md`.
No clinical performance claims or completed scientific Go criteria.

## 0.3.0

- Selective receptive-field tokenizer and default full 30-second input protocol.
- Optional 2/8-second crops and tokenizer ablations.

## 0.2.0

- Multiview encoder and checkpoint-compatible local encoder.
- MIMIC/VitalDB preprocessing and HDF5 ingestion.
