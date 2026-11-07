# Changelog

## 0.4.0 — in development

- Shared final-context transfer feature contract for calibration and KD.
- Real HDF5 recording/time metadata and explicit unknown quality provenance.
- Versioned, fingerprint-verified, memory-mapped frozen feature caches.
- COSD target calibration with separate basis/discovery/audit patients, repeated OOF probes,
  conditional patient bootstrap, original/PCA/random bases and rank-zero abstention.
- Direct low-dimensional ECG targets, independent weighting, soft direction-loss weights,
  vanilla statistics-only calibration and shared warm-up SSL continuation.
- Fixed-step checkpoint option and sampled encoder-gradient diagnostics.
- CPU tests for identity projection equivalence, patient isolation, cache relocation,
  exact resume and PPG-only export; synthetic CLI verification.

Remaining work is tracked in `docs/DEVELOPMENT_PLAN.md`. No clinical performance claims.

## 0.3.0

- Selective receptive-field tokenizer and default full 30-second input protocol.
- Optional 2/8-second crops and tokenizer ablations.

## 0.2.0

- Multiview encoder and checkpoint-compatible local encoder.
- MIMIC/VitalDB preprocessing and HDF5 ingestion.
