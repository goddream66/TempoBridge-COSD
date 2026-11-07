# TempoBridge-COSD implementation ledger

Source specification: `../TempoBridge_COSD_完整研究方案.md` (preserved as supplied).
This ledger distinguishes executable research infrastructure from unverified scientific claims.
No real-data training or GPU training is authorized for this development task; verification uses CPU fixtures.

## Iterations and acceptance evidence

1. **Representation and identity contracts**: one last-layer context-mean transfer API;
   frozen teacher feature specification; calibration schema migration; real record/time metadata
   with explicit unknown states. Test gradient boundaries and HDF5 metadata propagation.
2. **Calibration and target selection**: reusable feature cache with content fingerprints;
   full-target PW-KD and direction-wise COSD; original/PCA/random bases; patient-balanced
   probes; fit-only normalization; separate basis/discovery/audit patients; bootstrap stability;
   rank-zero abstention; versioned frozen transformations. Test synthetic signal/noise and leakage.
3. **Training and controls**: independent target and weighting options; vanilla/quality/PW/COSD;
   matched-rank controls; paired/unpaired operation; reproducible pairing controls; continuation
   from shared warm-up; exact resume; fixed-step checkpoint option; effective supervision metrics.
4. **Data and evaluation**: optional patient-aware sampling; explicit quality diagnostics;
   reference-derived physiological targets; frozen probes, label fractions, fine-tuning support,
   patient-cluster intervals, paired method comparisons, source-held-out validation and auditing.
5. **Repository and delivery**: configs and experiment matrix; documented CLI; method and migration
   docs; changelog; CI; synthetic end-to-end verification; package/relocation checks; rename root
   to `TempoBridge-COSD`; verify paths and local environment; clean, reviewable Git commits.

## Research boundaries

- Coordinate selection is a baseline, not proof of a physiological shared subspace.
- OOF refers to probes on fixed pretrained representations, not an OOF backbone.
- Independent audit patients cannot determine directions, thresholds or normalization.
- Projection targets come from real ECG teacher outputs, never only probe predictions.
- Sample weights and target rank are separate experimental axes; fixed batch denominator.
- Patient/source/quality/record confounding, random projection and explicit physiology baselines
  are required evaluation controls. Low distillation loss is not evidence of downstream benefit.
- Native-rate labels/external teachers must declare their input protocol. No invented clinical labels.
- No empirical Go criterion in the source specification is considered met by synthetic tests.

## Current status

- Baseline inspected at `fa10bfe`; source document was the only untracked user file.
- Confirmed 512-dimensional calibration versus 256-dimensional KD features.
- Confirmed HDF5 record ID currently aliases patient/group ID.
- Iteration 1 implemented: shared transfer feature API, calibration schema v2 (v1 requires
  regeneration), explicit teacher feature metadata, real HDF5 record/time metadata and unknown flags.
- CPU checks: 67 targeted tests passed; Ruff passed. Initial pytest attempt failed to create
  sandbox temporary directories; rerun used an explicit workspace basetemp and passed.
- Iteration 2 next. Remaining iterations remain required before completion.

## Organization references

- Mantis: https://github.com/vfeofanov/mantis — src layout, tests, changelog and compatibility notes.
- Semiparametric distillation: https://github.com/microsoft/semiparametric-distillation —
  separate experiment configurations and training/evaluation entry points.

These references inform repository organization; this task does not copy their implementations.
