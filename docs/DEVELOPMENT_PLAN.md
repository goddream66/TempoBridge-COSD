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
- Iteration 2 implemented: mmap feature cache, schema-3 calibration, patient-disjoint
  basis/discovery/audit, conditional cluster bootstrap, hard/soft selection and rank-zero fallback.
- Iteration 3 core implemented: projected targets, vanilla statistics without probes,
  shared-warm-up SSL continuation, fixed-budget checkpoint selection, gradient diagnostics,
  full/PCA/random/soft configurations and real CLI smoke runner.
- CPU full suite: 218 passed (14 existing dependency deprecation warnings); Ruff passed.
- Five-method 30-second synthetic CLI verification is in `runs/cosd_validation/core_cli_20261107`.
  `summary.json` confirms completion, including a nonzero COSD rank of 4 and finite KD gradients.
  This is not physiological efficacy evidence. A subsequent paired-count logging correction passed
  all 15 COSD training/gradient/resume/export tests; Ruff and `git diff --check` passed.
- Iteration 3 controls implemented: dataset-level within-record/cross-patient/cross-source/time-shift
  donors; matched correct-pair coverage and conservative common ECG reliability; real timestamps;
  deterministic source/group/record/window sampling. Initial targeted checks: 45 passed. Follow-up
  10 control tests passed, including bit-exact interrupted/resumed training on all three encoders.
- Iteration 4 implemented: raw quality diagnostics (converter contract v2); native-rate
  ECG/ABP/RESP heuristic references and external scalar label import; masked physiological
  supervision with train-only normalization and exact pretraining resume; metadata-rich exports;
  frozen regression/classification probes, whole-group label fractions, full fine-tuning and
  random initialization; paired group bootstrap and source-held-out provenance guards.
- CPU full suite: 251 passed, 14 existing dependency warnings, 79.03 seconds. Separate downstream
  CLI completed reference extraction, physiological pretraining, export, all-split features,
  regression/classification probes, paired comparison and pretrained/random full fine-tuning:
  `runs/cosd_validation/downstream_cli_20261107/summary.json`. These are synthetic fixtures only.
- Follow-up resume-metadata and paired-support validation changes: 29 targeted checks passed;
  all 19 experiment configurations load; Ruff and `git diff --check` passed.
- Directory migration completed to `D:/TempoBridge-COSD` after the user closed the
  occupying VS Code project. The local editable installation and Windows Python
  command launchers were repaired; package imports now resolve to the new source.
  Development package version is `0.4.0.dev1`; Python import name remains `tempobridge`.
  Wheel build, independent-directory wheel/encoder loading and `pip check` passed.
- Post-relocation full suite: 251 passed, 14 dependency warnings, 81.32 seconds;
  relocated wheel import and encoder loading passed again. Old project directory
  no longer exists; runtime artifacts/environments remain excluded from Git.
- Final iteration implemented: representation/identity/source/quality/within-record audits;
  predeclared experiment families; actual per-scale rank matching, sample/direction weight controls,
  paired/unpaired subsets, native teacher readout/random controls, guarded source-held-out evaluation,
  per-layer gradient/update diagnostics and source/quality supervision totals.
- Experiment execution verifies source/inputs/output hashes, shares compatible warm-up/cache stages,
  retains interrupted non-training products and resumes training at epoch checkpoints. Reports separate
  seed variation from paired patient bootstrap; stages record wall-clock and sampled process RSS.
- Downstream reports include quality/availability/case/device strata; PPG feature noise is reproducible
  per sample. PPG-only paths preserve ECG quality metadata without reading ECG waveform arrays.
  Converter v3 records shared-grid timing explicitly without claiming verified physiological delay.
- Final CPU suite: 277 passed, 14 existing dependency warnings, 113.75 seconds. Ruff, whitespace checks,
  all 19 configurations, and pip dependency checks passed. Seven-method CLI and verified job reuse:
  `runs/cosd_validation/matrix_release_20261107/summary.json`.
- Version 0.4.0 wheel built and imported outside the repository; relative-path exported encoder load
  produced finite [2,32] features. Evidence: `runs/cosd_validation/release_portability_20261107.json`.
  CPU benchmark also exercised inference latency, teacher overhead and partial profiler FLOPs.
- Authorized engineering scope is complete; `REQUIREMENT_AUDIT.md` maps the source specification to
  implementation and limitations. Real-data teacher validation, full three-seed scientific experiments,
  clinical reference review and Go/No-Go conclusions remain unperformed research work, not test results.

## Organization references

- Mantis: https://github.com/vfeofanov/mantis — src layout, tests, changelog and compatibility notes.
- Semiparametric distillation: https://github.com/microsoft/semiparametric-distillation —
  separate experiment configurations and training/evaluation entry points.

These references inform repository organization; this task does not copy their implementations.
