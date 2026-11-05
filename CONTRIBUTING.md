# Development

Use a branch for each change. Keep generated data, checkpoints and local environments out of Git.
Install `.[dev]`, run `ruff check .` and `pytest -q`; run `scripts/smoke.py` for pipeline changes.

Before changing normalization, masking, cropping, gradient routing or calibration, state the invariant
and add a behavioral test. Do not replace patient-level splits with random windows.
Never label synthetic integration numbers as physiological or downstream efficacy results.

Use configs and explicit seeds. Save provenance, environment and metrics alongside each run.
Update METHOD.md when changing equations or numerical conventions. Reference reused code and its license.
Do not add external services, telemetry, credentials or automatic dataset downloads to training.
