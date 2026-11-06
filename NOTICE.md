# Sources and implementation provenance

TempoBridge source is independently written for this project. No pretrained weights are bundled.
No PulseBound or Mantis source files were copied into the package.

Design references inspected on 2026-11-05:

- MantisV2: https://github.com/vfeofanov/mantis , particularly architecture/version2.py and
  transformer_v2_utils/transformer.py. Public repository LICENSE is Apache-2.0 at inspection time;
  its component choices inspired this independently implemented Transformer. No checkpoint compatibility is claimed.
- PulseBound: user-supplied paper and supplementary PulseBound_ICLR source; v6/model.py,
  v6/physio_targets.py, pretraining/scaffold_losses.py and masking implementation were read to assess overlap.
  This is an analysis/design reference, not vendored code. The supplied archive is not redistributed here.
- PyTorch public API documentation for scaled_dot_product_attention and model/optimizer serialization.

The project name is a working research name, not a claim of global naming uniqueness.
See docs/RESEARCH.md for related-work overlap and limits of novelty claims.

Data compatibility reference inspected on 2026-11-06:

- Supplied PulseBound `preprocessing/processor.py`, `preprocessing/quality.py`,
  `preprocessing/physio_extractors.py`, `v6/group_split.py`, and `v6/data.py`.
  The identifier normalization, hash namespaces, default split salt, ranking and
  90/5/5 integer apportionment protocol are reproduced for wire compatibility.
  These protocol constants and formulas are not claimed as original contributions.
  No upstream source files or clinical datasets are bundled. See docs/PULSEBOUND_DATA.md
  for deliberate cleaning differences and limits of the new VitalDB converter.
- WFDB official I/O documentation: https://wfdb.readthedocs.io/en/latest/io.html
- VitalDB official local-file library: https://github.com/vitaldb/vitalutils
