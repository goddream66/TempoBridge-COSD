# Physiological references and downstream evaluation

These commands are engineering infrastructure. Generated smoke data and heuristic
reference targets do not establish clinical accuracy or COSD efficacy.
All paths below are relative to the repository, or may be replaced with absolute
paths. No runtime code assumes a drive letter or a particular data location.

## Reference targets

Keep native-rate reference waveforms separately from the 50 Hz PPG model input.
Prepare a JSONL file with one row per **exact full window**:

```json
{"sample_id":"window_001","path":"reference_windows/window_001.npz","duration_seconds":30}
```

Each NPZ may contain `ecg, ecg_fs`, `abp, abp_fs, abp_units`, and/or `resp, resp_fs`.
Sampling rates are explicit; waveforms must span the declared window. `abp_units`
must be exactly `mmHg`. This input is not inferred from normalized PPG shards.

```text
python -m tempobridge reference-targets --references data/references.jsonl --output data/targets.csv
```

The output is a long CSV keyed by `(sample_id,target)`. Required import columns:
`sample_id,target,value,valid,units,reference,method,duration_seconds`.
The built-in extractor additionally records reason, reference file SHA256 and rate.
An absent or invalid target contributes **no loss**. Its storage placeholder is not
a zero-valued measurement. Duplicate keys, mixed units and mismatched window durations fail.
External/adjudicated scalar labels use the same format. Classification values must
be numeric class identifiers; do not encode unverified rhythm diagnoses automatically.

Implemented proxies:

- `hr_bpm`: QRS energy detection, refined ECG peak positions, median plausible RR.
- `rmssd_ms`: adjacent plausible RR differences, without bridging rejected intervals;
  requires >=100 Hz, >=30 seconds, >=20 adjacent interval pairs. This is an ultra-short
  proxy, not interchangeable with standard long-recording HRV. MIMIC timing jitter remains.
- `sbp_mmhg`, `dbp_mmhg`, `map_mmhg`: ABP maxima, inter-peak minima and complete-beat mean.
- `rr_bpm`: dominant respiratory frequency within 0.08–0.7 Hz, >=30 seconds.

These are explicit heuristics, not validated detectors. Invalid references remain
masked; no PPG fallback, label imputation, rhythm diagnosis or synthetic clinical
ground truth is substituted. Inspect representative real windows and detector errors
before using these targets as scientific reference measurements.

`preprocess` also stores `*_raw_missing_fraction`, `*_raw_flatline_fraction`,
`*_raw_extreme_occupancy` and spectral diagnostics before interpolation/filtering.
Extreme occupancy is only a clipping proxy, and the existing ECG quality weight
is still finite coverage, not a calibrated SQI. Converter v2 changes the conversion
contract: resume old v1 output using the old version, or write v2 to a fresh directory.

## Direct physiological supervision baseline

Set `train.transfer_method="physio"`, `transfer_mode="all"`, and select target names
in `train.physio_targets`. Full-window mode is required. For a fair comparison,
continue from the same PPG warm-up and match the continuation budget:

```text
python -m tempobridge pretrain --config configs/physio_30s.json --manifest data/manifest.json --modality ppg --student runs/warmup/last.pt --reference-targets data/targets.csv --output runs/physio
```

No ECG neural teacher is loaded. Target means/scales use known training groups with
patient-balanced weights. The masked Huber loss uses fixed target-count and batch
denominators. PPG SSL remains active, and `protect_shallow` uses the same gradient
boundary as embedding KD. Changing the target CSV changes resume provenance.
Missing reference targets do not require discarding PPG windows.

## Export, frozen probes and label efficiency

```text
python -m tempobridge export --checkpoint runs/cosd/last.pt --output runs/cosd/encoder.pt
python -m tempobridge features --checkpoint runs/cosd/encoder.pt --manifest data/manifest.json --split train --output runs/cosd/train.npz --device cpu
python -m tempobridge features --checkpoint runs/cosd/encoder.pt --manifest data/manifest.json --split val --output runs/cosd/val.npz --device cpu
python -m tempobridge features --checkpoint runs/cosd/encoder.pt --manifest data/manifest.json --split test --output runs/cosd/test.npz --device cpu
python -m tempobridge evaluate-probe --train-features runs/cosd/train.npz --val-features runs/cosd/val.npz --test-features runs/cosd/test.npz --labels data/targets.csv --target hr_bpm --output runs/cosd/hr_probe
```

Feature archives include source, record, quality, identity level, split, encoder hash
and pretraining provenance. Old archives without this metadata must be regenerated.
Unknown pretraining data is reported; it is never silently declared disjoint.

Regression uses patient-weighted ridge with train-only feature normalization.
Classification uses a linear softmax head; use `--task classification` and a matching
target. Validation patient-macro MAE or patient-balanced log loss chooses regularization.
No test score selects a head or hyperparameter. Classification thresholds are argmax,
not test-optimized. Undefined AUROC/AUPRC is null, not replaced by a favorable number.

`--label-fraction .01/.05/.1/.25/1` selects whole training groups, rounded upward,
with reproducible seeded group ordering. It is a **group fraction**, not a window fraction.
If selected groups omit a required class, the run fails explicitly instead of silently
resampling until an easier split is found. Reports contain the selected IDs and counts.

`--protocol source_held_out --target-source vitaldb` requires the target source to be
absent from student pretraining, teacher pretraining, downstream train and validation.
Use source-specific manifests/features. Joint-source pretraining with unseen patients
is group-disjoint, not unseen-source zero-shot. If only case/record identity is known,
the result must be described as case/record-disjoint; metadata retains that limitation.

## Full fine-tuning and M0

```text
python -m tempobridge finetune --config configs/ssl_30s.json --manifest data/manifest.json --labels data/targets.csv --target hr_bpm --encoder runs/cosd/encoder.pt --output runs/cosd/hr_finetune
python -m tempobridge finetune --config configs/ssl_30s.json --manifest data/manifest.json --labels data/targets.csv --target hr_bpm --output runs/random/hr_finetune
```

Omit `--encoder` only for M0 random initialization. Fine-tuning loads PPG only,
updates the entire encoder and a linear task head, and selects the epoch using
validation alone. Regression target scaling is train-only. Config controls learning
rate, epochs, batch size, accumulation, precision and device. Use identical settings
and a predeclared validation search across methods. This command currently runs a
complete downstream trial; epoch-resume is supported by pretraining, not this command.
`best.pt` is a task model (encoder + task head), not a generic encoder export.

## Paired comparisons and uncertainty

```text
python -m tempobridge compare-predictions --first runs/cosd/hr_probe/predictions.npz --second runs/pw/hr_probe/predictions.npz --output runs/comparison.json
```

The comparison checks identical sample support, identities, labels and class mapping,
aligns by sample ID, and bootstraps whole groups together. Differences are **first minus
second**: negative favors the first model for error metrics, positive for AUROC/F1.
Intervals are conditional on trained models; repeat independent pretraining seeds to
measure training variation. Do not count downstream-head seeds as pretraining seeds.
Reports separate sources rather than averaging incompatible units across tasks.

CPU-only end-to-end check (fresh output directory):

```text
python scripts/smoke_downstream.py --output runs/downstream_smoke
```
