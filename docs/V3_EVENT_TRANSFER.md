# TempoBridge V3-A / V3-B

V3-A trains PPG reconstruction, pulse events and ECG-derived RR intervals. V3-B
adds retrospective changes between adjacent windows. Both use the existing
30-second, 50 Hz TempoBridge backbone and update the entire PPG encoder.

## Native reference windows

Create a JSONL file with one row per available reference window, including held-out
windows for evaluation. Each row must refer to the exact sample ID, time range and
channels of its model window. The HDF5 index determines HDF5 sample IDs.

```json
{"sample_id":"window_001","path":"native/window_001.npz","duration_seconds":30,"source_id":"mimic","subject_id":"p001","record_id":"record_001","case_id":"","window_start_s":60,"window_end_s":90,"ecg_channel":"II","ppg_channel":"PLETH","ecg_native_fs":125,"ppg_native_fs":125,"native_timestamp_origin":"record start, seconds","clock_alignment_status":"verified_common_clock","max_alignment_error_s":0.008,"clock_evidence":"local timebase audit ID for record_001"}
```

Each NPZ contains `ppg`, `ppg_fs` and, where available, `ecg`, `ecg_fs` from the
original recording. Paths resolve relative to the JSONL file or may be absolute.
Existing 50 Hz shards cannot recover the original signal samples. Preparing these
references from MIMIC WFDB or VitalDB requires the original recordings and a
verified mapping from each model window to the native recording and time range.

Use the model manifest's exact IDs. For HDF5, use the canonical `subject_id` and
`recording_id` emitted by the dataset metadata. Use `verified_common_clock` or
`verified_offset_corrected` only after a timebase check. The latter expects the
reference arrays to be corrected already. The builder checks identity, window
bounds, both NPZ sample rates, channel names, a time origin, evidence ID, and a
declared residual timing bound of at most 20 ms. `clock_evidence` is a reference
to the actual verification record. It is the data provider's evidence, not a
clock measurement made by the builder. A shared SDK grid is insufficient.

The table stores the reference SHA-256, provenance, and manifest digest. An
optional `reference_sha256` in the JSONL must agree with the actual file. The old
boolean `pairing_verified` no longer grants RR eligibility, including on schema-1
tables. Rebuild these tables for RR training. Rows with `unverified` or omitted
clock status remain available for pulse events and window-level ECG metrics.

The reference detector uses QRS-energy candidates and PPG trough/peak proxies.
The extractor abstains on gaps, flatlines or native rates below 50 Hz. RMSSD is a
short-window RR-derived proxy with >=100 Hz, >=30 s and >=20 eligible adjacent
interval pairs. By default V3-B uses RMSSD changes only when both windows have
annotated NN intervals. With no such annotations, HR can still provide dynamic
supervision. `event.rmssd_reference="allow_rr_proxy"` enables the explicitly
named proxy control. Each pair still requires matching RMSSD definitions.

## Audit reference consistency

Use `audit-events` to compare detections on the actual model PPG input and the
native PPG reference. It samples up to 500 windows per source by default and
stratifies by quality, reference HR, rhythm tag and annotation availability.
Use the resulting `review_windows` to locate windows and timestamps for review.
The native/model comparisons measure preprocessing agreement.

For independent reference accuracy, add a separate JSONL with independently
annotated events in seconds from window start. The annotation must include the
actual `reference_sha256` from the review row. The following illustrates the
format with only two events. `nn_valid` has one boolean per adjacent R-peak pair.
`pat_r_index` has one annotated R index per foot, or `-1` when unmatched.

```json
{"sample_id":"window_001","annotation_source":"review batch 01","reference_sha256":"<actual SHA-256>","rpeaks_s":[0.5,1.4],"feet_s":[0.7,1.6],"peaks_s":[0.9,1.8],"nn_valid":[true],"pat_r_index":[0,1],"rhythm":"regular"}
```

```powershell
python -m tempobridge audit-events --manifest data/manifest.json --event-targets data/events_v3.h5 --config configs/v3_beat_30s.json --output runs/reference_review.json
python -m tempobridge event-targets --manifest data/manifest.json --references data/native_windows.jsonl --annotations data/manual_events.jsonl --config configs/v3_beat_30s.json --output data/events_reviewed.h5
python -m tempobridge audit-events --manifest data/manifest.json --event-targets data/events_v3.h5 --manual-targets data/events_reviewed.h5 --config configs/v3_beat_30s.json --output runs/reference_audit.json
```

The manual comparison reports event precision/recall and timing MAE, interval MAE,
and PAT error where an independently annotated correspondence and timing evidence
exist. Coverage is reported separately. Use `evaluate-events --require-manual`
with the reviewed table to score only annotated held-out windows. For independent
RR evaluation, provide the reviewed ECG-to-PPG correspondences in `pat_r_index`.

## Train and evaluate

Install `.[data]` for HDF5 support. Reuse a trained PPG warm-up checkpoint with the
same corpus and input protocol. The example configs default to CPU. Change device,
precision and budget in a copy before an actual training run.
The default patient sampler needs at least one complete effective batch per epoch
(32 windows with the supplied settings). For smaller engineering fixtures, reduce
batch size and accumulation or set a compatible `samples_per_epoch` explicitly.

```powershell
python -m tempobridge event-targets --manifest data/manifest.json --references data/native_windows.jsonl --config configs/v3_beat_30s.json --output data/events_v3.h5
python -m tempobridge train-events --manifest data/manifest.json --event-targets data/events_v3.h5 --config configs/v3_beat_30s.json --student runs/warmup/last.pt --output runs/v3_a
python -m tempobridge train-events --manifest data/manifest.json --event-targets data/events_v3.h5 --config configs/v3_dynamic_30s.json --student runs/v3_a/last.pt --output runs/v3_b
python -m tempobridge evaluate-events --manifest data/manifest.json --event-targets data/events_v3.h5 --checkpoint runs/v3_b/last.pt --split val --output runs/v3_b/events_val.json
python -m tempobridge evaluate-events --manifest data/manifest.json --event-targets data/events_v3.h5 --checkpoint runs/v3_b/last.pt --split test --output runs/v3_b/events_test.json
python -m tempobridge export --checkpoint runs/v3_b/last.pt --output runs/v3_b/encoder.pt
```

`--stop-after 1` and `--resume <run>/last.pt` use the existing epoch checkpoint
contract. Keep the original config, student and references arguments on resume.
Starting B from A copies its event heads. Starting B directly from the common
warm-up provides a separate control. B initializes its window change head in both cases.

The event head refines and upsamples patch tokens to the model sample grid. Its
`ppg_quality_proxy` output learns window-level PPG quality around labeled events.
`quality_proxy_weight` controls this objective. Old `confidence_weight` configs
are accepted as an alias. RR uses local soft pooling with the configured
`event.rr_anchor_mode`: `reference`, `predicted`, or `mixed`. Evaluation always
uses predicted feet. Predicted training uses the same decoder and threshold as
evaluation by default. It aligns predictions to reference feet within
`rr_anchor_tolerance_s` and drops unmatched/non-consecutive intervals. `mixed`
ramps the predicted-window fraction from zero over `rr_anchor_ramp_epochs`.
`metrics.jsonl` distinguishes eligible and actually used RR labels. Pure predicted
training can start from a previously trained event checkpoint to avoid losing
most RR supervision early. A valid interval requires two consecutive PPG events paired to two
consecutive ECG events. Each event must have a unique candidate within the delay
range. The delay range is a configurable engineering filter. ECG timestamps are
used to build labels and do not enter the student. Timing is resolved on the 20 ms
model grid. RR predicts a bounded Gaussian distribution, with Huber as a control.

Stage B pairs consecutive non-overlapping windows within the same source, group,
record and case. It checks duplicate starts, gaps, device metadata, ECG availability
and signal quality. The default target is ECG-derived HR/RMSSD change between the
two windows. This is a retrospective estimate using both PPG windows. For a separate
latent control, set `event.dynamic_target` to `latent` and pass a frozen ECG
checkpoint with `--teacher`. This trains against the change in its window embeddings.

`metrics.jsonl` reports supervision coverage, parameter updates and gradient
cosines for shared backbone parameters, tokenizer, shallow/context blocks and
the heads where both gradients exist. `evaluate-events` reports pulse F1,
conditional RR errors and coverage. Gaussian RR also reports NLL, observed
50/80/90/95% interval coverage, interval width and mean coverage error. Dynamic
metrics include Pearson/Spearman, change direction, magnitude strata and a paired
group bootstrap against the zero-change baseline.
Choose the event threshold on validation and freeze it for test. Review identity
levels for case/record-only groups. Mixed-corpus source reports are within-corpus
held-out scores. Use the existing source-held-out probe protocols for separate
encoder generalization experiments.

`export` keeps the existing PPG encoder API. To run the event heads on a PPG
tensor, use `load_event_model` and `predict_events` in `tempobridge.event_evaluation`.
For downstream comparisons, use the V3 `encoder.pt` with the `features`,
`evaluate-probe` and `finetune` commands in [EVALUATION.md](EVALUATION.md).
Use the same downstream labels and split files for SSL, COSD and V3 encoders.
This evaluates the exported representations independently of their auxiliary heads.

## Experiment gates

Match initialization, splits and total optimizer steps across PPG SSL, window KD,
direct HR supervision and A/B. Use the event weights and `event.rr_loss` for
foot/peak/RR/uncertainty ablations. Compare staged A→B against direct B and compare
metric changes against latent changes in separate runs. Report RR error with
detection coverage and use the exported encoder to measure downstream tasks beyond
the training targets.

V3-C soft partial correspondence, unmatched-event learning and PAT uncertainty
require held-out evidence from A/B. V3-D dynamic subspace and optional COSD follow
the matching experiment. The CPU fixtures establish execution and data boundaries.
Real-data experiments are needed to evaluate the scientific gates.
