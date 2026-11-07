# TempoBridge-COSD: executable method contract

This document describes the implementation, not a claim of clinical efficacy.
The original research proposal is preserved at the repository root. Changes to
its statistical details below avoid ambiguous projections and post-selection claims.

## Shared representation

`TempoBridge.transfer_features` always returns the last context layer's patch mean.
It is used by calibration and distillation. Exported `feature_pool` is independent;
the default exported representation remains `[summary; context_mean]`.
Native ECG teachers retain their declared context layer and pooling convention.

## Frozen feature cache

`cache-features` extracts only known training groups with valid ECG passing the
configured quality threshold, using deterministic evaluation crops. Arrays are float32
memory maps. The manifest stores actual row count, hashes of arrays and row metadata,
checkpoint fingerprints, input protocol, feature definitions and sampling settings.
Incomplete caches have no manifest and cannot be loaded. Capacity beyond actual rows
is unused. Paths inside a cache are relative; moving the directory preserves identity.
Changing checkpoints, waveforms, splits or the extraction settings requires a new cache.

## Calibration schema 3

Schema 1 is rejected because it used the old calibration feature contract. Schema 2
can still be used for full-target KD; COSD requires schema 3.

For the same cache, seed and COSD partition settings, vanilla, PW-KD and COSD use the
same final teacher centering/scaling fitted to the basis-patient subset. With fewer
than three patients, only full-target statistics can fall back to all training patients.
Probe fitting always uses its own fit-fold statistics, separate from final target scaling.

PW-KD retains patient-macro, fit-fold-null-normalized MSE and conservative probe consensus.
Vanilla with `transfer_mode=all` or `quality` computes target statistics without OOF probes.

COSD partitions identified training groups into three disjoint sets, deterministically:

1. Basis patients fit the mean, scale and candidate basis (identity/PCA/random orthogonal).
2. Discovery patients supply repeated patient-fold OOF direction errors and selection.
3. Audit patients measure the frozen selection's prediction skill. They cannot alter it.

Direction skill is `1 - patient_macro_error / patient_macro_null_error`. The null uses
each probe fit fold's teacher mean. Consensus takes the minimum skill over probe families.
The stability option selects using a patient-cluster bootstrap lower confidence bound.
These intervals are conditional on the fitted OOF probes, not confidence intervals for
the entire encoder/probe-learning process. OOF does not mean the encoders were trained
without the score patients. PCA directions can rotate inside degenerate eigenspaces.

Top-k retains only nonconstant candidates whose criterion exceeds `min_skill`, up to
the requested rank. Empty selection is valid and disables KD. `all`, `random` and PCA
`variance` selection are explicit controls, not predictability-qualified directions.
Each run reports requested/actual rank, retained energy, patient partitions, fold IDs,
direction scores/intervals and independent audit results per source.

## Target and loss

For row-vector teacher output `e`, the target is `u = ((e - mean) / scale) @ V_r`.
The student head predicts r coordinates directly. V_r has orthonormal columns; the
normalization and projection are frozen buffers. The target comes from the real ECG
teacher, not the PPG probe prediction. Zero-filled D-dimensional targets are not used.

`KD = sum_i sample_weight_i * mean_j(direction_weight_j * Huber(pred_ij, u_ij)) / batch_size`.
Hard selection uses direction weights 1. Soft selection clips discovery skill into [0,1]
and applies it only to the loss, not again to the target. There is no division by the
sum of sample weights. Target rank and sample weighting are separate experimental axes.

PPG masked reconstruction remains active for paired and unpaired samples. The ECG
teacher, calibration and projection have no gradients. With shallow protection, only
context layers and transfer heads receive KD gradients. SSL still reaches the tokenizer.
Inference export contains only the PPG encoder, with no calibration/ECG input dependency.

## Remaining research requirements

Use matched initialization, sample sequence, steps, target scaling and checkpoint rules.
`checkpoint_selection=last` selects the fixed-budget endpoint; reconstruction selection
is retained as a legacy option. A lower KD loss is not a downstream result. Random-rank,
explicit physiological supervision, record/source confounding and pairing controls must
be evaluated alongside COSD. Real-data Go/No-Go criteria remain untested.
