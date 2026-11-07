# Compatibility and migration

## Encoder weights

The `tempobridge` Python import and `TempoBridge` encoder class remain stable.
COSD changes training targets, not the tokenizer or Transformer parameter layout.
Existing PPG-only exports keep their original pooling, dimensions and crop/full-window protocol.
The new transfer feature API always uses the final context patch mean independently of export pooling.

## Calibration artifacts

- Schema 1: rejected for new training because calibration used a different feature readout.
  Regenerate features and calibration from the intended warm-up and teacher checkpoints.
- Schema 2: supported for full-target KD with the matching feature specification and fingerprints.
- Schema 3: supports portable feature-cache provenance, common target scaling and COSD transforms.
  Configuration changes affecting calibration/selection require a new artifact.

Schema 3 fits final target scaling on basis patients to match full and subspace comparisons.
This is an explicit numerical change from historical all-calibration-patient scaling.
Old and new KD-loss numbers should not be compared as efficacy evidence.

## Data identity

Patient/group IDs still determine the existing split protocol. HDF5 recording IDs are
now read from `record_key`/`record_id`; missing record IDs remain unknown rather than
being filled with patient IDs. Unknown time is represented by -1 seconds, not a fabricated zero.
Quality values retain their numerical fallback for old shards, but their source is explicitly
`unknown` where no measurement is supplied. A fallback value of 1 is not a quality guarantee.
CSV source/time/case/quality-provenance metadata now participates in artifact identity.

## Experiments

Legacy `transfer_mode` remains the weighting axis. `transfer_method` selects the target workflow:
`none`, `vanilla`, `pw_kd`, or `cosd`. Vanilla requires weighting `all` or `quality` and can
generate target statistics automatically without OOF probes. COSD requires a schema-3 artifact.
An empty selected subspace disables KD while retaining PPG self-supervision.

`checkpoint_selection=reconstruction` preserves the old selection behavior;
`last` makes `best.pt` the fixed-budget endpoint. Both always save `last.pt`.
Resume checks bind optimizer/head state to the exact run configuration and calibration.

## Portable use

Use `python -m tempobridge` in the active environment. Paths are supplied through CLI/config
or resolved relative to manifests; generated caches are portable directories. Virtual environments
are local installations, not distribution artifacts, and should be recreated on another machine.
