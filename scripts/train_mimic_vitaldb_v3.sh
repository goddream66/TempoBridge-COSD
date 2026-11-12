#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train_mimic_vitaldb_v3.sh --manifest FILE [options]
  bash scripts/train_mimic_vitaldb_v3.sh --sources FILE [options]
  bash scripts/train_mimic_vitaldb_v3.sh --mimic-raw DIR --vital-raw DIR [options]

Use exactly one data input mode. --sources takes the same JSON as index-h5,
with sources named mimic and vitaldb. Relative paths are relative to the repo.

Options:
  --references FILE       Native reference JSONL. Omit to run only PPG warm-up.
  --annotations FILE      Optional reviewed event JSONL for event-targets.
  --mimic-id-map FILE     Optional patient/case mapping for MIMIC conversion.
  --vital-id-map FILE     Optional patient/case mapping for VitalDB conversion.
  --run-dir DIR           Output directory (default runs/mimic_vitaldb_v3).
  --device cuda|cpu       Training and evaluation device (default cuda).
  --epochs N              Epochs per training stage (default 50).
  --samples-per-epoch N   Training windows per epoch (default 0, auto).
                          Explicit values must be multiples of 160.
  --help                  Show this help.

Activate a Python environment and run: python -m pip install -e '.[data]'
Set PYTHON to select a different interpreter. Repeat the same command to resume.
EOF
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

manifest='' sources='' mimic_raw='' vital_raw='' references='' annotations=''
mimic_id_map='' vital_id_map=''
run_dir='runs/mimic_vitaldb_v3' device='cuda' epochs='50' samples_per_epoch='0'
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --manifest|--sources|--mimic-raw|--vital-raw|--references|--annotations|\
        --mimic-id-map|--vital-id-map|--run-dir|--device|--epochs|--samples-per-epoch)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $1"
            option=${1#--}
            printf -v "${option//-/_}" '%s' "$2"
            shift 2
            ;;
        *) fail "Unknown argument: $1 (see --help)" ;;
    esac
done

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_dir"
python_bin=${PYTHON:-python}

if [[ -n "$manifest" ]]; then
    [[ -z "$sources$mimic_raw$vital_raw" ]] || fail 'Use only one data input mode'
    [[ -f "$manifest" ]] || fail "Manifest not found: $manifest"
elif [[ -n "$sources" ]]; then
    [[ -z "$mimic_raw$vital_raw" ]] || fail 'Use only one data input mode'
    [[ -f "$sources" ]] || fail "Sources JSON not found: $sources"
else
    [[ -n "$mimic_raw" && -n "$vital_raw" ]] || fail 'Provide --manifest, --sources, or both raw roots'
    [[ -d "$mimic_raw" && -d "$vital_raw" ]] || fail 'Both raw data directories must exist'
fi
[[ -z "$references" || -f "$references" ]] || fail "References not found: $references"
[[ -z "$annotations" || ( -n "$references" && -f "$annotations" ) ]] || fail 'Annotations require --references and an existing annotation file'
[[ -z "$mimic_id_map" || ( -n "$mimic_raw" && -f "$mimic_id_map" ) ]] || fail 'MIMIC ID map requires --mimic-raw and an existing map file'
[[ -z "$vital_id_map" || ( -n "$vital_raw" && -f "$vital_id_map" ) ]] || fail 'VitalDB ID map requires --vital-raw and an existing map file'

mkdir -p -- "$run_dir"
if command -v flock >/dev/null 2>&1; then
    exec 9>"$run_dir/.train.lock"
    flock -n 9 || fail "Another process is using $run_dir"
fi

"$python_bin" - "$run_dir" "$device" "$epochs" "$samples_per_epoch" \
    "$manifest" "$sources" "$mimic_raw" "$vital_raw" "$mimic_id_map" "$vital_id_map" <<'PY'
import json
import sys
from pathlib import Path

import torch

from tempobridge.config import ExperimentConfig
from tempobridge.utils import file_sha256, write_json

run, device, epochs, samples, *inputs = sys.argv[1:]
run = Path(run)
epochs, samples = int(epochs), int(samples)
if device not in {"cuda", "cpu"}:
    raise SystemExit("--device must be cuda or cpu")
if epochs < 1 or samples < 0 or samples % 160:
    raise SystemExit("Use positive --epochs and nonnegative --samples-per-epoch divisible by 160")
if device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in this Python environment")
precision = "fp32"
if device == "cuda":
    precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

def save_or_check(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise SystemExit(f"Existing run settings differ in {path}. Reuse the original arguments or choose a new --run-dir.")
    else:
        write_json(path, value)

keys = ("manifest", "sources", "mimic_raw", "vital_raw", "mimic_id_map", "vital_id_map")
identity = {}
for key, value in zip(keys, inputs):
    if value:
        path = Path(value).resolve()
        identity[key] = {"path": str(path), "sha256": file_sha256(path) if path.is_file() else None}
save_or_check(run / "data_inputs.json", identity)

for stage in ("beat", "dynamic"):
    cfg = ExperimentConfig.load(f"configs/v3_{stage}_30s.json")
    cfg.data.source_weights = {"mimic": 4, "vitaldb": 1}
    cfg.data.samples_per_epoch = samples
    cfg.train.device, cfg.train.precision, cfg.train.epochs = device, precision, epochs
    cfg.validate()
    save_or_check(run / f"{stage}.json", cfg.to_dict())
PY

run_cli() { "$python_bin" -m tempobridge "$@"; }

if [[ -n "$mimic_raw" ]]; then
    mimic_args=() vital_args=()
    if [[ -n "$mimic_id_map" ]]; then mimic_args+=(--id-map "$mimic_id_map"); fi
    if [[ -n "$vital_id_map" ]]; then vital_args+=(--id-map "$vital_id_map"); fi
    run_cli preprocess --kind mimic --input-root "$mimic_raw" --output "$run_dir/processed/mimic" \
        --target-fs 50 --window-seconds 30 --resume "${mimic_args[@]}"
    run_cli preprocess --kind vitaldb --input-root "$vital_raw" --output "$run_dir/processed/vitaldb" \
        --target-fs 50 --window-seconds 30 --resume "${vital_args[@]}"
    sources="$run_dir/data_sources.json"
    cat >"$sources" <<'JSON'
{"sources":{"mimic":{"root":"processed/mimic","pattern":"*.h5","fs":50},"vitaldb":{"root":"processed/vitaldb","pattern":"*.h5","fs":50}}}
JSON
fi

if [[ -z "$manifest" ]]; then
    manifest="$run_dir/manifest.json"
    if [[ ! -f "$manifest" ]]; then
        run_cli index-h5 --sources "$sources" --output "$manifest"
    fi
fi

beat_config="$run_dir/beat.json"
dynamic_config="$run_dir/dynamic.json"
run_cli audit-data --config "$beat_config" --manifest "$manifest" --max-windows 32

prepare_events() {
    local table=$1 manual=$2
    if [[ -f "$table" ]]; then
        "$python_bin" - "$table" "$references" "$manual" <<'PY'
import sys

import h5py

from tempobridge.utils import file_sha256

events, references, annotations = sys.argv[1:]
with h5py.File(events, "r") as table:
    expected = {"references_sha256": file_sha256(references),
                "annotations_sha256": file_sha256(annotations) if annotations else ""}
    if any(table.attrs.get(key) != value for key, value in expected.items()):
        raise SystemExit(f"Reference inputs differ from {events}. Choose a new --run-dir.")
PY
    else
        local annotation_args=()
        if [[ -n "$manual" ]]; then annotation_args+=(--annotations "$manual"); fi
        run_cli event-targets --config "$beat_config" --manifest "$manifest" \
            --references "$references" --output "$table" "${annotation_args[@]}"
    fi
}

if [[ -n "$references" ]]; then
    native_events="$run_dir/events.h5"
    events="$native_events"
    audit_report="$run_dir/reference_review.json"
    audit_args=()
    prepare_events "$native_events" ''
    if [[ -n "$annotations" ]]; then
        events="$run_dir/events_reviewed.h5"
        audit_report="$run_dir/reference_audit.json"
        prepare_events "$events" "$annotations"
        audit_args+=(--manual-targets "$events")
    fi
    if [[ ! -f "$audit_report" ]]; then
        run_cli audit-events --config "$beat_config" --manifest "$manifest" \
            --event-targets "$native_events" --output "$audit_report" "${audit_args[@]}"
    fi
fi

train_stage() {
    local output=$1
    shift
    local resume_args=()
    if [[ -f "$output/last.pt" ]]; then resume_args+=(--resume "$output/last.pt"); fi
    run_cli "$@" --output "$output" "${resume_args[@]}"
}

train_stage "$run_dir/warmup" pretrain --config "$beat_config" --manifest "$manifest" --modality ppg

if [[ -z "$references" ]]; then
    printf 'Warm-up checkpoint: %s/warmup/last.pt\nRerun with --references to continue to V3-A/B.\n' "$run_dir"
    exit 0
fi

train_stage "$run_dir/v3_a" train-events --config "$beat_config" --manifest "$manifest" \
    --event-targets "$events" --student "$run_dir/warmup/last.pt"
train_stage "$run_dir/v3_b" train-events --config "$dynamic_config" --manifest "$manifest" \
    --event-targets "$events" --student "$run_dir/v3_a/last.pt"

checkpoint="$run_dir/v3_b/last.pt"
report="$run_dir/v3_b/events_test.json"
encoder="$run_dir/v3_b/encoder.pt"
if [[ ! -f "$report" ]]; then
    run_cli evaluate-events --checkpoint "$checkpoint" --manifest "$manifest" \
        --event-targets "$events" --split test --device "$device" --output "$report"
fi
if [[ ! -f "$encoder" ]]; then
    run_cli export --checkpoint "$checkpoint" --output "$encoder"
fi
printf 'Encoder: %s\nTest report: %s\n' "$encoder" "$report"
