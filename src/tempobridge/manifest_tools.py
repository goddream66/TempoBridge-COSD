"""Source-only corpus views preserving original group assignments and data files."""

import csv
import json
import os
from pathlib import Path

from .config import DataConfig
from .data import read_manifest, source_ids
from .h5data import canonical_hash
from .utils import write_json


def relative_path(path, parent):
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(parent).resolve())).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def filter_sources(manifest, output, sources, data_config=None):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = data_config or DataConfig()
    records = read_manifest(manifest, require_train=False, data_config=config)
    available = set().union(*(source_ids(records, split) for split in ("train", "val", "test")))
    if not sources or not set(sources).issubset(available):
        raise ValueError("Requested sources must be a nonempty subset of the corpus")
    output.parent.mkdir(parents=True, exist_ok=True)
    if manifest.suffix.lower() == ".json":
        if output.suffix.lower() != ".json":
            raise ValueError("HDF5 source view must use a .json extension")
        payload = json.loads(json.dumps(records.payload))
        payload["sources"] = {k: v for k, v in payload["sources"].items() if k in sources}
        for source, spec in payload["sources"].items():
            root = manifest.parent / config.source_roots.get(source, spec["root"])
            spec["root"] = relative_path(root, output.parent)
        payload["manifest_sha256"] = canonical_hash(payload)
        write_json(output, payload)
    else:
        if output.suffix.lower() != ".csv":
            raise ValueError("NPZ source view must use a .csv extension")
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = [r for r in reader if (r.get("source") or "default") in sources]
        for row in rows:
            row["path"] = relative_path(manifest.parent / row["path"], output.parent)
        with output.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    read_manifest(output, require_train=False, data_config=DataConfig())
    return output
