"""Dataset-level pairing controls, with explicit matched-coverage baselines.

Time shifts use real neighboring record windows. No circular rolls, synthetic
padding, or fabricated timestamps are used to imitate physiological delays.
"""

import hashlib
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


def pairing_map(rows, mode, seed, shift_seconds=30., quality_edges=(), quality_match=False):
    if mode not in {"within_record", "time_shift", "cross_patient", "cross_source"}:
        raise ValueError("Unknown pairing control")
    rng = np.random.default_rng(seed)
    donor = np.full(len(rows), -1, dtype=np.int64)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if not row["subject_known"] or not row.get("ecg_available", False):
            continue
        quality_bin = int(np.digitize(row["ppg_quality"], quality_edges)) if quality_match else 0
        if mode in {"within_record", "time_shift"}:
            if not row["recording_known"]:
                continue
            key = (row["source"], row["subject_id"], row["recording_id"], quality_bin)
        else:
            key = (row["source"] if mode == "cross_patient" else "all_sources", quality_bin)
        groups[key].append(i)
    for indices in groups.values():
        if mode == "within_record":
            if len(indices) > 1:
                order = rng.permutation(indices)
                donor[order] = np.roll(order, int(rng.integers(1, len(order))))
        elif mode == "time_shift":
            by_time = defaultdict(list)
            for i in indices:
                if rows[i]["window_start_s"] >= 0:
                    by_time[round(rows[i]["window_start_s"], 6)].append(i)
            for i in indices:
                if rows[i]["window_start_s"] < 0:
                    continue
                candidates = by_time.get(round(rows[i]["window_start_s"] + shift_seconds, 6), [])
                if candidates:
                    donor[i] = rng.choice(candidates)
        else:
            # Group first, then sample a different patient/source, so long records do not
            # dominate donor identity. Donor windows may repeat; report that explicitly.
            field = "subject_id" if mode == "cross_patient" else "source"
            pools = defaultdict(list)
            for i in indices:
                pools[rows[i][field]].append(i)
            keys = sorted(pools)
            if len(keys) < 2:
                continue
            positions = {key: j for j, key in enumerate(keys)}
            for i in indices:
                own = positions[rows[i][field]]
                selected = int(rng.integers(len(keys) - 1))
                selected += selected >= own
                donor[i] = rng.choice(pools[keys[selected]])
    return donor


class PairingControlDataset(Dataset):
    def __init__(self, dataset, train_config, quality_edges):
        self.dataset, self.records = dataset, dataset.records
        self.settings, self.edges = train_config, quality_edges
        self.rows = [dataset.metadata(i) for i in range(len(dataset))]
        self.donors = None
        self.set_epoch(0)

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
        self.epoch = epoch
        c = self.settings
        self.donors = pairing_map(self.rows, c.pairing_control, c.seed + epoch,
                                  c.pairing_shift_seconds, self.edges, c.pairing_quality_match)
        if not (self.donors >= 0).any():
            raise ValueError("No eligible pairing-control donors; check record IDs, timestamps and sources")

    def metadata(self, index):
        return self.rows[index]

    def __getitem__(self, index):
        original = self.dataset[index]
        donor = int(self.donors[index])
        original["original_ecg_valid"] = original["ecg_valid"]
        if donor < 0:
            original.update(ecg_valid=False, ecg_quality=0.,
                            ecg=[torch.zeros_like(x) for x in original["ecg"]],
                            control_donor_id="", control_eligible=False)
            return original
        other = self.dataset[donor]
        valid = original["ecg_valid"] and other["ecg_valid"]
        common_quality = min(original["ecg_quality"], other["ecg_quality"])
        # The correct-pair counterpart has EXACTLY the same donor validity check.
        if self.settings.pairing_apply:
            original.update(ecg=other["ecg"], ecg_quality=other["ecg_quality"],
                            ecg_quality_source=other["ecg_quality_source"])
        original.update(ecg_valid=valid, control_donor_id=other["sample_id"], control_eligible=valid)
        if self.settings.pairing_weight_match:
            # Same reliability and PPG quality bin in correct/wrong counterparts.
            # Minimum uses both references conservatively; never upweights a poor donor.
            original.update(ecg_quality=common_quality,
                            ecg_quality_source="pair_control_min_original_donor")
        if not valid:
            original["ecg_quality"] = 0.
        return original

    def report(self):
        active = self.donors >= 0
        payload = [(r["sample_id"], int(d)) for r, d in zip(self.rows, self.donors)]
        return {"mode": self.settings.pairing_control, "applied": self.settings.pairing_apply,
                "matched_ecg_reliability": self.settings.pairing_weight_match,
                "epoch": self.epoch, "windows": len(self), "metadata_eligible": int(active.sum()),
                "unique_donors": len(set(self.donors[active].tolist())),
                "mapping_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
                "actual_validity": "checked at waveform load; see active_transfer_windows",
                "shift_seconds": self.settings.pairing_shift_seconds if self.settings.pairing_control == "time_shift" else None}

    def close(self):
        self.dataset.close()


class WindowSubsetDataset(Dataset):
    """Explicit paired/unpaired ablation using actual waveform validity.

    Indexing reads each training waveform once; metadata presence alone cannot
    certify that an ECG is usable. Original manifest identities remain unchanged.
    """
    def __init__(self, dataset, subset):
        if subset not in {"paired", "unpaired"}:
            raise ValueError("Subset must be paired or unpaired")
        self.dataset, self.config, self.subset = dataset, dataset.config, subset
        self.indices = [i for i in range(len(dataset)) if bool(dataset[i]["ecg_valid"]) == (subset == "paired")]
        if not self.indices:
            raise ValueError(f"No valid windows in {subset} subset")
        self.records = [dataset.records[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def metadata(self, index):
        return self.dataset.metadata(self.indices[index])

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def report(self):
        return {"subset": self.subset, "windows": len(self), "original_windows": len(self.dataset),
                "sample_ids": [r.sample_id for r in self.records], "selection": "actual ECG waveform validity"}

    def close(self):
        self.dataset.close()
