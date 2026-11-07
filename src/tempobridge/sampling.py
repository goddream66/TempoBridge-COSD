"""Compact shard-local sampling and exact no-replacement source mixtures."""

import bisect
import math
import random
from collections import defaultdict
from fractions import Fraction

import numpy as np
from torch.utils.data import Sampler


def source_counts(lengths, weights, requested=0, update_size=1):
    if set(lengths) != set(weights) or any(n <= 0 for n in lengths.values()):
        raise ValueError("Source weights must name every nonempty training source exactly")
    values = {name: Fraction(str(weights[name])).limit_denominator(10000) for name in lengths}
    if any(w <= 0 for w in values.values()):
        raise ValueError("Source weights must be positive")
    total = sum(values.values())
    ratios = {name: w / total for name, w in values.items()}
    unit = math.lcm(update_size, *(w.denominator for w in ratios.values()))
    maximum = min(lengths[name] * w.denominator // w.numerator for name, w in ratios.items())
    count = requested or (maximum // unit * unit)
    if count < unit or count % unit or count > maximum:
        raise ValueError(f"Mixed epoch must be a multiple of {unit}, <= {maximum}, without oversampling")
    return {name: count * w.numerator // w.denominator for name, w in ratios.items()}


class SourceBlockSampler(Sampler):
    def __init__(self, records, config, batch_size, accumulation, seed, epoch=0):
        self.seed, self.epoch, self.block_size = seed, epoch, config.shuffle_block_size
        self.ranges = {}
        if hasattr(records, "segments"):
            for offset, begin, end, _, _, source, *_ in records.segments:
                self.ranges.setdefault(source, []).append((offset, end - begin))
        else:
            for i, record in enumerate(records):
                self.ranges.setdefault(record.source, []).append((i, 1))
        self.lengths = {k: sum(n for _, n in ranges) for k, ranges in self.ranges.items()}
        if config.source_weights:
            self.counts = source_counts(self.lengths, config.source_weights, config.samples_per_epoch,
                                        batch_size * accumulation)
        else:
            self.counts = self.lengths.copy()
            if config.samples_per_epoch:
                raise ValueError("samples_per_epoch requires explicit source_weights")

    def __len__(self):
        return sum(self.counts.values())

    def _source_stream(self, source, seed):
        rng = np.random.default_rng(seed)
        ranges = self.ranges[source]
        ends = np.cumsum([n for _, n in ranges])
        n = int(ends[-1])
        blocks = rng.permutation(math.ceil(n / self.block_size))
        for block in blocks:
            begin = int(block) * self.block_size
            for position in begin + rng.permutation(min(self.block_size, n - begin)):
                segment = bisect.bisect_right(ends, position)
                previous = int(ends[segment - 1]) if segment else 0
                yield int(ranges[segment][0] + position - previous)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        sources = sorted(self.counts)
        streams = {name: self._source_stream(name, self.seed + self.epoch + 104729 * i)
                   for i, name in enumerate(sources)}
        left = self.counts.copy()
        while sum(left.values()):
            probabilities = np.asarray([left[name] for name in sources], np.float64)
            source = sources[int(rng.choice(len(sources), p=probabilities / probabilities.sum()))]
            count = min(self.block_size, left[source])
            for _ in range(count):
                yield next(streams[source])
            left[source] -= count


def calibration_indices(records, per_subject, seed):
    """Sample at most K windows/group without expanding HDF5 rows or using held-out IDs."""
    groups = {}
    if hasattr(records, "segments"):
        for offset, begin, end, group, split, source, *_ in records.segments:
            if group is not None and split == "train":
                groups.setdefault(f"{source}:{group}", []).append((offset, end - begin))
    else:
        for i, record in enumerate(records):
            if record.subject_known and record.split == "train":
                groups.setdefault(record.subject_id, []).append((i, 1))
    selected, rng = [], random.Random(seed)
    for subject in sorted(groups):
        ranges = groups[subject]
        ends = np.cumsum([n for _, n in ranges])
        n = int(ends[-1])
        for position in sorted(rng.sample(range(n), min(per_subject, n))):
            segment = bisect.bisect_right(ends, position)
            previous = int(ends[segment - 1]) if segment else 0
            selected.append(int(ranges[segment][0] + position - previous))
    return selected


class PatientSourceSampler(Sampler):
    """Exact source quotas, balanced groups, then uniform records/windows.

Sampling is WITH replacement within records. Unknown identities form an explicit
pool per source and are never counted as independent patients.
"""
    def __init__(self, dataset, config, batch_size, accumulation, seed, epoch=0):
        self.seed, self.epoch = seed, epoch
        self.groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        lengths = defaultdict(int)
        for i in range(len(dataset)):
            row = dataset.metadata(i)
            source = row["source"]
            patient = row["subject_id"] if row["subject_known"] else "__unknown_pool__"
            record = row["recording_id"] if row["recording_known"] else "__unknown_record__"
            self.groups[source][patient][record].append(i)
            lengths[source] += 1
        weights = config.source_weights or dict(lengths)
        if set(weights) != set(lengths):
            raise ValueError("Patient sampler weights must name every source")
        ratios = {k: Fraction(str(v)).limit_denominator(10000) for k, v in weights.items()}
        total = sum(ratios.values())
        ratios = {k: v / total for k, v in ratios.items()}
        unit = math.lcm(batch_size * accumulation, *(r.denominator for r in ratios.values()))
        n = config.samples_per_epoch or sum(lengths.values()) // unit * unit
        if n < unit or n % unit:
            raise ValueError(f"Patient-sampled epoch must be a positive multiple of {unit}")
        self.counts = {k: n * r.numerator // r.denominator for k, r in ratios.items()}
        self.last_report = None

    def __len__(self):
        return sum(self.counts.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order, group_counts = [], {}
        for source in sorted(self.groups):
            patients = sorted(self.groups[source])
            patient_order = []
            while len(patient_order) < self.counts[source]:
                patient_order.extend(rng.permutation(patients).tolist())
            group_counts[source] = {p: 0 for p in patients}
            for patient in patient_order[:self.counts[source]]:
                records = self.groups[source][patient]
                record = rng.choice(sorted(records))
                order.append(int(rng.choice(records[record])))
                group_counts[source][patient] += 1
        rng.shuffle(order)
        self.last_report = {"kind": "source/group/record/window sampling with replacement",
                            "source_counts": self.counts, "group_counts": group_counts,
                            "unique_windows": len(set(order)), "sampled_windows": len(order),
                            "repeated_windows": len(order) - len(set(order)),
                            "omitted_windows": sum(len(v) for src in self.groups.values()
                                                   for rec in src.values() for v in rec.values()) - len(set(order))}
        yield from order
