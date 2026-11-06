"""Compact shard-local sampling and exact no-replacement source mixtures."""

import bisect
import math
import random
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
