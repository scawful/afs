"""Rebalance training datasets by source or domain."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class RebalanceResult:
    """Summary of a rebalance run."""

    total_input: int = 0
    total_output: int = 0
    target_total: int = 0
    group_counts_in: dict[str, int] = field(default_factory=dict)
    group_counts_out: dict[str, int] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"input={self.total_input} output={self.total_output} "
            f"target={self.target_total} groups={len(self.group_counts_out)} "
            f"errors={len(self.errors)}"
        )


def rebalance_dataset(
    input_paths: Iterable[Path],
    output_path: Path,
    *,
    group_by: str = "source",
    weights: dict[str, float] | None = None,
    max_total: int | None = None,
    allow_oversample: bool = False,
    include_unweighted: bool = False,
    seed: int = 42,
    shuffle: bool = True,
    append_paths: Iterable[Path] | None = None,
) -> RebalanceResult:
    """Rebalance a dataset by source or domain."""
    result = RebalanceResult()
    samples = _load_samples(input_paths, result)
    result.total_input = len(samples)

    groups = _group_samples(samples, group_by)
    result.group_counts_in = {name: len(items) for name, items in groups.items()}

    selected: list[dict] = []
    rng = random.Random(seed)

    if weights:
        normalized = _normalize_weights(weights, result)
        if not normalized:
            return _write_output(output_path, selected, result, shuffle=shuffle, rng=rng)

        weighted_groups = {
            name: groups.get(name, [])
            for name in normalized.keys()
        }

        target_total = max_total
        if target_total is None:
            target_total = _max_target_without_oversample(
                weighted_groups, normalized
            )
        result.target_total = max(target_total or 0, 0)

        targets = _allocate_targets(
            weighted_groups,
            normalized,
            result.target_total,
            allow_oversample=allow_oversample,
        )

        for name, target in targets.items():
            pool = list(weighted_groups.get(name, []))
            if not pool:
                result.errors.append(f"No samples for group '{name}'")
                continue
            if allow_oversample and target > len(pool):
                chosen = [rng.choice(pool) for _ in range(target)]
            else:
                rng.shuffle(pool)
                chosen = pool[: min(target, len(pool))]
            selected.extend(chosen)

        if include_unweighted:
            for name, pool in groups.items():
                if name in normalized:
                    continue
                selected.extend(pool)
    else:
        selected = list(samples)

    if append_paths:
        appended = _load_samples(append_paths, result)
        selected.extend(appended)

    result.group_counts_out = _count_groups(selected, group_by)
    return _write_output(output_path, selected, result, shuffle=shuffle, rng=rng)


def _load_samples(paths: Iterable[Path], result: RebalanceResult) -> list[dict]:
    samples: list[dict] = []
    for path in paths:
        path = Path(path).expanduser()
        if not path.exists():
            result.errors.append(f"Missing input: {path}")
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        result.errors.append(f"Invalid JSON line in {path}")
        except OSError as exc:
            result.errors.append(f"{path}: {exc}")
    return samples


def _group_samples(samples: list[dict], group_by: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for sample in samples:
        name = _group_value(sample, group_by)
        groups.setdefault(name, []).append(sample)
    return groups


def _count_groups(samples: list[dict], group_by: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        name = _group_value(sample, group_by)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _group_value(sample: dict, group_by: str) -> str:
    value = sample.get(group_by)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown"


def _normalize_weights(
    weights: dict[str, float],
    result: RebalanceResult,
) -> dict[str, float]:
    filtered: dict[str, float] = {}
    for name, value in weights.items():
        try:
            weight = float(value)
        except (TypeError, ValueError):
            result.errors.append(f"Invalid weight for '{name}'")
            continue
        if weight <= 0:
            result.errors.append(f"Weight for '{name}' must be > 0")
            continue
        filtered[name] = weight
    total = sum(filtered.values())
    if total <= 0:
        result.errors.append("No valid weights provided")
        return {}
    normalized = {name: weight / total for name, weight in filtered.items()}
    result.weights = dict(normalized)
    return normalized


def _max_target_without_oversample(
    groups: dict[str, list[dict]],
    weights: dict[str, float],
) -> int:
    targets = []
    for name, weight in weights.items():
        if weight <= 0:
            continue
        count = len(groups.get(name, []))
        if count == 0:
            continue
        targets.append(int(count / weight))
    return min(targets) if targets else 0


def _allocate_targets(
    groups: dict[str, list[dict]],
    weights: dict[str, float],
    target_total: int,
    *,
    allow_oversample: bool,
) -> dict[str, int]:
    targets: dict[str, int] = {}
    allocated = 0
    for name, weight in weights.items():
        count = int(weight * target_total)
        targets[name] = count
        allocated += count

    remainder = max(target_total - allocated, 0)
    if remainder == 0:
        return targets

    candidates: list[str] = list(weights.keys())
    while remainder > 0 and candidates:
        progressed = False
        for name in list(candidates):
            available = len(groups.get(name, []))
            if allow_oversample or targets[name] < available:
                targets[name] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
            else:
                candidates.remove(name)
        if not progressed:
            break
    return targets


def _write_output(
    output_path: Path,
    samples: list[dict],
    result: RebalanceResult,
    *,
    shuffle: bool,
    rng: random.Random,
) -> RebalanceResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shuffle:
        rng.shuffle(samples)
    result.total_output = len(samples)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return result
