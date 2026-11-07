"""Seed-level summaries and paired patient-cluster comparisons of completed runs."""

import json
import statistics

from .evaluation import compare_predictions
from .manifest_tools import relative_path
from .utils import file_sha256


def summarize_evaluations(runner):
    groups, evaluations = {}, {}
    for name, run in runner.state["variants"].items():
        evaluations[name] = {}
        for relative in run["evaluations"]:
            path = runner.input(relative)
            report = json.loads(path.read_text(encoding="utf-8"))
            setting = path.relative_to(runner.output(name + "/evaluation")).as_posix()
            evaluations[name][setting] = path
            for metric, value in report["test"].items():
                if isinstance(value, (int, float)):
                    key = (run["variant"], setting, metric)
                    groups.setdefault(key, []).append((run["seed"], float(value)))
    aggregates = []
    for (variant, setting, metric), values in sorted(groups.items()):
        scores = [v for _, v in values]
        aggregates.append({"variant": variant, "evaluation": setting, "metric": metric,
            "seeds": [s for s, _ in values], "values": scores, "mean": statistics.mean(scores),
            "sample_std": statistics.stdev(scores) if len(scores) > 1 else None,
            "all_planned_seeds": len(scores) == len(runner.plan["seeds"])})
    pairs = [("cosd", other) for other in ("ssl", "pw_kd", "random_rank", "pca_rank", "physio")]
    pairs.append(("cosd_pw", "pw_kd"))
    counterparts = {}
    for variant in runner.plan["variants"]:
        if variant.get("paired_counterpart"):
            counterparts.setdefault(variant["paired_counterpart"], []).append(variant["name"])
    pairs.extend(tuple(sorted(names)) for names in counterparts.values() if len(names) == 2)
    comparisons = []
    for seed in runner.plan["seeds"]:
        for first, second in pairs:
            a = evaluations.get(f"seed_{seed}/{first}", {})
            b = evaluations.get(f"seed_{seed}/{second}", {})
            for setting in sorted(a.keys() & b.keys()):
                pa, pb = (p.with_name("predictions.npz") for p in (a[setting], b[setting]))
                name = f"comparisons/seed_{seed}/{first}_vs_{second}/{setting}"
                output = runner.output(name)
                runner.fresh_job(name, {"first": file_sha256(pa), "second": file_sha256(pb), "seed": seed}, [output],
                    lambda: compare_predictions(pa, pb, output, runner.plan["evaluation"]["bootstrap_samples"], seed))
                comparisons.append({"seed": seed, "first": first, "second": second,
                                    "report": relative_path(output, runner.root)})
    return {"seed_aggregates": aggregates, "paired_downstream": comparisons,
            "statistical_scope": "seed variability and per-seed patient-cluster intervals are separate; no pooled-seed CI or automatic Go claim"}
