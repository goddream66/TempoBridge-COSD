"""Predeclared ablations from the research specification, without efficacy claims."""

import copy


def merge_config(base, patch):
    result = copy.deepcopy(base)
    for section, values in patch.items():
        result.setdefault(section, {}).update(copy.deepcopy(values))
    return result


def variants_for(families, dimension, rank, sources, has_labels=False, depth=6, native_teacher=True):
    allowed = {"core", "backbone", "calibration", "subspace", "quality", "pairing", "data", "gradient", "sampling", "teacher"}
    families = allowed if "all" in families else set(families)
    if not families.issubset(allowed):
        raise ValueError(f"Unknown experiment families: {families - allowed}")
    result, unavailable = {}, []

    def add(name, patch=None, family="core", **options):
        defaults = {"train": {"transfer_method": "cosd", "transfer_mode": "quality"}}
        result[name] = {"name": name, "family": family,
                        "patch": merge_config(defaults, patch or {}), **options}

    # Dependency controls are always included so a selected family remains interpretable.
    add("ssl", {"train": {"transfer_method": "none", "transfer_mode": "none"}})
    add("vanilla", {"train": {"transfer_method": "vanilla", "transfer_mode": "all"}})
    add("quality", {"train": {"transfer_method": "vanilla", "transfer_mode": "quality"}})
    add("pw_kd", {"train": {"transfer_method": "pw_kd", "transfer_mode": "calibrated"}})
    add("cosd")
    add("cosd_pw", {"train": {"transfer_mode": "calibrated"}}, matched_sample_weights_to="pw_kd")
    add("random_rank", {"cosd": {"basis": "random", "selection": "all"}}, match_rank_to="cosd")
    add("pca_rank", {"cosd": {"basis": "pca", "selection": "variance"}}, match_rank_to="cosd")
    add("soft", {"cosd": {"soft": True}})
    add("soft_weight_matched", {"cosd": {"soft": True}, "train": {"normalize_direction_weights": True}},
        matched_direction_weight_mass_to="cosd")
    if has_labels:
        add("physio", {"train": {"transfer_method": "physio", "transfer_mode": "all"}})
    else:
        unavailable.append({"name": "physio", "reason": "independent reference-target CSV not supplied"})
    if "backbone" in families:
        for encoder in ("local", "multiview"):
            add(f"encoder_{encoder}", {"model": {"encoder": encoder}}, "backbone")
        for name, patch in {"mean_fusion": {"selective_fusion": "mean"},
                            "no_coverage": {"selective_coverage": False},
                            "no_morphology": {"morphology_branch": False}}.items():
            add(name, {"model": patch}, "backbone")
        for size in (10, 25, 50):
            add(f"patch_{size}", {"model": {"patch_size": size}}, "backbone")
    if "calibration" in families:
        for probe in ("ridge", "mlp", "consensus"):
            for bins in (False, True):
                add(f"probe_{probe}_{'quality' if bins else 'global'}", {
                    "train": {"transfer_method": "pw_kd", "transfer_mode": "calibrated"},
                    "calibration": {"probe": probe}, "data": {"quality_edges": [.33, .67] if bins else []}}, "calibration")
        add("scales_2_8_30", {"data": {"scales_seconds": [2., 8., 30.], "window_mode": "crop"}}, "calibration")
    if "subspace" in families:
        for basis in ("standardized", "pca", "random"):
            add(f"basis_{basis}", {"cosd": {"basis": basis}}, "subspace")
        add("random_coordinates", {"cosd": {"selection": "random"}}, "subspace", match_rank_to="cosd")
        add("without_stability", {"cosd": {"stability": False}}, "subspace")
        add("quality_stable_directions", {"cosd": {"quality_strata": True}}, "subspace")
        for r in sorted({r for r in (16, 32, 64, 128, 192, 256, rank, dimension) if 0 < r <= dimension}):
            add(f"rank_{r}", {"cosd": {"rank": r}}, "subspace")
        for threshold in (0., .1, .25):
            add(f"threshold_{str(threshold).replace('.', '_')}", {"cosd": {"min_skill": threshold}}, "subspace")
    if "quality" in families:
        for ppg, ecg, observable in ((False, False, False), (True, False, False), (False, True, False),
                                      (True, True, False), (False, False, True), (True, True, True)):
            add(f"quality_p{int(ppg)}e{int(ecg)}o{int(observable)}", {"train": {
                "transfer_method": "cosd" if observable else "vanilla", "transfer_mode": "quality",
                "ppg_quality_weight": ppg, "ecg_quality_weight": ecg}}, "quality")
    if "pairing" in families:
        for mode in ("within_record", "cross_patient", "cross_source", "time_shift"):
            if mode == "cross_source" and len(sources) < 2:
                unavailable.append({"name": mode, "reason": "requires at least two sources"})
                continue
            offsets = (-30., 30., 60.) if mode == "time_shift" else (30.,)
            for offset in offsets:
                suffix = f"_{int(offset)}".replace("-", "minus") if mode == "time_shift" else ""
                for apply in (False, True):
                    add(f"{mode}{suffix}_{'wrong' if apply else 'correct'}", {"train": {
                        "pairing_control": mode, "pairing_shift_seconds": offset, "pairing_apply": apply,
                        "pairing_weight_match": True}}, "pairing", paired_counterpart=f"{mode}{suffix}")
    if "data" in families:
        add("paired_only", {"train": {"sample_subset": "paired"}}, "data")
        add("unpaired_only", {"train": {"sample_subset": "unpaired", "transfer_method": "none", "transfer_mode": "none"}}, "data")
        if len(sources) >= 2:
            for source in sources:
                add(f"source_only_{source}", family="data", sources=[source], source_held_out=[s for s in sources if s != source])
    if "gradient" in families:
        add("protected", {"train": {"gradient_audit_interval": 10}}, "gradient")
        add("unprotected", {"train": {"protect_shallow": False, "gradient_audit_interval": 10}}, "gradient")
        add("reduced_ssl", {"train": {"ssl_weight": .25, "gradient_audit_interval": 10}}, "gradient")
    if "sampling" in families and len(sources) == 2:
        for first in (.8, .6, .5):
            add(f"source_mix_{int(first * 100)}", {"data": {"source_weights": {
                sources[0]: first, sources[1]: 1 - first}}}, "sampling")
    if "teacher" in families:
        for threshold in (0., .5, .8):
            add(f"ecg_threshold_{int(threshold * 100)}", {"calibration": {"min_ecg_quality": threshold}}, "teacher")
        if native_teacher:
            for layer in sorted({0, depth // 2, depth - 1}):
                add(f"teacher_layer_{layer}", family="teacher", teacher_target={"layer": layer})
            for pool in ("summary", "combined"):
                add(f"teacher_pool_{pool}", family="teacher", teacher_target={"pool": pool})
            add("teacher_l2", family="teacher", teacher_target={"normalization": "l2"})
            add("teacher_random_diagnostic", family="teacher", teacher_target={"random": True}, diagnostic_only=True)
        else:
            unavailable.append({"name": "native_teacher_readout_controls", "reason": "external adapter controls its own readout"})
    return list(result.values()), unavailable
