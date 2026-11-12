"""Pulse event heads and V3-A/B training losses over the existing encoder."""

import itertools

import torch
from scipy.signal import find_peaks
from torch import nn
from torch.nn import functional as F

from .event_targets import match_event_times
from .model import random_mask, reconstruction_loss
from .transfer import parameter_group

METRIC_SCALES = {"hr_bpm": 10., "rmssd_ms": 20.}


class PhysiologicalEventHead(nn.Module):
    def __init__(self, dim, hidden, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_encoder = nn.Sequential(nn.Conv1d(dim, hidden, 3, padding=1), nn.GELU())
        self.refine = nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden), nn.GELU())
        self.event_logits = nn.Conv1d(hidden, 3, 1)
        self.beat_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.beat_change = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU())
        self.rr = nn.Linear(hidden, 2)
        with torch.no_grad():
            self.event_logits.bias[:2].fill_(-3.)
            self.rr.bias[0] = -1.1

    def events(self, tokens):
        h = self.temporal_encoder(tokens.transpose(1, 2))
        h = F.interpolate(h, scale_factor=self.patch_size, mode="linear", align_corners=False)
        foot, peak, quality = self.event_logits(h + self.refine(h)).float().unbind(1)
        return {"foot_logits": foot, "peak_logits": peak, "ppg_quality_proxy_logits": quality,
                "ppg_quality_proxy": quality.sigmoid()}

    def intervals(self, tokens, foot_logits, feet_s, beat_valid, fs):
        positions = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.float32)
        centers = (positions + .5) * self.patch_size / fs - .5 / fs
        # Reference/predicted feet define only a local neighborhood. They are never concatenated
        # as timing features. In particular, the RR head cannot subtract supplied timestamps.
        logits = F.logsigmoid(foot_logits).reshape(len(tokens), -1, self.patch_size).amax(-1)
        distance = feet_s[..., None] - centers
        support = distance.abs() <= max(.35, self.patch_size / fs)
        weights = (logits[:, None] - .5 * (distance / .16).square()).masked_fill(~support, -1e4)
        weights = weights.softmax(-1) * beat_valid[..., None]
        beats = self.beat_projection(weights.to(tokens.dtype) @ tokens) * beat_valid[..., None]
        first, second = beats[:, :-1], beats[:, 1:]
        dynamic = self.beat_change(torch.cat((first, second, second - first), -1))
        mean, logvar = self.rr(dynamic).float().unbind(-1)
        return {"beat_tokens": beats, "state_embedding": beats, "dynamic_embedding": dynamic,
                "rr_mean": .15 + 2.55 * mean.sigmoid(), "rr_logvar": -9. + 7. * logvar.sigmoid(),
                "beat_valid": beat_valid, "rr_valid": beat_valid[:, :-1] & beat_valid[:, 1:]}


class EventTransferHeads(nn.Module):
    def __init__(self, config, dynamic_dim=0):
        super().__init__()
        dim, hidden = config.model.dim, config.event.hidden_dim
        self.events = PhysiologicalEventHead(dim, hidden, config.model.patch_size)
        self.dynamic = (nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, hidden), nn.GELU(),
                                      nn.Linear(hidden, dynamic_dim)) if dynamic_dim else None)

    def encode_physiological(self, model, ppg):
        encoded = model.encode(ppg)
        return {"patch_tokens": encoded["context"], "context": encoded["context"],
                "summary": encoded["summary"], **self.events.events(encoded["context"])}

    def predict_change(self, first, second):
        return self.dynamic(torch.cat((first, second, second - first), -1)).float()


def _heatmap(times, valid, length, fs):
    grid = torch.arange(length, device=times.device, dtype=torch.float32) / fs
    if times.shape[1] == 0:
        return grid.new_zeros((len(times), length))
    distance = grid - times[..., None]
    heat = torch.exp(-.5 * (distance / max(.025, 1 / fs)).square())
    return (heat * valid[..., None]).amax(1)


def decode_event_times(logits, fs, threshold):
    probabilities = logits.detach().float().sigmoid().cpu().numpy()
    return [find_peaks(row, height=threshold, distance=max(1, round(.27 * fs)))[0] / fs for row in probabilities]


def pulse_event_losses(outputs, batch, device, fs):
    use = batch["event_labelled"].to(device)
    if not use.any():
        return {}
    valid = batch["beat_valid"].to(device)[use]
    losses = {}
    for name, column in (("foot", "feet_s"), ("peak", "peaks_s")):
        logits = outputs[f"{name}_logits"][use]
        heat = _heatmap(batch[column].to(device)[use], valid, logits.shape[-1], fs)
        # Balance sparse positives and background per labelled window.
        positive = (heat * F.softplus(-logits)).sum(-1) / heat.sum(-1).clamp_min(1.)
        negative = ((1 - heat) * F.softplus(logits)).sum(-1) / (1 - heat).sum(-1).clamp_min(1.)
        losses[name] = .5 * (positive + negative).sum() / len(use)
        if name == "foot":
            quality = batch["ppg_quality"].to(device, dtype=torch.float32)[use, None]
            quality_error = F.binary_cross_entropy_with_logits(outputs["ppg_quality_proxy_logits"][use],
                                                               quality.expand_as(heat), reduction="none")
            losses["quality_proxy"] = ((heat * quality_error).sum(-1) / heat.sum(-1).clamp_min(1.)).sum() / len(use)
    return losses


def select_rr_anchors(foot_logits, feet, beat_valid, valid, fs, config, epoch):
    if config.rr_anchor_mode == "reference":
        return feet, beat_valid, valid, 0
    active = valid.any(-1)
    if config.rr_anchor_mode == "mixed":
        probability = min(1., epoch / config.rr_anchor_ramp_epochs)
        active = active & (torch.rand(len(feet), device=feet.device) < probability)
    indices = active.nonzero(as_tuple=True)[0].tolist()
    if not indices:
        return feet, beat_valid, valid, 0
    anchors, keep, intervals = feet.clone(), beat_valid.clone(), valid.clone()
    decoded = decode_event_times(foot_logits[indices], fs, config.rr_anchor_threshold)
    for i, predicted in zip(indices, decoded):
        reference = feet[i, beat_valid[i]].cpu().numpy()
        pairs = match_event_times(predicted, reference, config.rr_anchor_tolerance_s)
        keep[i] = False
        intervals[i] = False
        for predicted_idx, reference_idx in pairs:
            anchors[i, reference_idx] = float(predicted[predicted_idx])
            keep[i, reference_idx] = True
        for (p, r), (next_p, next_r) in zip(pairs, pairs[1:]):
            if next_p == p + 1 and next_r == r + 1:
                intervals[i, r] = valid[i, r]
    return anchors, keep, intervals, len(indices)


def rr_distribution_loss(head, tokens, outputs, batch, device, fs, config, epoch):
    weights = batch["rr_weight"].to(device, dtype=torch.float32)
    valid = batch["rr_valid"].to(device) & (weights[:, None] > 0) & (weights[:, None] >= config.minimum_quality)
    eligible = int(valid.sum())
    feet, beats, valid, using_predicted = select_rr_anchors(outputs["foot_logits"], batch["feet_s"].to(device),
                                                         batch["beat_valid"].to(device), valid, fs, config, epoch)
    count = int(valid.sum())
    if not count:
        return None, count, eligible, using_predicted
    predicted = head.intervals(tokens, outputs["foot_logits"], feet, beats, fs)
    target = batch["rr_s"].to(device)
    mean, logvar = predicted["rr_mean"], predicted["rr_logvar"]
    if config.rr_loss == "gaussian":
        error = .5 * ((target - mean).square() * (-logvar).exp() + logvar)
    else:
        error = F.smooth_l1_loss(mean, target, beta=.1, reduction="none")
    per_window = (error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    return (per_window * weights).sum() / len(weights), count, eligible, using_predicted


def within_subject_dynamic_loss(model, heads, current, batch, teacher, device, config):
    weights = batch["pair_weight"].to(device, dtype=torch.float32)
    use = weights > 0
    if not use.any():
        return None
    selected = use.cpu()
    after = model.transfer_features(batch["next_ppg"][selected].to(device))
    prediction = heads.predict_change(current[use], after)
    if teacher is not None:
        with torch.no_grad():
            before_ecg = teacher(batch["ecg"][0][selected].to(device)).float()
            after_ecg = teacher(batch["next_ecg"][selected].to(device)).float()
        error = F.smooth_l1_loss(prediction, after_ecg - before_ecg, reduction="none").mean(-1)
    else:
        scales = prediction.new_tensor([METRIC_SCALES[name] for name in config.dynamic_metrics])
        target = batch["delta_metric"][selected].to(device) / scales
        valid = batch["delta_valid"][selected].to(device)
        element = F.smooth_l1_loss(prediction, target, reduction="none")
        error = (element * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    return (weights[use] * error).sum() / len(weights)


def objective_gradient_audit(losses, model, heads):
    named = list(model.named_parameters()) + [(f"head/{name.split('.')[0]}", p)
                                               for name, p in heads.named_parameters()]
    gradients = {key: torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
                 for key, loss in losses.items() if loss.requires_grad}
    results, norms = {}, {}
    for key, gradient in gradients.items():
        squared = {}
        for (name, _), value in zip(named, gradient):
            if value is not None:
                group = parameter_group(name)
                squared[group] = squared.get(group, 0.) + float(value.detach().float().square().sum())
        norms[key] = sum(squared.values()) ** .5
        results[f"v3_gradient_norm/{key}"] = norms[key]
        results.update({f"v3_gradient_norm/{key}/{group}": value ** .5 for group, value in squared.items()})
    scopes = {"": ("",), "backbone": ("stem.", "summary_token", "shallow.", "context.", "context_norm."),
              "stem": ("stem.",), "shallow": ("shallow.",), "context": ("context.", "context_norm."),
              "event_head": ("head/events",), "dynamic_head": ("head/dynamic",)}
    for scope, prefixes in scopes.items():
        indices = [i for i, (name, _) in enumerate(named) if name.startswith(prefixes)]
        by_loss = {key: [values[i] for i in indices] for key, values in gradients.items()}
        magnitudes = {key: sum(float(g.detach().float().square().sum()) for g in values if g is not None) ** .5
                      for key, values in by_loss.items()} if scope else norms
        for first, second in itertools.combinations(by_loss, 2):
            if magnitudes[first] and magnitudes[second]:
                product = sum(float((x.detach().float() * y.detach().float()).sum())
                              for x, y in zip(by_loss[first], by_loss[second]) if x is not None and y is not None)
                suffix = f"/{scope}" if scope else ""
                results[f"v3_gradient_cosine/{first}/{second}{suffix}"] = product / (magnitudes[first] * magnitudes[second])
    return results


def compute_v3_loss(config, model, heads, batch, teacher, device, audit_gradients, epoch=0):
    ppg = batch["ppg"][0].to(device)
    fs, settings = config.data.ppg_fs, config.event
    mask = random_mask(ppg, config.model.patch_size, config.train.mask_ratio)
    ssl = reconstruction_loss(model, ppg, mask, config.train.morphology_loss_weight)
    terms, counts = {}, {}
    wants_events = any((settings.foot_weight, settings.peak_weight, settings.quality_proxy_weight, settings.rr_weight))
    if wants_events:
        outputs = heads.encode_physiological(model, ppg)
        tokens = outputs["patch_tokens"]
        if any((settings.foot_weight, settings.peak_weight, settings.quality_proxy_weight)):
            terms.update(pulse_event_losses(outputs, batch, device, fs))
            counts["v3_event_fraction"] = float(batch["event_labelled"].float().mean())
        if settings.rr_weight:
            rr, count, eligible, predicted = rr_distribution_loss(heads.events, tokens, outputs, batch, device, fs,
                                                                  settings, epoch)
            counts["v3_rr_labels_per_window"] = count / len(ppg)
            counts["v3_rr_eligible_per_window"] = eligible / len(ppg)
            counts["v3_predicted_anchor_fraction"] = predicted / len(ppg)
            if rr is not None:
                terms["rr"] = rr
        current = tokens.mean(1)
    else:
        current = model.transfer_features(ppg)
    if settings.dynamic_weight:
        dynamic = within_subject_dynamic_loss(model, heads, current, batch, teacher, device, settings)
        counts["v3_dynamic_pair_fraction"] = float((batch["pair_weight"] > 0).float().mean())
        if settings.dynamic_target == "metrics":
            for i, name in enumerate(settings.dynamic_metrics):
                valid = batch["delta_valid"][:, i] & (batch["pair_weight"] > 0)
                counts[f"v3_dynamic_label_fraction/{name}"] = float(valid.float().mean())
        if dynamic is not None:
            terms["dynamic"] = dynamic
    weighted = {key: value * getattr(settings, f"{key}_weight") for key, value in terms.items()
                if getattr(settings, f"{key}_weight")}
    objectives = {"ssl": config.train.ssl_weight * ssl, **weighted}
    auxiliary = sum(weighted.values(), ssl.new_zeros(()))
    metrics = {"ssl": float(ssl.detach()), "transfer": float(auxiliary.detach()), **counts,
               **{f"v3_loss/{key}": float(value.detach()) for key, value in terms.items()}}
    if audit_gradients:
        metrics.update(objective_gradient_audit(objectives, model, heads))
        metrics["v3_gradient_audit_fraction"] = 1.
    return objectives["ssl"] + auxiliary, metrics
