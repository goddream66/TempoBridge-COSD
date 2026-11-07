"""Frozen teacher targets and sample weights, separate from the training loop."""

import torch
from torch.nn import functional as F


def distillation_loss(model, heads, teacher, calibrator, ppg, ecg, valid,
                      ppg_quality, ecg_quality, scale_index, mode, protect_shallow):
    rank = calibrator.output_dims[scale_index]
    weights = calibrator.lookup(scale_index, ppg_quality, ecg_quality, valid, mode)
    if rank == 0 or not (calibrator.direction_weights(scale_index) > 0).any():
        weights = torch.zeros_like(weights)
    selected = weights > 0
    metrics = {"mean_transfer_weight": float(weights.mean()),
               "paired_fraction": float(valid.float().mean()),
               "active_transfer_fraction": float(selected.float().mean()),
               "target_rank": float(rank)}
    if not selected.any():
        # No absent-supervision updates or weight decay on transfer-only parameters.
        return ppg.new_zeros(()), metrics
    with torch.no_grad():
        target = calibrator.normalize(teacher(ecg[selected]), scale_index)
    features = model.transfer_features(ppg[selected], detach_shallow=protect_shallow)
    prediction = heads(features, scale_index).float()
    if prediction.shape != target.shape:
        raise ValueError("Prediction/teacher target shape mismatch")
    errors = F.smooth_l1_loss(prediction, target.float(), reduction="none")
    # Soft selection is a loss weight, not a second attenuation of target amplitudes.
    errors = (errors * calibrator.direction_weights(scale_index)).mean(-1)
    return (weights[selected] * errors).sum() / len(ppg), metrics


def gradient_audit(ssl, kd, model, transfer_weight):
    """Occasional diagnostic of actual encoder gradients; does not modify .grad."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    a = torch.autograd.grad(ssl, parameters, retain_graph=True, allow_unused=True)
    b = (torch.autograd.grad(kd * transfer_weight, parameters, retain_graph=True, allow_unused=True)
         if kd.requires_grad else [None] * len(parameters))
    norm_a = sum(float(g.detach().float().square().sum()) for g in a if g is not None) ** .5
    norm_b = sum(float(g.detach().float().square().sum()) for g in b if g is not None) ** .5
    dot = sum(float((x.detach().float() * y.detach().float()).sum())
              for x, y in zip(a, b) if x is not None and y is not None)
    return {"ssl_gradient_norm": norm_a, "kd_gradient_norm": norm_b,
            "gradient_cosine": dot / max(norm_a * norm_b, 1e-12), "gradient_audit_fraction": 1.}
