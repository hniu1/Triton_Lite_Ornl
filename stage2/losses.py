"""Losses for direct multi-horizon hydraulic forecasting."""

import torch
import torch.nn.functional as F


def masked_huber(prediction, target, weights, delta):
    raw = F.huber_loss(prediction, target, reduction="none", delta=delta)
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def depth_bin_weights(target, args):
    weights = torch.full_like(target, args.depth_weight_shallow)
    weights = torch.where(
        target >= args.depth_moderate_threshold,
        torch.as_tensor(args.depth_weight_moderate, device=target.device),
        weights,
    )
    weights = torch.where(
        target >= args.depth_deep_threshold,
        torch.as_tensor(args.depth_weight_deep, device=target.device),
        weights,
    )
    return torch.where(
        target >= args.depth_extreme_threshold,
        torch.as_tensor(args.depth_weight_extreme, device=target.device),
        weights,
    )


def soft_dice_loss(logits, target, mask, smoothing=1.0):
    probability = torch.sigmoid(logits) * mask
    target = target * mask
    intersection = (probability * target).sum()
    denominator = probability.sum() + target.sum()
    return 1.0 - (2.0 * intersection + smoothing) / (denominator + smoothing)


def classify_transition_regimes(
    previous_depth,
    target_depth,
    mask,
    wet_threshold,
    stable_depth_threshold=0.01,
    stable_extent_threshold=0.01,
    rapid_depth_threshold=0.10,
    rapid_extent_threshold=0.05,
):
    """Classify total origin-to-target change for each forecast patch."""

    previous_wet = (previous_depth >= wet_threshold).float() * mask
    target_wet = (target_depth >= wet_threshold).float() * mask
    active = torch.maximum(previous_wet, target_wet)
    delta = target_depth - previous_depth
    active_count = active.sum(dim=(-2, -1)).clamp_min(1.0)
    valid_count = mask.sum(dim=(-2, -1)).clamp_min(1.0)
    signed_depth_change = (delta * active).sum(dim=(-2, -1)) / active_count
    mean_abs_depth_change = (delta.abs() * active).sum(dim=(-2, -1)) / active_count
    extent_change = (target_wet - previous_wet).sum(dim=(-2, -1)) / valid_count
    stable = (
        (mean_abs_depth_change < stable_depth_threshold)
        & (extent_change.abs() < stable_extent_threshold)
    )
    direction = signed_depth_change + extent_change
    rapid = (
        (mean_abs_depth_change >= rapid_depth_threshold)
        | (extent_change.abs() >= rapid_extent_threshold)
    )
    return {
        "all": torch.ones_like(stable, dtype=torch.bool),
        "stable": stable,
        "filling": (~stable) & (direction >= 0),
        "draining": (~stable) & (direction < 0),
        "rapid": rapid,
    }


def transition_loss_terms(output, batch, args, device):
    depth, wet_logits, component_x, component_y, depth_delta = output[:5]
    mask = batch["mask"]
    wet = (batch["depth"] >= args.wet_threshold).float() * mask
    previous_wet = (batch["previous_depth"] >= args.wet_threshold).float() * mask
    dry = mask * (1.0 - wet)
    weights = wet * depth_bin_weights(batch["depth"], args)
    depth_log = masked_huber(
        torch.log1p(depth), torch.log1p(batch["depth"]), weights, 0.20
    )
    depth_physical = masked_huber(depth, batch["depth"], weights, 1.0)
    depth_loss = (
        args.depth_log_loss_weight * depth_log
        + args.depth_physical_loss_weight * depth_physical
    )
    true_depth_delta = batch["depth"] - batch["previous_depth"]
    transition_active = torch.maximum(wet, previous_wet)
    transition_active = torch.maximum(
        transition_active, (true_depth_delta.abs() >= 0.01).float() * mask
    )
    transition_depth = masked_huber(
        depth_delta, true_depth_delta, transition_active, 0.10
    )
    dry_depth = (depth.square() * dry).sum() / dry.sum().clamp_min(1.0)
    wet_bce_raw = F.binary_cross_entropy_with_logits(
        wet_logits,
        wet,
        reduction="none",
        pos_weight=torch.tensor(args.wet_pos_weight, device=device),
    )
    wet_bce = (wet_bce_raw * mask).sum() / mask.sum().clamp_min(1.0)
    wet_dice = soft_dice_loss(wet_logits, wet, mask)
    component = 0.5 * (
        masked_huber(component_x, batch["component_x"], wet, 0.25)
        + masked_huber(component_y, batch["component_y"], wet, 0.25)
    )
    dry_component = (
        ((component_x.square() + component_y.square()) * dry).sum()
        / dry.sum().clamp_min(1.0)
    )
    loss = (
        depth_loss
        + args.transition_depth_loss_weight * transition_depth
        + args.dry_depth_loss_weight * dry_depth
        + args.wet_loss_weight * wet_bce
        + args.wet_dice_loss_weight * wet_dice
        + args.component_loss_weight * component
        + args.dry_component_loss_weight * dry_component
    )
    values = {
        "loss": loss,
        "depth": depth_loss,
        "depth_log": depth_log,
        "depth_physical": depth_physical,
        "transition_depth": transition_depth,
        "dry_depth": dry_depth,
        "wet_bce": wet_bce,
        "wet_dice": wet_dice,
        "component": component,
        "dry_component": dry_component,
    }
    nonfinite = [name for name, value in values.items() if not torch.isfinite(value)]
    if nonfinite:
        raise FloatingPointError(f"Non-finite transition losses: {nonfinite}")
    return values


def weighted_masked_huber(prediction, target, mask, sample_selection, delta):
    selection_shape = [prediction.shape[0]] + [1] * (prediction.ndim - 1)
    weights = mask * sample_selection.bool().reshape(selection_shape).to(mask.dtype)
    denominator = weights.sum()
    if float(denominator.detach()) <= 0:
        return prediction.sum() * 0.0
    raw = F.huber_loss(prediction, target, reduction="none", delta=delta)
    return (raw * weights).sum() / denominator


def weighted_masked_mse(prediction, target, mask, sample_selection):
    selection_shape = [prediction.shape[0]] + [1] * (prediction.ndim - 1)
    weights = mask * sample_selection.bool().reshape(selection_shape).to(mask.dtype)
    denominator = weights.sum()
    if float(denominator.detach()) <= 0:
        return prediction.sum() * 0.0
    return (((prediction - target) ** 2) * weights).sum() / denominator


def delta_aware_loss_terms(output, batch, args, device):
    """Add total-change, velocity, storage, and activity objectives."""

    values = transition_loss_terms(output, batch, args, device)
    depth, _, component_x, component_y, depth_delta, component_delta = output[:6]
    true_previous_depth = batch["true_previous_depth"]
    regimes = classify_transition_regimes(
        true_previous_depth,
        batch["depth"],
        batch["mask"],
        args.wet_threshold,
        args.stable_depth_threshold,
        args.stable_extent_threshold,
        args.rapid_depth_threshold,
        args.rapid_extent_threshold,
    )
    true_depth_delta = batch["depth"] - batch["previous_depth"]
    target_wet = (batch["depth"] >= args.wet_threshold).float() * batch["mask"]
    previous_wet = (true_previous_depth >= args.wet_threshold).float() * batch["mask"]
    active = torch.maximum(target_wet, previous_wet)
    active = torch.maximum(
        active, (true_depth_delta.abs() >= 0.01).float() * batch["mask"]
    )
    rapid_depth_delta = weighted_masked_huber(
        depth_delta, true_depth_delta, active, regimes["rapid"], 0.10
    )
    stable_depth_delta = weighted_masked_huber(
        depth_delta, true_depth_delta, batch["mask"], regimes["stable"], 0.05
    )
    true_component_delta = torch.stack(
        [
            batch["component_x"] - batch["previous_component_x"],
            batch["component_y"] - batch["previous_component_y"],
        ],
        dim=1,
    )
    component_mask = active.unsqueeze(1).expand_as(component_delta)
    component_delta_loss = weighted_masked_huber(
        component_delta, true_component_delta, component_mask, regimes["all"], 0.10
    )
    predicted_denominator = depth.clamp_min(args.wet_threshold)
    target_denominator = batch["depth"].clamp_min(args.wet_threshold)
    predicted_velocity = torch.stack(
        [component_x / predicted_denominator, component_y / predicted_denominator], dim=1
    )
    target_velocity = torch.stack(
        [
            batch["component_x"] / target_denominator,
            batch["component_y"] / target_denominator,
        ],
        dim=1,
    )
    velocity_mask = target_wet.unsqueeze(1).expand_as(predicted_velocity)
    if getattr(args, "derived_velocity_loss_type", "huber") == "mse":
        derived_velocity = weighted_masked_mse(
            predicted_velocity, target_velocity, velocity_mask, regimes["all"]
        )
    else:
        derived_velocity = weighted_masked_huber(
            predicted_velocity,
            target_velocity,
            velocity_mask,
            regimes["all"],
            getattr(args, "derived_velocity_huber_delta", 0.25),
        )
    valid_count = batch["mask"].sum(dim=(-2, -1)).clamp_min(1.0)
    predicted_storage_change = (
        ((depth - batch["previous_depth"]) * batch["mask"]).sum(dim=(-2, -1))
        / valid_count
    )
    target_storage_change = (
        (true_depth_delta * batch["mask"]).sum(dim=(-2, -1)) / valid_count
    )
    storage_weights = torch.where(
        regimes["rapid"],
        torch.full_like(predicted_storage_change, 2.0),
        torch.ones_like(predicted_storage_change),
    )
    storage_change = (
        F.huber_loss(
            predicted_storage_change,
            target_storage_change,
            reduction="none",
            delta=0.05,
        )
        * storage_weights
    ).sum() / storage_weights.sum().clamp_min(1.0)
    activity_gate = depth.sum() * 0.0
    if len(output) > 6:
        activity_logits = output[6]
        patch_dynamic = (~regimes["stable"]).to(batch["mask"].dtype)[:, None, None]
        changed_extent = (target_wet != previous_wet).to(batch["mask"].dtype)
        changed_depth = (true_depth_delta.abs() >= 0.01).to(batch["mask"].dtype)
        activity_target = patch_dynamic * torch.maximum(
            torch.maximum(target_wet, previous_wet),
            torch.maximum(changed_extent, changed_depth),
        )
        activity_target = activity_target * batch["mask"]
        positive = activity_target.sum().clamp_min(1.0)
        valid = batch["mask"].sum().clamp_min(1.0)
        positive_weight = ((valid - positive).clamp_min(1.0) / positive).clamp(1.0, 20.0)
        raw = F.binary_cross_entropy_with_logits(
            activity_logits,
            activity_target,
            reduction="none",
            pos_weight=positive_weight,
        )
        activity_gate = (raw * batch["mask"]).sum() / valid
    values["loss"] = (
        values["loss"]
        + args.rapid_depth_delta_loss_weight * rapid_depth_delta
        + args.stable_depth_delta_loss_weight * stable_depth_delta
        + args.component_delta_loss_weight * component_delta_loss
        + args.derived_velocity_loss_weight * derived_velocity
        + args.storage_change_loss_weight * storage_change
        + getattr(args, "activity_gate_loss_weight", 0.0) * activity_gate
    )
    values.update(
        {
            "rapid_depth_delta": rapid_depth_delta,
            "stable_depth_delta": stable_depth_delta,
            "component_delta": component_delta_loss,
            "derived_velocity": derived_velocity,
            "storage_change": storage_change,
            "activity_gate": activity_gate,
        }
    )
    nonfinite = [name for name, value in values.items() if not torch.isfinite(value)]
    if nonfinite:
        raise FloatingPointError(f"Non-finite direct-horizon losses: {nonfinite}")
    return values

