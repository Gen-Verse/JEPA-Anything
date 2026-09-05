"""JEPA Anything objectives plus separate statistical monitoring utilities."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


Reduction = Literal["mean", "sum", "none"]


def _validate_eps(eps: float) -> None:
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def _validate_min_std(min_std: float) -> None:
    if not math.isfinite(min_std) or min_std < 0:
        raise ValueError("min_std must be finite and non-negative")


def _flatten_factor_samples(factors: Tensor) -> Tensor:
    if factors.ndim < 3:
        raise ValueError(f"factors must have shape (..., K, r), got {tuple(factors.shape)}")
    if not factors.is_floating_point():
        raise TypeError("factors must be floating point")
    if factors.shape[-2] <= 0 or factors.shape[-1] <= 0:
        raise ValueError("factor dimensions K and r must be positive")
    sample_count = math.prod(factors.shape[:-2])
    if sample_count == 0:
        raise ValueError("factors must contain at least one sample")
    return factors.reshape(sample_count, factors.shape[-2], factors.shape[-1])


def _accumulation_samples(factors: Tensor) -> Tensor:
    """Flatten factors and promote low-precision reductions to float32."""

    samples = _flatten_factor_samples(factors)
    if samples.dtype in (torch.float16, torch.bfloat16):
        return samples.float()
    return samples


def _coordinate_variance(factors: Tensor) -> Tensor:
    """Population coordinate variance (``correction=0``)."""

    samples = _accumulation_samples(factors)
    centered = samples - samples.mean(dim=0, keepdim=True)
    return centered.square().mean(dim=0)


def _reduce(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"unsupported reduction: {reduction!r}")


def factor_cross_correlation(factors: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Compute coordinate correlations between every pair of factors.

    Args:
        factors: Tensor with shape ``(..., K, r)``.  All leading dimensions are
            treated as sample dimensions.
        eps: Finite, positive variance stabilizer.

    Returns:
        Correlation blocks with shape ``(K, K, r, r)``.  Entry
        ``[i, j, a, b]`` correlates coordinate ``a`` of factor ``i`` with
        coordinate ``b`` of factor ``j``.
    """

    _validate_eps(eps)
    samples = _accumulation_samples(factors)
    centered = samples - samples.mean(dim=0, keepdim=True)
    covariance = torch.einsum("nka,nlb->klab", centered, centered) / samples.shape[0]
    variance = centered.square().mean(dim=0)
    scale = torch.sqrt(
        variance[:, None, :, None].clamp_min(eps)
        * variance[None, :, None, :].clamp_min(eps)
    )
    # Round-off can otherwise produce values just outside the mathematical
    # correlation range (for example 1.000000119 in float32).
    return (covariance / scale).clamp(min=-1.0, max=1.0)


def projector_orthogonality_loss(
    basis: Tensor,
    *,
    reduction: Reduction = "sum",
) -> Tensor:
    """Apply the within- and cross-projector Gram objective.

    ``basis`` has shape ``(K, r, d)`` and stores ``P_k.T`` by row.  The loss is
    ``sum_k ||P_k.T P_k - I||_F^2 + sum_{i<j} ||P_i.T P_j||_F^2``.
    ``reduction='none'`` returns the concatenated within- and cross-block scalar
    penalties; the composite objective uses ``reduction='sum'``.
    """

    if basis.ndim != 3:
        raise ValueError(f"basis must have shape (K, r, d), got {tuple(basis.shape)}")
    if not basis.is_floating_point():
        raise TypeError("basis must be floating point")
    num_factors, factor_dim, state_dim = basis.shape
    if num_factors <= 0 or factor_dim <= 0 or state_dim <= 0:
        raise ValueError("basis dimensions K, r, and d must be positive")
    if num_factors * factor_dim != state_dim:
        raise ValueError("complete projector Gram loss requires K * r == d")
    work = basis.float() if basis.dtype in (torch.float16, torch.bfloat16) else basis
    identity = torch.eye(factor_dim, device=work.device, dtype=work.dtype)
    within_grams = torch.einsum("kad,kbd->kab", work, work)
    within_penalties = (within_grams - identity).square().sum(dim=(-2, -1))

    cross_penalties: list[Tensor] = []
    for first in range(num_factors):
        for second in range(first + 1, num_factors):
            cross_gram = work[first] @ work[second].transpose(0, 1)
            cross_penalties.append(cross_gram.square().sum())
    if cross_penalties:
        penalties = torch.cat((within_penalties, torch.stack(cross_penalties)))
    else:
        penalties = within_penalties
    return _reduce(penalties, reduction)


def factor_orthogonality_loss(
    basis: Tensor,
    *,
    reduction: Reduction = "sum",
) -> Tensor:
    """Compatibility name for :func:`projector_orthogonality_loss`."""

    return projector_orthogonality_loss(basis, reduction=reduction)


def factor_decorrelation_loss(
    factors: Tensor,
    *,
    normalize: bool = True,
    eps: float = 1e-6,
    reduction: Reduction = "mean",
) -> Tensor:
    """Penalize statistical coupling between distinct predictive factors.

    This is a statistical diagnostic or optional auxiliary objective.  It is not
    the projector-Gram orthogonality term, and statistical
    decorrelation does not establish geometric subspace orthogonality.

    With ``reduction='none'``, the returned tensor has shape
    ``(K * (K - 1), r, r)``.  In particular, ``K=1`` returns an empty
    ``(0, r, r)`` tensor; scalar reductions return a differentiable zero.

    Float16 and bfloat16 inputs are accumulated in float32.  Covariance uses
    population normalization (``correction=0``), matching the activity losses
    and variance diagnostics.
    """

    _validate_eps(eps)
    samples = _accumulation_samples(factors)
    num_factors = samples.shape[1]
    if num_factors < 2:
        zero = samples.sum() * 0.0
        if reduction == "none":
            return samples.new_empty((0, samples.shape[-1], samples.shape[-1]))
        if reduction not in ("mean", "sum"):
            raise ValueError(f"unsupported reduction: {reduction!r}")
        return zero

    if normalize:
        blocks = factor_cross_correlation(samples, eps=eps)
    else:
        centered = samples - samples.mean(dim=0, keepdim=True)
        blocks = torch.einsum("nka,nlb->klab", centered, centered) / samples.shape[0]

    off_diagonal = ~torch.eye(num_factors, device=samples.device, dtype=torch.bool)
    penalties = blocks[off_diagonal].square()
    return _reduce(penalties, reduction)


def factor_coordinate_standard_deviation(factors: Tensor) -> Tensor:
    """Return population standard deviation for every ``(K, r)`` coordinate."""

    return torch.sqrt(_coordinate_variance(factors).clamp_min(0.0))


def factor_standard_deviation(factors: Tensor) -> Tensor:
    """Return population RMS coordinate standard deviation per factor.

    No epsilon is added to the reported value, so a constant factor has exactly
    zero standard deviation.  Variance uses ``correction=0`` over all leading
    sample dimensions.  Float16 and bfloat16 reductions are promoted to float32.
    This factor-level RMS is a summary only; activity loss is evaluated for each
    coordinate by :func:`factor_activity_loss`.
    """

    coordinate_variance = _coordinate_variance(factors)
    return torch.sqrt(coordinate_variance.mean(dim=-1).clamp_min(0.0))


def factor_activity_loss(
    factors: Tensor,
    *,
    min_std: float = 0.1,
    eps: float = 1e-6,
    reduction: Reduction = "mean",
) -> Tensor:
    """Require every predictive factor to remain active across samples.

    Activity is evaluated independently for every one of the ``K * r`` target
    coordinates.  Standard deviation is ``sqrt(Var + eps)`` and the objective is
    the linear hinge ``max(0, min_std - std)``. Therefore
    one high-variance coordinate cannot hide a collapsed coordinate in the same
    factor.

    A perfectly constant factor is a symmetric stationary point: this variance
    loss has zero gradient there.  It prevents near-collapse but cannot, by
    itself, break exact collapse; initialization or another asymmetric signal is
    required.
    """

    _validate_min_std(min_std)
    _validate_eps(eps)
    standard_deviation = torch.sqrt(_coordinate_variance(factors) + eps)
    penalties = torch.relu(min_std - standard_deviation)
    return _reduce(penalties, reduction)


def encoder_variance_loss(
    context_states: Tensor,
    *,
    valid_mask: Tensor | None = None,
    min_std: float = 0.1,
    eps: float = 1e-6,
    reduction: Reduction = "mean",
) -> Tensor:
    """Apply the online-encoder activity floor.

    Args:
        context_states: Online encoder representations with shape ``(..., d)``.
            All leading positions are variance samples.
        valid_mask: Optional boolean mask matching the leading shape.  This is
            intended for padded token-valued contexts; only valid tokens
            contribute to the statistic.
        min_std: Coordinate activity floor ``gamma_enc``.
        eps: Stabilizer used inside ``sqrt(Var + eps)``.
        reduction: ``'none'`` returns one penalty per encoder coordinate.
    """

    _validate_min_std(min_std)
    _validate_eps(eps)
    if context_states.ndim < 2:
        raise ValueError(
            "context_states must have shape (..., d) with a sample axis, got "
            f"{tuple(context_states.shape)}"
        )
    if not context_states.is_floating_point():
        raise TypeError("context_states must be floating point")
    if context_states.shape[-1] <= 0:
        raise ValueError("encoder dimension d must be positive")
    samples = context_states.reshape(-1, context_states.shape[-1])
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")
        if valid_mask.device != context_states.device:
            raise ValueError("valid_mask and context_states must be on the same device")
        if tuple(valid_mask.shape) != tuple(context_states.shape[:-1]):
            raise ValueError(
                "valid_mask must match context_states leading shape: expected "
                f"{tuple(context_states.shape[:-1])}, got {tuple(valid_mask.shape)}"
            )
        samples = samples[valid_mask.reshape(-1)]
    if samples.shape[0] == 0:
        raise ValueError("encoder variance requires at least one valid context sample")
    if samples.dtype in (torch.float16, torch.bfloat16):
        samples = samples.float()
    centered = samples - samples.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    penalties = torch.relu(min_std - torch.sqrt(variance + eps))
    return _reduce(penalties, reduction)


def online_encoder_variance_loss(
    context_states: Tensor,
    *,
    valid_mask: Tensor | None = None,
    min_std: float = 0.1,
    eps: float = 1e-6,
    reduction: Reduction = "mean",
) -> Tensor:
    """Alias for :func:`encoder_variance_loss`."""

    return encoder_variance_loss(
        context_states,
        valid_mask=valid_mask,
        min_std=min_std,
        eps=eps,
        reduction=reduction,
    )


class OnlineVarianceTracker(nn.Module):
    """Checkpointable Welford variance state for ``(K, r)`` coordinates.

    The tracker stores an exact int64 sample count plus mean and sum of squared
    deviations.  Statistics use at least float32 even when a containing model is
    cast with ``half()`` or ``bfloat16()``.  Batch updates use the parallel
    Welford merge formula and are performed without gradients.
    :meth:`combined_variance` offers a differentiable population-variance view
    of the variance that would result from adding the current batch.
    """

    count: Tensor
    mean: Tensor
    m2: Tensor

    def __init__(
        self,
        num_factors: int,
        factor_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if num_factors <= 0 or factor_dim <= 0:
            raise ValueError("num_factors and factor_dim must be positive")
        dtype_probe = torch.empty((), dtype=dtype)
        if not dtype_probe.is_floating_point():
            raise TypeError("tracker dtype must be floating point")
        statistics_dtype = (
            torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        )
        self.num_factors = num_factors
        self.factor_dim = factor_dim
        self.register_buffer("count", torch.zeros((), dtype=torch.int64, device=device))
        self.register_buffer(
            "mean",
            torch.zeros(num_factors, factor_dim, dtype=statistics_dtype, device=device),
        )
        self.register_buffer(
            "m2",
            torch.zeros(num_factors, factor_dim, dtype=statistics_dtype, device=device),
        )

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> OnlineVarianceTracker:
        """Apply device/dtype transforms without degrading statistics below fp32."""

        # Module._apply accepts a tensor transform callable.  Keep snapshots
        # because super()._apply may first quantize a float64/float32 buffer.
        mean_before = self.mean.detach().clone()
        m2_before = self.m2.detach().clone()
        super()._apply(fn, recurse=recurse)
        if self.mean.dtype in (torch.float16, torch.bfloat16):
            preserved_dtype = (
                torch.float64 if mean_before.dtype == torch.float64 else torch.float32
            )
            self.mean = mean_before.to(device=self.mean.device, dtype=preserved_dtype)
            self.m2 = m2_before.to(device=self.m2.device, dtype=preserved_dtype)
        return self

    def _samples(self, factors: Tensor) -> Tensor:
        samples = _flatten_factor_samples(factors)
        if tuple(samples.shape[-2:]) != (self.num_factors, self.factor_dim):
            raise ValueError(
                "factor dimensions do not match tracker: expected "
                f"({self.num_factors}, {self.factor_dim}), got {tuple(samples.shape[-2:])}"
            )
        if samples.device != self.mean.device:
            raise ValueError("factors and tracker must be on the same device")
        if not torch.isfinite(samples).all():
            raise ValueError("factors contain non-finite values; tracker state was not updated")
        return samples

    def _working_samples(self, factors: Tensor) -> Tensor:
        samples = self._samples(factors)
        accumulation_dtype = torch.promote_types(samples.dtype, self.mean.dtype)
        if accumulation_dtype in (torch.float16, torch.bfloat16):
            accumulation_dtype = torch.float32
        return samples.to(dtype=accumulation_dtype)

    @torch.no_grad()
    def reset(self) -> None:
        """Clear all accumulated sufficient statistics."""

        self.count.zero_()
        self.mean.zero_()
        self.m2.zero_()

    @torch.no_grad()
    def update(self, factors: Tensor) -> None:
        """Merge a batch into the running sufficient statistics."""

        samples = self._working_samples(factors).detach()
        batch_count = samples.shape[0]
        batch_mean = samples.mean(dim=0)
        batch_m2 = (samples - batch_mean).square().sum(dim=0)

        old_count = int(self.count.item())
        total_count = old_count + batch_count
        if total_count > torch.iinfo(torch.int64).max:
            raise OverflowError("tracker sample count exceeds int64 capacity")
        old_mean = self.mean.to(dtype=samples.dtype)
        old_m2 = self.m2.to(dtype=samples.dtype)
        delta = batch_mean - old_mean
        merged_mean = old_mean + delta * (batch_count / total_count)
        correction = delta.square() * (old_count * batch_count / total_count)
        merged_m2 = old_m2 + batch_m2 + correction

        mean_for_storage = merged_mean.to(dtype=self.mean.dtype)
        m2_for_storage = merged_m2.to(dtype=self.m2.dtype)
        if (
            not torch.isfinite(mean_for_storage).all()
            or not torch.isfinite(m2_for_storage).all()
        ):
            raise ValueError(
                "variance update produced non-finite statistics; state was not updated"
            )

        self.mean.copy_(mean_for_storage)
        self.m2.copy_(m2_for_storage)
        self.count.fill_(int(total_count))

    def variance(self, *, unbiased: bool = False) -> Tensor:
        """Return the current coordinate variance without changing state."""

        count = int(self.count.item())
        denominator = count - 1 if unbiased else count
        if denominator <= 0:
            return torch.zeros_like(self.m2)
        return self.m2 / denominator

    def combined_variance(self, factors: Tensor, *, unbiased: bool = False) -> Tensor:
        """Differentiably estimate variance after hypothetically adding a batch."""

        samples = self._working_samples(factors)
        batch_count = samples.shape[0]

        batch_mean = samples.mean(dim=0)
        batch_m2 = (samples - batch_mean).square().sum(dim=0)
        # Clones prevent a later update() from invalidating tensors saved by autograd.
        old_mean = self.mean.detach().clone().to(dtype=samples.dtype)
        old_m2 = self.m2.detach().clone().to(dtype=samples.dtype)
        old_count = int(self.count.item())
        total_count = old_count + batch_count
        delta = batch_mean - old_mean
        correction = delta.square() * (old_count * batch_count / total_count)
        total_m2 = old_m2 + batch_m2 + correction
        denominator = total_count - 1.0 if unbiased else total_count
        if denominator <= 0:
            result = torch.zeros_like(total_m2)
        else:
            result = total_m2 / denominator
        if not torch.isfinite(result).all():
            raise ValueError("combined variance produced non-finite statistics")
        return result


def streaming_factor_activity_loss(
    factors: Tensor,
    tracker: OnlineVarianceTracker,
    *,
    min_std: float = 0.1,
    eps: float = 1e-6,
    update: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Streaming factor-activity monitor using current and historical samples.

    The population variance (``correction=0``) is computed with a differentiable
    Welford merge, so gradients flow to the current batch while history remains
    detached. This is useful for monitoring tiny batches, but it is distinct
    from :func:`encoder_variance_loss`, which operates directly on the current
    mini-batch context representation.
    """

    _validate_min_std(min_std)
    _validate_eps(eps)
    coordinate_variance = tracker.combined_variance(factors)
    standard_deviation = torch.sqrt(coordinate_variance + eps)
    penalties = torch.relu(min_std - standard_deviation)
    loss = _reduce(penalties, reduction)
    if update:
        tracker.update(factors)
    return loss


def online_variance_loss(
    factors: Tensor,
    tracker: OnlineVarianceTracker,
    *,
    min_std: float = 0.1,
    eps: float = 1e-6,
    update: bool = True,
    reduction: Reduction = "mean",
) -> Tensor:
    """Backward-compatible alias for :func:`streaming_factor_activity_loss`.

    Despite its historical name, this function does not implement ``L_enc``.
    """

    return streaming_factor_activity_loss(
        factors,
        tracker,
        min_std=min_std,
        eps=eps,
        update=update,
        reduction=reduction,
    )


class OnlineVarianceLoss(nn.Module):
    """Compatibility wrapper for streaming factor-activity monitoring.

    Use :func:`encoder_variance_loss` for current-batch online-encoder activity.
    """

    def __init__(
        self,
        num_factors: int,
        factor_dim: int,
        *,
        min_std: float = 0.1,
        eps: float = 1e-6,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        _validate_min_std(min_std)
        _validate_eps(eps)
        self.min_std = min_std
        self.eps = eps
        self.reduction = reduction
        self.tracker = OnlineVarianceTracker(num_factors, factor_dim)

    def forward(self, factors: Tensor, *, update: bool | None = None) -> Tensor:
        """Evaluate the online loss and optionally update streaming state.

        If ``update`` is omitted, statistics update only in training mode.
        """

        should_update = self.training if update is None else update
        return online_variance_loss(
            factors,
            self.tracker,
            min_std=self.min_std,
            eps=self.eps,
            update=should_update,
            reduction=self.reduction,
        )


def factor_prediction_loss(predicted_factors: Tensor, target_factors: Tensor) -> Tensor:
    """Directly regress factor direction and magnitude.

    Both tensors must have shape ``(..., K, r)``.  The mean squared error over
    every sample, target, factor, and coordinate is normalized by the complete
    tensor element count.

    The caller should stop gradients through the EMA target *encoder output*
    before projecting it.  Do not detach ``target_factors`` here: projector
    gradients from prediction and factor activity must remain available.
    """

    if predicted_factors.shape != target_factors.shape:
        raise ValueError(
            "predicted_factors and target_factors must have identical shapes, got "
            f"{tuple(predicted_factors.shape)} and {tuple(target_factors.shape)}"
        )
    if predicted_factors.ndim < 2:
        raise ValueError("factor predictions must have shape (..., K, r)")
    if predicted_factors.shape[-2] <= 0 or predicted_factors.shape[-1] <= 0:
        raise ValueError("factor dimensions K and r must be positive")
    if predicted_factors.device != target_factors.device:
        raise ValueError("predicted_factors and target_factors must be on the same device")
    if predicted_factors.dtype != target_factors.dtype:
        raise ValueError("predicted_factors and target_factors must have the same dtype")
    if not predicted_factors.is_floating_point():
        raise TypeError("factor predictions must be floating point")
    if predicted_factors.numel() == 0:
        raise ValueError("factor predictions must not be empty")
    difference = predicted_factors - target_factors
    if difference.dtype in (torch.float16, torch.bfloat16):
        difference = difference.float()
    return difference.square().mean()


@dataclass(frozen=True)
class JEPAAnythingLoss:
    """Named components of the composite JEPA Anything objective."""

    total: Tensor
    prediction: Tensor
    orthogonality: Tensor
    factor_activity: Tensor
    encoder_variance: Tensor


def jepa_anything_objective(
    predicted_factors: Tensor,
    target_factors: Tensor,
    analysis_basis: Tensor,
    context_states: Tensor,
    *,
    valid_context_mask: Tensor | None = None,
    orthogonality_weight: float = 1.0,
    factor_activity_weight: float = 1.0,
    encoder_variance_weight: float = 1.0,
    factor_min_std: float = 0.1,
    encoder_min_std: float = 0.1,
    eps: float = 1e-6,
) -> JEPAAnythingLoss:
    """Compose the four deterministic JEPA Anything loss terms.

    This pure function intentionally performs no optimizer or EMA update.  It is
    suitable for generated task skeletons while keeping training lifecycle and
    target-encoder stop-gradient decisions explicit at the call site.

    Unlike :func:`factor_prediction_loss`, this composite requires at least one
    leading sample axis because both activity terms estimate population variance.
    """

    weights = {
        "orthogonality_weight": orthogonality_weight,
        "factor_activity_weight": factor_activity_weight,
        "encoder_variance_weight": encoder_variance_weight,
    }
    invalid = [name for name, value in weights.items() if not math.isfinite(value) or value < 0]
    if invalid:
        raise ValueError(
            "objective weights must be finite and non-negative; invalid: " + ", ".join(invalid)
        )
    if analysis_basis.ndim != 3:
        raise ValueError(
            "analysis_basis must have shape (K, r, d), got "
            f"{tuple(analysis_basis.shape)}"
        )
    num_factors, factor_dim, state_dim = analysis_basis.shape
    if predicted_factors.ndim < 3:
        raise ValueError(
            "the composite objective requires factor samples with shape (..., K, r)"
        )
    if predicted_factors.shape[-2:] != (
        num_factors,
        factor_dim,
    ):
        raise ValueError(
            "factor predictions must end in the analysis basis dimensions "
            f"({num_factors}, {factor_dim}), got {tuple(predicted_factors.shape)}"
        )
    if context_states.ndim < 2 or context_states.shape[-1] != state_dim:
        raise ValueError(
            "context_states must end in the analysis basis state dimension "
            f"{state_dim}, got {tuple(context_states.shape)}"
        )
    objective_tensors = {
        "target_factors": target_factors,
        "analysis_basis": analysis_basis,
        "context_states": context_states,
    }
    wrong_devices = [
        name
        for name, tensor in objective_tensors.items()
        if tensor.device != predicted_factors.device
    ]
    if wrong_devices:
        raise ValueError(
            "all objective tensors must be on the same device; mismatched: "
            + ", ".join(wrong_devices)
        )
    prediction = factor_prediction_loss(predicted_factors, target_factors)
    orthogonality = projector_orthogonality_loss(analysis_basis)
    factor_activity = factor_activity_loss(target_factors, min_std=factor_min_std, eps=eps)
    encoder_variance = encoder_variance_loss(
        context_states,
        valid_mask=valid_context_mask,
        min_std=encoder_min_std,
        eps=eps,
    )
    total = (
        prediction
        + orthogonality_weight * orthogonality
        + factor_activity_weight * factor_activity
        + encoder_variance_weight * encoder_variance
    )
    return JEPAAnythingLoss(
        total=total,
        prediction=prediction,
        orthogonality=orthogonality,
        factor_activity=factor_activity,
        encoder_variance=encoder_variance,
    )
