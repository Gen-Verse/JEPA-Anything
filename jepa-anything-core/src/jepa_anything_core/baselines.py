"""Standard JEPA and capacity-matched unconstrained multi-head baselines."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .opf import validate_factorization


def parameter_count(module: nn.Module, *, trainable_only: bool = True) -> int:
    """Count scalar parameters, optionally excluding frozen parameters."""

    parameters = module.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


@dataclass(frozen=True)
class CapacityMatchReport:
    """Serializable parameter and optional prediction-FLOP comparison."""

    reference_parameters: int
    candidate_parameters: int
    absolute_difference: int
    relative_difference: float
    tolerance: float
    parameter_matched: bool
    reference_flops: int | None
    candidate_flops: int | None
    flop_absolute_difference: int | None
    flop_relative_difference: float | None
    flop_tolerance: float | None
    flops_matched: bool | None
    matched: bool

    def to_dict(self) -> dict[str, int | float | bool | None]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def capacity_match_report(
    reference: nn.Module,
    candidate: nn.Module,
    *,
    tolerance: float = 0.0,
    trainable_only: bool = True,
    reference_flops: int | None = None,
    candidate_flops: int | None = None,
    flop_tolerance: float = 0.0,
) -> CapacityMatchReport:
    """Compare parameter counts and, when supplied, prediction FLOP budgets."""

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not math.isfinite(flop_tolerance) or flop_tolerance < 0:
        raise ValueError("flop_tolerance must be finite and non-negative")
    if (reference_flops is None) != (candidate_flops is None):
        raise ValueError("reference_flops and candidate_flops must be supplied together")
    flop_values = (("reference_flops", reference_flops), ("candidate_flops", candidate_flops))
    for name, value in flop_values:
        invalid_flop_value = value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        )
        if invalid_flop_value:
            raise ValueError(f"{name} must be a non-negative integer")
    reference_parameters = parameter_count(reference, trainable_only=trainable_only)
    candidate_parameters = parameter_count(candidate, trainable_only=trainable_only)
    difference = abs(candidate_parameters - reference_parameters)
    relative = difference / max(reference_parameters, 1)
    parameter_matched = relative <= tolerance
    if reference_flops is None or candidate_flops is None:
        flop_difference = None
        flop_relative = None
        recorded_flop_tolerance = None
        flops_matched = None
    else:
        flop_difference = abs(candidate_flops - reference_flops)
        flop_relative = flop_difference / max(reference_flops, 1)
        recorded_flop_tolerance = flop_tolerance
        flops_matched = flop_relative <= flop_tolerance
    return CapacityMatchReport(
        reference_parameters=reference_parameters,
        candidate_parameters=candidate_parameters,
        absolute_difference=difference,
        relative_difference=relative,
        tolerance=tolerance,
        parameter_matched=parameter_matched,
        reference_flops=reference_flops,
        candidate_flops=candidate_flops,
        flop_absolute_difference=flop_difference,
        flop_relative_difference=flop_relative,
        flop_tolerance=recorded_flop_tolerance,
        flops_matched=flops_matched,
        matched=parameter_matched and flops_matched is not False,
    )


def _make_trunk(input_dim: int, hidden_dim: int, depth: int) -> tuple[nn.Module, int]:
    if input_dim <= 0 or hidden_dim <= 0:
        raise ValueError("input_dim and hidden_dim must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if depth == 1:
        return nn.Identity(), input_dim
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    for _ in range(depth - 2):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU()))
    return nn.Sequential(*layers), hidden_dim


class MLPPredictor(nn.Module):
    """Shared-trunk MLP predictor used by the standard JEPA baseline."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, *, depth: int = 2) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.trunk, trunk_dim = _make_trunk(input_dim, hidden_dim, depth)
        self.output = nn.Linear(trunk_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, context_state: Tensor) -> Tensor:
        """Predict a complete target state from a context state."""

        if context_state.shape[-1] != self.input_dim:
            raise ValueError(f"context_state must end in dimension {self.input_dim}")
        return self.output(self.trunk(context_state))


class UnconstrainedMultiHeadPredictor(nn.Module):
    """Shared-trunk predictor with ``K`` independent, unconstrained output heads.

    No orthogonal basis, cross-head constraint, or semantic factor label is
    imposed.  With identical dimensions, its parameter count exactly matches
    :class:`MLPPredictor`: splitting one ``d``-wide output layer into ``K``
    ``r``-wide layers changes grouping, not capacity.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int,
        *,
        depth: int = 2,
    ) -> None:
        super().__init__()
        validate_factorization(state_dim, num_heads, head_dim)
        self.trunk, trunk_dim = _make_trunk(input_dim, hidden_dim, depth)
        self.heads = nn.ModuleList(nn.Linear(trunk_dim, head_dim) for _ in range(num_heads))
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, context_state: Tensor) -> Tensor:
        """Return unconstrained head outputs with trailing shape ``(K, r)``."""

        if context_state.shape[-1] != self.input_dim:
            raise ValueError(f"context_state must end in dimension {self.input_dim}")
        hidden = self.trunk(context_state)
        return torch.stack([head(hidden) for head in self.heads], dim=-2)

    def flatten_output(self, head_predictions: Tensor) -> Tensor:
        """Flatten ``(..., K, r)`` head predictions to ``(..., d)``."""

        if head_predictions.shape[-2:] != (self.num_heads, self.head_dim):
            raise ValueError(
                f"head_predictions must end in ({self.num_heads}, {self.head_dim})"
            )
        return head_predictions.flatten(start_dim=-2)


def build_capacity_matched_predictors(
    input_dim: int,
    state_dim: int,
    num_factors: int,
    factor_dim: int,
    hidden_dim: int,
    *,
    depth: int = 2,
) -> tuple[MLPPredictor, UnconstrainedMultiHeadPredictor]:
    """Build standard and multi-head predictors with exactly equal capacity."""

    validate_factorization(state_dim, num_factors, factor_dim)
    standard = MLPPredictor(input_dim, state_dim, hidden_dim, depth=depth)
    multi_head = UnconstrainedMultiHeadPredictor(
        input_dim,
        state_dim,
        num_factors,
        factor_dim,
        hidden_dim,
        depth=depth,
    )
    report = capacity_match_report(standard, multi_head)
    if not report.matched:  # pragma: no cover - guards future architecture edits
        raise RuntimeError(f"predictor construction is not capacity matched: {report}")
    return standard, multi_head


@dataclass(frozen=True)
class JEPAPrediction:
    """Outputs shared by the standard and multi-head JEPA baselines."""

    prediction: Tensor
    target: Tensor
    context_state: Tensor
    head_predictions: Tensor | None = None


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    """Update a target encoder from its online encoder using exponential averaging."""

    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must lie in [0, 1)")
    if target is online:
        raise ValueError("target and online encoders must be distinct module instances")

    target_parameters = dict(target.named_parameters())
    online_parameters = dict(online.named_parameters())
    if target_parameters.keys() != online_parameters.keys():
        raise ValueError("target and online encoders have different parameter structures")

    # Validate the entire update before mutating any tensor.  This keeps a failed
    # update atomic and also prevents an accidentally shared tensor from being
    # read after it has already been modified in place.
    for name, target_parameter in target_parameters.items():
        online_parameter = online_parameters[name]
        if target_parameter.shape != online_parameter.shape:
            raise ValueError(f"encoder parameter shape mismatch at {name!r}")
        if target_parameter.dtype != online_parameter.dtype:
            raise ValueError(f"encoder parameter dtype mismatch at {name!r}")
        if target_parameter.device != online_parameter.device:
            raise ValueError(f"encoder parameter device mismatch at {name!r}")
        if target_parameter is online_parameter:
            raise ValueError(f"target and online encoders share parameter {name!r}")
        if not (target_parameter.is_floating_point() or target_parameter.is_complex()):
            raise ValueError(f"encoder parameter at {name!r} must be floating point or complex")

    target_buffers = dict(target.named_buffers())
    online_buffers = dict(online.named_buffers())
    if target_buffers.keys() != online_buffers.keys():
        raise ValueError("target and online encoders have different buffer structures")
    for name, target_buffer in target_buffers.items():
        online_buffer = online_buffers[name]
        if target_buffer.shape != online_buffer.shape:
            raise ValueError(f"encoder buffer shape mismatch at {name!r}")
        if target_buffer.dtype != online_buffer.dtype:
            raise ValueError(f"encoder buffer dtype mismatch at {name!r}")
        if target_buffer.device != online_buffer.device:
            raise ValueError(f"encoder buffer device mismatch at {name!r}")
        if target_buffer is online_buffer:
            raise ValueError(f"target and online encoders share buffer {name!r}")

    for name, target_parameter in target_parameters.items():
        online_parameter = online_parameters[name]
        target_parameter.mul_(momentum).add_(online_parameter, alpha=1.0 - momentum)

    for name, target_buffer in target_buffers.items():
        online_buffer = online_buffers[name]
        if target_buffer.is_floating_point() or target_buffer.is_complex():
            target_buffer.mul_(momentum).add_(online_buffer, alpha=1.0 - momentum)
        else:
            target_buffer.copy_(online_buffer)


class _BaseJEPABaseline(nn.Module):
    """Common online/target encoder lifecycle for baseline models."""

    def __init__(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module | None,
        *,
        target_momentum: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= target_momentum < 1.0:
            raise ValueError("target_momentum must lie in [0, 1)")
        if target_encoder is context_encoder:
            raise ValueError("target_encoder must be a distinct module instance")
        self.context_encoder = context_encoder
        self.target_encoder = (
            copy.deepcopy(context_encoder) if target_encoder is None else target_encoder
        )
        self.target_momentum = target_momentum
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        """EMA-update the target encoder; call once after each optimizer step."""

        ema_update(
            self.target_encoder,
            self.context_encoder,
            self.target_momentum if momentum is None else momentum,
        )

    def train(self, mode: bool = True) -> _BaseJEPABaseline:
        """Set training mode while keeping the stop-gradient target encoder in eval mode."""

        super().train(mode)
        self.target_encoder.eval()
        return self

    def _encode(
        self,
        context_observation: Tensor,
        target_observation: Tensor,
    ) -> tuple[Tensor, Tensor]:
        context_state = self.context_encoder(context_observation)
        self.target_encoder.eval()
        with torch.no_grad():
            target_state = self.target_encoder(target_observation)
        return context_state, target_state.detach()


class StandardJEPABaseline(_BaseJEPABaseline):
    """Standard joint-embedding prediction with an EMA target encoder."""

    def __init__(
        self,
        context_encoder: nn.Module,
        predictor: nn.Module,
        target_encoder: nn.Module | None = None,
        *,
        target_momentum: float = 0.996,
    ) -> None:
        super().__init__(context_encoder, target_encoder, target_momentum=target_momentum)
        self.predictor = predictor

    def forward(self, context_observation: Tensor, target_observation: Tensor) -> JEPAPrediction:
        """Encode context/target and predict the complete target state."""

        context_state, target_state = self._encode(context_observation, target_observation)
        prediction = self.predictor(context_state)
        if prediction.shape != target_state.shape:
            raise ValueError(
                "prediction and target encoder output must have identical shapes, got "
                f"{tuple(prediction.shape)} and {tuple(target_state.shape)}"
            )
        return JEPAPrediction(prediction, target_state, context_state)


class UnconstrainedMultiHeadJEPABaseline(_BaseJEPABaseline):
    """JEPA ablation with capacity-matched, unconstrained prediction heads."""

    predictor: UnconstrainedMultiHeadPredictor

    def __init__(
        self,
        context_encoder: nn.Module,
        predictor: UnconstrainedMultiHeadPredictor,
        target_encoder: nn.Module | None = None,
        *,
        target_momentum: float = 0.996,
    ) -> None:
        super().__init__(context_encoder, target_encoder, target_momentum=target_momentum)
        self.predictor = predictor

    def forward(self, context_observation: Tensor, target_observation: Tensor) -> JEPAPrediction:
        """Predict target coordinates through independent unconstrained heads."""

        context_state, target_state = self._encode(context_observation, target_observation)
        head_predictions = self.predictor(context_state)
        prediction = self.predictor.flatten_output(head_predictions)
        if prediction.shape != target_state.shape:
            raise ValueError(
                "flattened head prediction and target encoder output must have "
                "identical shapes, got "
                f"{tuple(prediction.shape)} and {tuple(target_state.shape)}"
            )
        return JEPAPrediction(prediction, target_state, context_state, head_predictions)
