from __future__ import annotations

import pytest
import torch

from jepa_anything_core import (
    OnlineVarianceLoss,
    OnlineVarianceTracker,
    encoder_variance_loss,
    factor_activity_loss,
    factor_coordinate_standard_deviation,
    factor_cross_correlation,
    factor_decorrelation_loss,
    factor_orthogonality_loss,
    factor_prediction_loss,
    factor_standard_deviation,
    jepa_anything_objective,
    online_variance_loss,
    projector_orthogonality_loss,
)


def _orthogonal_samples() -> torch.Tensor:
    first = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    second = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    return torch.stack((first, second), dim=-1).unsqueeze(-1)


def test_projector_orthogonality_loss_matches_gram_contract() -> None:
    basis = torch.eye(4).reshape(2, 2, 4)
    duplicated_direction = basis.clone()
    duplicated_direction[1, 0] = duplicated_direction[0, 0]

    assert projector_orthogonality_loss(basis).item() == 0.0
    assert factor_orthogonality_loss(basis).item() == 0.0
    assert projector_orthogonality_loss(duplicated_direction).item() > 0.9


def test_decorrelation_diagnostic_detects_cross_factor_correlation() -> None:
    uncoupled = _orthogonal_samples()
    coupled = torch.stack((uncoupled[:, 0], uncoupled[:, 0]), dim=1)

    assert factor_decorrelation_loss(uncoupled).item() < 1e-7
    assert factor_decorrelation_loss(coupled).item() > 0.9


def test_correlation_is_clamped_to_its_mathematical_range() -> None:
    generator = torch.Generator().manual_seed(27)
    first = torch.randn(257, 1, generator=generator)
    duplicated = torch.stack((first, first), dim=1)
    correlations = factor_cross_correlation(duplicated)

    assert correlations.abs().max().item() <= 1.0


def test_activity_loss_audits_each_factor_separately() -> None:
    samples = _orthogonal_samples()
    samples[:, 1] = 0.0
    penalties = factor_activity_loss(samples, min_std=0.5, reduction="none")

    assert penalties.shape == (2, 1)
    assert penalties[0, 0].item() == 0.0
    assert penalties[1, 0].item() > 0.4


def test_activity_loss_audits_every_coordinate_with_linear_hinge() -> None:
    active = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    samples = torch.stack((active, torch.zeros_like(active)), dim=-1).unsqueeze(1)

    standard_deviations = factor_coordinate_standard_deviation(samples)
    penalties = factor_activity_loss(samples, min_std=0.5, reduction="none")

    assert standard_deviations.shape == (1, 2)
    torch.testing.assert_close(standard_deviations, torch.tensor([[1.0, 0.0]]))
    assert penalties.shape == (1, 2)
    assert penalties[0, 0].item() == 0.0
    assert 0.49 < penalties[0, 1].item() < 0.5


def test_online_tracker_matches_direct_population_variance() -> None:
    generator = torch.Generator().manual_seed(9)
    first = torch.randn(7, 3, 2, generator=generator)
    second = torch.randn(5, 3, 2, generator=generator) + 0.4
    tracker = OnlineVarianceTracker(3, 2)

    tracker.update(first)
    combined_before_commit = tracker.combined_variance(second)
    expected = torch.cat((first, second), dim=0).var(dim=0, correction=0)
    torch.testing.assert_close(combined_before_commit, expected)

    tracker.update(second)
    assert tracker.count.item() == 12
    torch.testing.assert_close(tracker.variance(), expected)


def test_online_variance_loss_is_differentiable_and_updates_after_forward() -> None:
    tracker = OnlineVarianceTracker(2, 1)
    tracker.update(torch.tensor([[[-1.0], [0.0]], [[1.0], [0.0]]]))
    current = torch.tensor([[[0.2], [0.0]], [[0.4], [0.0]]], requires_grad=True)

    loss = online_variance_loss(current, tracker, min_std=0.5)
    loss.backward()

    assert tracker.count.item() == 4
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert loss.item() > 0.0


def test_online_variance_module_updates_only_while_training_by_default() -> None:
    loss_module = OnlineVarianceLoss(2, 1)
    samples = _orthogonal_samples()

    loss_module.eval()
    loss_module(samples)
    assert loss_module.tracker.count.item() == 0

    loss_module.train()
    loss_module(samples)
    assert loss_module.tracker.count.item() == 4


def test_population_variance_semantics_match_direct_and_online_small_batches() -> None:
    two_samples = torch.tensor([[[-1.0]], [[1.0]]])
    single_sample = torch.tensor([[[3.0]]])

    torch.testing.assert_close(factor_standard_deviation(two_samples), torch.ones(1))
    tracker = OnlineVarianceTracker(1, 1)
    tracker.update(two_samples)
    torch.testing.assert_close(tracker.variance(), torch.ones(1, 1))

    direct = factor_activity_loss(single_sample, min_std=0.5)
    online = online_variance_loss(
        single_sample,
        OnlineVarianceTracker(1, 1),
        min_std=0.5,
        update=False,
    )
    torch.testing.assert_close(online, direct)
    assert direct.item() > 0


def test_low_precision_loss_reductions_accumulate_in_float32() -> None:
    factors = _orthogonal_samples().to(torch.float16)
    basis = torch.eye(2, dtype=torch.float16).reshape(2, 1, 2)

    standard_deviation = factor_standard_deviation(factors)
    correlation = factor_cross_correlation(factors)
    activity = factor_activity_loss(factors)
    orthogonality = factor_orthogonality_loss(basis)
    online = online_variance_loss(
        factors,
        OnlineVarianceTracker(2, 1),
        update=False,
    )

    assert standard_deviation.dtype == torch.float32
    assert correlation.dtype == torch.float32
    assert activity.dtype == torch.float32
    assert orthogonality.dtype == torch.float32
    assert online.dtype == torch.float32
    torch.testing.assert_close(standard_deviation, factor_standard_deviation(factors.float()))


def test_tracker_count_and_statistics_survive_module_dtype_casts() -> None:
    tracker = OnlineVarianceTracker(2, 1)
    exact_count = 2**24 + 1
    tracker.count.fill_(exact_count)
    tracker.mean.copy_(torch.tensor([[1.234567], [-2.345678]]))
    expected_mean = tracker.mean.clone()

    tracker.half()
    assert tracker.count.dtype == torch.int64
    assert tracker.count.item() == exact_count
    assert tracker.mean.dtype == torch.float32
    torch.testing.assert_close(tracker.mean, expected_mean)

    tracker.bfloat16()
    assert tracker.count.dtype == torch.int64
    assert tracker.count.item() == exact_count
    assert tracker.m2.dtype == torch.float32

    tracker.float()
    assert tracker.count.dtype == torch.int64
    assert tracker.count.item() == exact_count


def test_tracker_rejects_nonfinite_batches_without_mutating_state() -> None:
    tracker = OnlineVarianceTracker(2, 1)
    tracker.update(torch.tensor([[[-1.0], [2.0]], [[1.0], [4.0]]]))
    before = {name: value.clone() for name, value in tracker.state_dict().items()}

    for bad_value in (float("nan"), float("inf")):
        bad = torch.tensor([[[bad_value], [0.0]]])
        with pytest.raises(ValueError, match="non-finite"):
            tracker.update(bad)
        with pytest.raises(ValueError, match="non-finite"):
            tracker.combined_variance(bad)
        for name, value in tracker.state_dict().items():
            torch.testing.assert_close(value, before[name])


def test_tracker_rejects_storage_overflow_before_mutating_state() -> None:
    tracker = OnlineVarianceTracker(1, 1)
    too_large_for_fp32 = torch.tensor([[[1e100]]], dtype=torch.float64)

    with pytest.raises(ValueError, match="non-finite statistics"):
        tracker.update(too_large_for_fp32)

    assert tracker.count.item() == 0
    assert tracker.mean.item() == 0
    assert tracker.m2.item() == 0


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_all_loss_stabilizers_require_finite_positive_eps(eps: float) -> None:
    factors = _orthogonal_samples()
    with pytest.raises(ValueError, match="finite and positive"):
        factor_cross_correlation(factors, eps=eps)
    with pytest.raises(ValueError, match="finite and positive"):
        factor_activity_loss(factors, eps=eps)
    with pytest.raises(ValueError, match="finite and positive"):
        encoder_variance_loss(torch.randn(4, 2), eps=eps)
    with pytest.raises(ValueError, match="finite and positive"):
        online_variance_loss(factors, OnlineVarianceTracker(2, 1), eps=eps)


@pytest.mark.parametrize("shape", [(4, 0, 1), (4, 1, 0)])
def test_factor_dimensions_must_be_positive(shape: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="K and r must be positive"):
        factor_standard_deviation(torch.empty(shape))


def test_single_factor_none_reduction_has_stable_empty_block_shape() -> None:
    factors = torch.randn(5, 1, 3)
    penalties = factor_decorrelation_loss(factors, reduction="none")

    assert penalties.shape == (0, 3, 3)
    assert penalties.dtype == factors.dtype


def test_encoder_variance_uses_only_valid_context_tokens() -> None:
    states = torch.tensor(
        [[[-1.0, 0.0], [99.0, 99.0]], [[1.0, 0.0], [-99.0, -99.0]]],
        requires_grad=True,
    )
    valid_mask = torch.tensor([[True, False], [True, False]])

    penalties = encoder_variance_loss(
        states,
        valid_mask=valid_mask,
        min_std=0.5,
        reduction="none",
    )
    penalties.sum().backward()

    assert penalties.shape == (2,)
    assert penalties[0].item() == 0.0
    assert 0.49 < penalties[1].item() < 0.5
    assert states.grad is not None
    torch.testing.assert_close(states.grad[:, 1], torch.zeros_like(states.grad[:, 1]))


def test_factor_prediction_and_composite_objective_are_differentiable() -> None:
    predicted = torch.zeros(3, 2, 1, requires_grad=True)
    target = torch.ones(3, 2, 1)
    basis = (torch.eye(2) * 1.1).reshape(2, 1, 2).requires_grad_()
    context = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], requires_grad=True)

    torch.testing.assert_close(factor_prediction_loss(predicted, target), torch.tensor(1.0))
    objective = jepa_anything_objective(
        predicted,
        target,
        basis,
        context,
        orthogonality_weight=0.5,
        factor_activity_weight=0.25,
        encoder_variance_weight=0.25,
    )
    objective.total.backward()

    assert predicted.grad is not None
    assert basis.grad is not None
    assert context.grad is not None
    torch.testing.assert_close(
        objective.total,
        objective.prediction
        + 0.5 * objective.orthogonality
        + 0.25 * objective.factor_activity
        + 0.25 * objective.encoder_variance,
    )


def test_factor_prediction_accepts_a_single_unbatched_factor_state() -> None:
    predicted = torch.zeros(2, 3)
    target = torch.ones(2, 3)

    torch.testing.assert_close(factor_prediction_loss(predicted, target), torch.tensor(1.0))


@pytest.mark.parametrize(
    ("predicted_shape", "basis_shape", "context_shape", "message"),
    [
        ((3, 2, 2), (3, 2, 6), (3, 6), "factor predictions must end"),
        ((3, 3, 2), (3, 2, 6), (3, 9), "context_states must end"),
    ],
)
def test_composite_objective_rejects_unrelated_geometry(
    predicted_shape: tuple[int, ...],
    basis_shape: tuple[int, ...],
    context_shape: tuple[int, ...],
    message: str,
) -> None:
    predicted = torch.zeros(predicted_shape)
    target = torch.ones(predicted_shape)
    basis = torch.eye(6).reshape(basis_shape)
    context = torch.zeros(context_shape)

    with pytest.raises(ValueError, match=message):
        jepa_anything_objective(predicted, target, basis, context)


def test_composite_objective_requires_a_factor_sample_axis() -> None:
    predicted = torch.zeros(2, 1)
    target = torch.ones(2, 1)
    basis = torch.eye(2).reshape(2, 1, 2)
    context = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="requires factor samples"):
        jepa_anything_objective(predicted, target, basis, context)


def test_exactly_constant_variance_objectives_are_zero_gradient_stationary_points() -> None:
    direct_factors = torch.zeros(4, 2, 1, requires_grad=True)
    direct_loss = factor_activity_loss(direct_factors, min_std=0.5)
    direct_loss.backward()
    assert direct_factors.grad is not None
    torch.testing.assert_close(direct_factors.grad, torch.zeros_like(direct_factors))

    online_factors = torch.zeros(4, 2, 1, requires_grad=True)
    online_loss = online_variance_loss(
        online_factors,
        OnlineVarianceTracker(2, 1),
        min_std=0.5,
        update=False,
    )
    online_loss.backward()
    assert online_factors.grad is not None
    torch.testing.assert_close(online_factors.grad, torch.zeros_like(online_factors))
