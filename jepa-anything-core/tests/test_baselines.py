from __future__ import annotations

import pytest
import torch
from torch import nn

from jepa_anything_core import (
    StandardJEPABaseline,
    UnconstrainedMultiHeadJEPABaseline,
    build_capacity_matched_predictors,
    capacity_match_report,
    ema_update,
)


def test_predictor_builder_is_exactly_capacity_matched() -> None:
    standard, multi_head = build_capacity_matched_predictors(
        input_dim=5,
        state_dim=6,
        num_factors=3,
        factor_dim=2,
        hidden_dim=11,
        depth=3,
    )
    report = capacity_match_report(standard, multi_head)

    assert report.matched
    assert report.absolute_difference == 0
    assert report.parameter_matched
    assert report.flops_matched is None
    assert report.to_dict()["candidate_parameters"] == report.reference_parameters
    assert standard(torch.randn(4, 5)).shape == (4, 6)
    assert multi_head(torch.randn(4, 5)).shape == (4, 3, 2)


def test_capacity_report_can_enforce_prediction_flop_tolerance() -> None:
    standard, multi_head = build_capacity_matched_predictors(5, 6, 3, 2, 11)
    passing = capacity_match_report(
        standard,
        multi_head,
        reference_flops=10_000,
        candidate_flops=10_100,
        flop_tolerance=0.02,
    )
    failing = capacity_match_report(
        standard,
        multi_head,
        reference_flops=10_000,
        candidate_flops=10_300,
        flop_tolerance=0.02,
    )

    assert passing.matched and passing.flops_matched
    assert not failing.matched and failing.flops_matched is False
    with pytest.raises(ValueError, match="supplied together"):
        capacity_match_report(standard, multi_head, reference_flops=10_000)


def test_standard_jepa_stops_target_gradients_and_updates_with_ema() -> None:
    encoder = nn.Linear(4, 6)
    standard_predictor, _ = build_capacity_matched_predictors(6, 6, 3, 2, 8)
    model = StandardJEPABaseline(encoder, standard_predictor, target_momentum=0.5)
    context = torch.randn(3, 4)
    target = torch.randn(3, 4)

    output = model(context, target)
    assert output.prediction.shape == (3, 6)
    assert output.target.shape == (3, 6)
    assert not output.target.requires_grad
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())

    old_target = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(2.0)
    online = next(model.context_encoder.parameters()).detach().clone()
    model.update_target_encoder()
    updated_target = next(model.target_encoder.parameters()).detach()
    torch.testing.assert_close(updated_target, old_target * 0.5 + online * 0.5)


def test_unconstrained_multihead_jepa_returns_flat_and_head_views() -> None:
    encoder = nn.Linear(4, 6)
    _, multi_head_predictor = build_capacity_matched_predictors(6, 6, 3, 2, 8)
    model = UnconstrainedMultiHeadJEPABaseline(encoder, multi_head_predictor)

    output = model(torch.randn(2, 4), torch.randn(2, 4))

    assert output.prediction.shape == (2, 6)
    assert output.target.shape == (2, 6)
    assert output.head_predictions is not None
    assert output.head_predictions.shape == (2, 3, 2)
    torch.testing.assert_close(output.prediction, output.head_predictions.flatten(start_dim=-2))


def test_target_encoder_remains_in_eval_mode() -> None:
    encoder = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6))
    predictor, _ = build_capacity_matched_predictors(6, 6, 2, 3, 8)
    model = StandardJEPABaseline(encoder, predictor)

    model.train()

    assert model.context_encoder.training
    assert not model.target_encoder.training


def test_ema_momentum_excludes_frozen_value_one() -> None:
    encoder = nn.Linear(4, 6)
    predictor, _ = build_capacity_matched_predictors(6, 6, 2, 3, 8)

    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        StandardJEPABaseline(encoder, predictor, target_momentum=1.0)


def test_target_encoder_cannot_alias_context_encoder() -> None:
    encoder = nn.Linear(4, 6)
    predictor, _ = build_capacity_matched_predictors(6, 6, 2, 3, 8)

    with pytest.raises(ValueError, match="distinct module"):
        StandardJEPABaseline(encoder, predictor, target_encoder=encoder)
    with pytest.raises(ValueError, match="distinct module"):
        ema_update(encoder, encoder, 0.5)


def test_ema_validation_is_atomic_on_late_shape_mismatch() -> None:
    target = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    online = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3))
    before = target[0].weight.detach().clone()

    with pytest.raises(ValueError, match="shape mismatch"):
        ema_update(target, online, 0.5)

    torch.testing.assert_close(target[0].weight, before)


def test_ema_rejects_shared_parameter_without_mutating_it() -> None:
    target = nn.Linear(2, 2)
    online = nn.Linear(2, 2)
    online.weight = target.weight
    before = target.weight.detach().clone()

    with pytest.raises(ValueError, match="share parameter"):
        ema_update(target, online, 0.5)

    torch.testing.assert_close(target.weight, before)
