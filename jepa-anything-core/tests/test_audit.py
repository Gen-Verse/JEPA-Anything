from __future__ import annotations

import json

import pytest
import torch

from jepa_anything_core import (
    OrthogonalFactorProjection,
    audit_basis_geometry,
    audit_factor_geometry,
    audit_opf_geometry,
    audit_state_round_trip,
    audit_transpose_synthesis,
    default_geometry_tolerance,
    orthonormalize_basis,
)


def _auditable_states(repeats: int = 64) -> torch.Tensor:
    first = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    second = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    block = torch.stack((first, second), dim=-1)
    return block.repeat(repeats, 1)


def test_basis_audit_reports_pass_and_failure() -> None:
    valid = torch.eye(4).reshape(2, 2, 4)
    passing = audit_basis_geometry(valid)
    assert passing.passed
    assert passing.rank == 4
    assert passing.minimum_singular_value == 1.0
    assert passing.condition_number == 1.0
    assert passing.max_cross_subspace_overlap == 0.0
    json.dumps(passing.to_dict())

    invalid = valid.clone()
    invalid[1] = invalid[0]
    failing = audit_basis_geometry(invalid)
    assert not failing.passed
    assert failing.violations
    assert failing.condition_number is None
    json.dumps(failing.to_dict(), allow_nan=False)


def test_factor_audit_detects_inactive_and_correlated_factors() -> None:
    active_uncorrelated = _auditable_states().reshape(-1, 2, 1)
    passing = audit_factor_geometry(active_uncorrelated, max_cross_factor_correlation=0.01)
    assert passing.passed

    bad = active_uncorrelated.clone()
    bad[:, 1] = bad[:, 0]
    failing = audit_factor_geometry(bad, max_cross_factor_correlation=0.1)
    assert not failing.passed
    assert failing.max_cross_factor_correlation > 0.9

    threshold_one = audit_factor_geometry(bad, max_cross_factor_correlation=1.0)
    assert threshold_one.max_cross_factor_correlation <= 1.0
    assert threshold_one.passed

    bad[:, 1] = 0
    inactive = audit_factor_geometry(bad)
    assert inactive.inactive_factors == (1,)
    assert inactive.inactive_coordinates == ((1, 0),)


def test_factor_audit_rejects_empty_and_nonfinite_samples() -> None:
    with pytest.raises(ValueError, match="empty"):
        audit_factor_geometry(torch.empty(0, 2, 1))
    with pytest.raises(ValueError, match="non-finite"):
        audit_factor_geometry(torch.tensor([[[float("nan")], [0.0]]]))


def test_constant_factor_is_inactive_even_below_default_epsilon_scale() -> None:
    constant = torch.zeros(8, 2, 1)
    report = audit_factor_geometry(constant, min_standard_deviation=1e-8)

    assert report.factor_standard_deviations == (0.0, 0.0)
    assert report.inactive_factors == (0, 1)
    assert not report.passed


def test_factor_audit_exposes_coordinate_level_collapse() -> None:
    active = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    factors = torch.stack((active, torch.zeros_like(active)), dim=-1).unsqueeze(1)

    report = audit_factor_geometry(factors, min_standard_deviation=0.1)

    assert report.factor_standard_deviations[0] > 0.1
    assert report.coordinate_standard_deviations == ((1.0, 0.0),)
    assert report.inactive_coordinates == ((0, 1),)
    assert report.inactive_factors == (0,)
    assert not report.passed


def test_round_trip_and_combined_opf_audit() -> None:
    projection = OrthogonalFactorProjection(2, 2, 1)
    state = _auditable_states()

    round_trip = audit_state_round_trip(state, projection.analysis_basis())
    combined = audit_opf_geometry(
        projection,
        state,
        max_cross_factor_correlation=0.01,
    )

    assert round_trip.passed
    assert combined.passed
    assert combined.transpose_synthesis.passed
    json.dumps(combined.to_dict(), allow_nan=False)


def test_nonorthogonal_basis_separates_pseudoinverse_and_transpose_synthesis() -> None:
    basis = torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]]])
    state = torch.tensor([[2.0, 3.0]])

    geometry = audit_basis_geometry(basis)
    round_trip = audit_state_round_trip(state, basis)
    transpose = audit_transpose_synthesis(state, basis)

    assert not geometry.passed
    assert geometry.minimum_singular_value > 0.0
    assert geometry.condition_number is not None
    assert geometry.condition_number > 1.0
    assert geometry.max_cross_subspace_overlap == 1.0
    assert round_trip.passed
    assert round_trip.normalized_mean_square_error < 1e-10
    assert not transpose.passed
    assert transpose.normalized_mean_square_error > 1.0


def test_combined_audit_reads_learned_analysis_basis_without_qr_projection() -> None:
    basis = torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]]])
    projection = OrthogonalFactorProjection(
        2,
        2,
        1,
        learnable=True,
        initial_basis=basis,
    )
    state = _auditable_states()

    report = audit_opf_geometry(
        projection,
        state,
        min_standard_deviation=0.0,
        max_cross_factor_correlation=1.0,
    )

    assert not report.basis.passed
    assert report.round_trip.passed
    assert not report.transpose_synthesis.passed
    assert not report.passed


@pytest.mark.parametrize(
    "shape",
    [(0, 1, 1), (1, 0, 1), (1, 1, 0)],
)
def test_basis_audit_rejects_zero_dimensions(shape: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        audit_basis_geometry(torch.empty(shape))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_qr_passes_dtype_aware_default_audit(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(18)
    raw = torch.randn(2, 2, 4, generator=generator).to(dtype)
    basis = orthonormalize_basis(raw)
    state = torch.randn(16, 4, generator=generator).to(dtype)

    basis_report = audit_basis_geometry(basis)
    round_trip_report = audit_state_round_trip(state, basis)

    assert basis.dtype == dtype
    assert basis_report.tolerance == default_geometry_tolerance(dtype)
    assert basis_report.passed
    assert round_trip_report.passed


def test_audits_reject_nonfinite_derived_statistics() -> None:
    huge_factors = torch.tensor(
        [[[1e200], [1e200]], [[-1e200], [-1e200]]],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="non-finite standard deviations"):
        audit_factor_geometry(huge_factors)

    huge_basis = torch.tensor(
        [[[torch.finfo(torch.float64).max]]],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="non-finite geometry statistics"):
        audit_basis_geometry(huge_basis)

    huge_state = torch.tensor(
        [[torch.finfo(torch.float64).max]],
        dtype=torch.float64,
    )
    amplifying_basis = torch.tensor([[[2.0]]], dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite reconstruction"):
        audit_state_round_trip(huge_state, amplifying_basis)


@pytest.mark.parametrize("eps", [0.0, float("nan"), float("inf")])
def test_factor_audit_requires_finite_positive_eps(eps: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        audit_factor_geometry(torch.randn(4, 2, 1), eps=eps)
