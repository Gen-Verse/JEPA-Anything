"""Deterministic geometry audits for OPF representations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from .losses import (
    factor_coordinate_standard_deviation,
    factor_cross_correlation,
    factor_standard_deviation,
)
from .opf import (
    OrthogonalFactorProjection,
    compose_state,
    decompose_state,
    default_geometry_tolerance,
    transpose_synthesize_state,
)


@dataclass(frozen=True)
class BasisGeometryReport:
    """Numerical checks for basis orthogonality and state completeness."""

    state_dim: int
    num_factors: int
    factor_dim: int
    rank: int
    minimum_singular_value: float
    condition_number: float | None
    max_cross_subspace_overlap: float
    max_orthogonality_error: float
    max_completeness_error: float
    tolerance: float
    passed: bool
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class FactorGeometryReport:
    """Activity and cross-factor correlation diagnostics."""

    num_samples: int
    factor_standard_deviations: tuple[float, ...]
    coordinate_standard_deviations: tuple[tuple[float, ...], ...]
    inactive_factors: tuple[int, ...]
    inactive_coordinates: tuple[tuple[int, int], ...]
    max_cross_factor_correlation: float
    rms_cross_factor_correlation: float
    min_standard_deviation: float
    max_allowed_correlation: float
    passed: bool
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class RoundTripReport:
    """State decompose/compose reconstruction diagnostics."""

    max_absolute_error: float
    root_mean_square_error: float
    normalized_mean_square_error: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, float | bool]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class TransposeSynthesisReport:
    """Direct ``P(P.T z)`` geometry audit, independent of predictor error."""

    max_absolute_error: float
    root_mean_square_error: float
    normalized_mean_square_error: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, float | bool]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class OPFGeometryReport:
    """Combined deterministic OPF audit."""

    basis: BasisGeometryReport
    factors: FactorGeometryReport
    round_trip: RoundTripReport
    transpose_synthesis: TransposeSynthesisReport
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a nested JSON-serializable dictionary."""

        return asdict(self)


def audit_basis_geometry(
    basis: Tensor,
    *,
    tolerance: float | None = None,
) -> BasisGeometryReport:
    """Audit ``K * r = d``, numerical rank, orthogonality, and completeness.

    Unlike state operators, this audit accepts incomplete or malformed geometry
    (provided it has three positive dimensions) so it can return an actionable
    report.  If ``tolerance`` is omitted, a dtype-aware default is recorded in
    the report.  Cross-subspace overlap is the maximum Frobenius norm of
    ``P_i.T P_j`` over ``i < j``, matching the cross-projector block.
    """

    if basis.ndim != 3:
        raise ValueError(f"basis must have shape (K, r, d), got {tuple(basis.shape)}")
    if not basis.is_floating_point():
        raise TypeError("basis must be floating point")
    if not torch.isfinite(basis).all():
        raise ValueError("basis contains non-finite values")
    num_factors, factor_dim, state_dim = basis.shape
    if num_factors <= 0 or factor_dim <= 0 or state_dim <= 0:
        raise ValueError("basis dimensions K, r, and d must be positive")
    effective_tolerance = (
        default_geometry_tolerance(basis.dtype) if tolerance is None else tolerance
    )
    if not math.isfinite(effective_tolerance) or effective_tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    flat = basis.reshape(num_factors * factor_dim, state_dim)
    # Audits are reporting paths, not training paths.  Use CPU float64 so the
    # diagnostics have stable precision and also work for backends such as MPS,
    # which do not implement float64 linear algebra.
    work = flat.detach().to(device="cpu", dtype=torch.float64)
    rank = int(torch.linalg.matrix_rank(work).item())
    singular_values = torch.linalg.svdvals(work)
    minimum_singular_value = float(singular_values.min().item())
    if rank < min(work.shape) or minimum_singular_value == 0.0:
        # ``None`` is the strict-JSON representation of an undefined/infinite
        # condition number for a rank-deficient analysis map.
        condition_number = None
    else:
        condition_number = float((singular_values.max() / singular_values.min()).item())

    block_rows = work.reshape(num_factors, factor_dim, state_dim)
    cross_overlaps: list[Tensor] = []
    for first in range(num_factors):
        for second in range(first + 1, num_factors):
            cross_gram = block_rows[first] @ block_rows[second].T
            cross_overlaps.append(torch.linalg.matrix_norm(cross_gram, ord="fro"))
    max_cross_subspace_overlap = (
        float(torch.stack(cross_overlaps).max().item()) if cross_overlaps else 0.0
    )

    row_identity = torch.eye(flat.shape[0], device=work.device, dtype=work.dtype)
    column_identity = torch.eye(state_dim, device=work.device, dtype=work.dtype)
    orthogonality_error = float((work @ work.T - row_identity).abs().max().item())
    completeness_error = float((work.T @ work - column_identity).abs().max().item())
    finite_geometry = (
        math.isfinite(minimum_singular_value)
        and math.isfinite(max_cross_subspace_overlap)
        and math.isfinite(orthogonality_error)
        and math.isfinite(completeness_error)
    )
    if not finite_geometry:
        raise ValueError("basis audit produced non-finite geometry statistics")

    violations: list[str] = []
    if num_factors * factor_dim != state_dim:
        violations.append("Kr != d: factors do not form a complete state")
    if rank != state_dim:
        violations.append(f"basis rank is {rank}, expected {state_dim}")
    if orthogonality_error > effective_tolerance:
        violations.append("basis rows are not orthonormal within tolerance")
    if completeness_error > effective_tolerance:
        violations.append("factor projectors do not sum to identity within tolerance")

    return BasisGeometryReport(
        state_dim=state_dim,
        num_factors=num_factors,
        factor_dim=factor_dim,
        rank=rank,
        minimum_singular_value=minimum_singular_value,
        condition_number=condition_number,
        max_cross_subspace_overlap=max_cross_subspace_overlap,
        max_orthogonality_error=orthogonality_error,
        max_completeness_error=completeness_error,
        tolerance=effective_tolerance,
        passed=not violations,
        violations=tuple(violations),
    )


def audit_factor_geometry(
    factors: Tensor,
    *,
    min_standard_deviation: float = 0.1,
    max_cross_factor_correlation: float = 0.2,
    eps: float = 1e-6,
) -> FactorGeometryReport:
    """Audit per-coordinate factor activity and cross-factor correlation.

    The report checks prediction geometry only.  Passing it is not evidence that
    any coordinate corresponds to a named physical or semantic concept.
    """

    if not math.isfinite(min_standard_deviation) or min_standard_deviation < 0:
        raise ValueError("min_standard_deviation must be finite and non-negative")
    if (
        not math.isfinite(max_cross_factor_correlation)
        or not 0 <= max_cross_factor_correlation <= 1
    ):
        raise ValueError("max_cross_factor_correlation must be finite and lie in [0, 1]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if factors.ndim < 3:
        raise ValueError(f"factors must have shape (..., K, r), got {tuple(factors.shape)}")
    if factors.shape[-2] <= 0 or factors.shape[-1] <= 0:
        raise ValueError("factor dimensions K and r must be positive")
    if factors.numel() == 0:
        raise ValueError("cannot audit an empty factor sample")
    if not torch.isfinite(factors).all():
        raise ValueError("factors contain non-finite values")

    samples = factors.reshape(-1, factors.shape[-2], factors.shape[-1])
    coordinate_standard_deviations = factor_coordinate_standard_deviation(samples).detach()
    standard_deviations = factor_standard_deviation(samples).detach()
    if (
        not torch.isfinite(coordinate_standard_deviations).all()
        or not torch.isfinite(standard_deviations).all()
    ):
        raise ValueError("factor audit produced non-finite standard deviations")
    inactive_coordinates = tuple(
        (factor_index, coordinate_index)
        for factor_index in range(samples.shape[1])
        for coordinate_index in range(samples.shape[2])
        if coordinate_standard_deviations[factor_index, coordinate_index].item()
        < min_standard_deviation
    )
    inactive = tuple(sorted({factor_index for factor_index, _ in inactive_coordinates}))

    num_factors = samples.shape[1]
    if num_factors < 2:
        max_correlation = 0.0
        rms_correlation = 0.0
    else:
        correlations = factor_cross_correlation(samples, eps=eps).detach()
        if not torch.isfinite(correlations).all():
            raise ValueError("factor audit produced non-finite correlations")
        mask = ~torch.eye(num_factors, device=samples.device, dtype=torch.bool)
        off_diagonal = correlations[mask]
        max_correlation = float(off_diagonal.abs().max().item())
        rms_correlation = float(torch.sqrt(off_diagonal.square().mean()).item())
        if not math.isfinite(max_correlation) or not math.isfinite(rms_correlation):
            raise ValueError("factor audit produced non-finite correlation summaries")

    violations: list[str] = []
    if inactive_coordinates:
        violations.append(f"inactive factor coordinates: {inactive_coordinates}")
    if max_correlation > max_cross_factor_correlation:
        violations.append("cross-factor correlation exceeds configured threshold")

    return FactorGeometryReport(
        num_samples=samples.shape[0],
        factor_standard_deviations=tuple(float(value) for value in standard_deviations.tolist()),
        coordinate_standard_deviations=tuple(
            tuple(float(value) for value in row)
            for row in coordinate_standard_deviations.tolist()
        ),
        inactive_factors=inactive,
        inactive_coordinates=inactive_coordinates,
        max_cross_factor_correlation=max_correlation,
        rms_cross_factor_correlation=rms_correlation,
        min_standard_deviation=min_standard_deviation,
        max_allowed_correlation=max_cross_factor_correlation,
        passed=not violations,
        violations=tuple(violations),
    )


def _reconstruction_statistics(
    reference: Tensor,
    reconstructed: Tensor,
) -> tuple[float, float, float]:
    error = (reconstructed - reference).detach().to(device="cpu", dtype=torch.float64)
    reference_work = reference.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(reconstructed).all() or not torch.isfinite(error).all():
        raise ValueError("non-finite reconstruction statistics")
    max_error = float(error.abs().max().item())
    mean_square_error = error.square().mean()
    rms_error = float(torch.sqrt(mean_square_error).item())
    reference_energy = reference_work.square().mean()
    if reference_energy.item() == 0.0:
        normalized_mse = 0.0 if mean_square_error.item() == 0.0 else float("inf")
    else:
        normalized_mse = float((mean_square_error / reference_energy).item())
    return max_error, rms_error, normalized_mse


def audit_state_round_trip(
    state: Tensor,
    basis: Tensor,
    *,
    tolerance: float | None = None,
) -> RoundTripReport:
    """Audit reconstruction after deterministic state decomposition/synthesis."""

    effective_tolerance = (
        default_geometry_tolerance(state.dtype) if tolerance is None else tolerance
    )
    if not math.isfinite(effective_tolerance) or effective_tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not torch.isfinite(state).all():
        raise ValueError("state contains non-finite values")
    if not torch.isfinite(basis).all():
        raise ValueError("basis contains non-finite values")
    if state.numel() == 0:
        raise ValueError("cannot audit an empty state sample")
    reconstructed = compose_state(decompose_state(state, basis), basis)
    max_error, rms_error, normalized_mse = _reconstruction_statistics(state, reconstructed)
    if (
        not math.isfinite(max_error)
        or not math.isfinite(rms_error)
        or not math.isfinite(normalized_mse)
    ):
        raise ValueError("round-trip audit produced non-finite error summaries")
    return RoundTripReport(
        max_error,
        rms_error,
        normalized_mse,
        effective_tolerance,
        max_error <= effective_tolerance,
    )


def audit_transpose_synthesis(
    state: Tensor,
    basis: Tensor,
    *,
    tolerance: float | None = None,
) -> TransposeSynthesisReport:
    """Audit direct ``P(P.T z)`` synthesis and report normalized MSE."""

    effective_tolerance = (
        default_geometry_tolerance(state.dtype) if tolerance is None else tolerance
    )
    if not math.isfinite(effective_tolerance) or effective_tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not torch.isfinite(state).all() or not torch.isfinite(basis).all():
        raise ValueError("state and basis must contain only finite values")
    if state.numel() == 0:
        raise ValueError("cannot audit an empty state sample")
    factors = decompose_state(state, basis)
    reconstructed = transpose_synthesize_state(factors, basis)
    max_error, rms_error, normalized_mse = _reconstruction_statistics(state, reconstructed)
    if (
        not math.isfinite(max_error)
        or not math.isfinite(rms_error)
        or not math.isfinite(normalized_mse)
    ):
        raise ValueError("transpose-synthesis audit produced non-finite error summaries")
    return TransposeSynthesisReport(
        max_absolute_error=max_error,
        root_mean_square_error=rms_error,
        normalized_mean_square_error=normalized_mse,
        tolerance=effective_tolerance,
        passed=normalized_mse <= effective_tolerance,
    )


def audit_opf_geometry(
    projection: OrthogonalFactorProjection,
    state: Tensor,
    *,
    basis_tolerance: float | None = None,
    reconstruction_tolerance: float | None = None,
    transpose_nmse_tolerance: float | None = None,
    min_standard_deviation: float = 0.1,
    max_cross_factor_correlation: float = 0.2,
) -> OPFGeometryReport:
    """Run basis, activity/correlation, and round-trip audits together."""

    basis = projection.analysis_basis()
    factors = projection.decompose(state)
    basis_report = audit_basis_geometry(basis, tolerance=basis_tolerance)
    factor_report = audit_factor_geometry(
        factors,
        min_standard_deviation=min_standard_deviation,
        max_cross_factor_correlation=max_cross_factor_correlation,
    )
    round_trip_report = audit_state_round_trip(
        state,
        basis,
        tolerance=reconstruction_tolerance,
    )
    transpose_report = audit_transpose_synthesis(
        state,
        basis,
        tolerance=transpose_nmse_tolerance,
    )
    return OPFGeometryReport(
        basis=basis_report,
        factors=factor_report,
        round_trip=round_trip_report,
        transpose_synthesis=transpose_report,
        passed=(
            basis_report.passed
            and factor_report.passed
            and round_trip_report.passed
            and transpose_report.passed
        ),
    )
