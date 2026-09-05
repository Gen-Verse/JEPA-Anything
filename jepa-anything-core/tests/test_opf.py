from __future__ import annotations

import pytest
import torch

from jepa_anything_core import (
    OrthogonalFactorProjection,
    compose_state,
    decompose_state,
    is_orthonormal_basis,
    orthonormalize_basis,
    transpose_synthesize_state,
    validate_factorization,
)


def test_fixed_projection_round_trip_and_component_sum() -> None:
    projection = OrthogonalFactorProjection(state_dim=6, num_factors=3, factor_dim=2)
    state = torch.randn(4, 5, 6)

    factors = projection(state)
    reconstructed = projection.compose(factors)
    components = projection.project(state)

    assert factors.shape == (4, 5, 3, 2)
    assert components.shape == (4, 5, 3, 6)
    torch.testing.assert_close(reconstructed, state)
    torch.testing.assert_close(components.sum(dim=-2), state)


def test_learnable_projection_uses_unprojected_analysis_basis_and_is_differentiable() -> None:
    generator = torch.Generator().manual_seed(4)
    initial_basis = torch.randn(2, 2, 4, generator=generator)
    projection = OrthogonalFactorProjection(
        state_dim=4,
        num_factors=2,
        factor_dim=2,
        learnable=True,
        initial_basis=initial_basis,
    )
    state = torch.randn(8, 4, generator=generator)

    analysis_basis = projection.analysis_basis()
    torch.testing.assert_close(analysis_basis, initial_basis)
    assert not is_orthonormal_basis(analysis_basis)
    assert is_orthonormal_basis(projection.orthonormal_basis())
    weighted_loss = (projection(state) * torch.arange(4).reshape(2, 2)).sum()
    weighted_loss.backward()

    assert projection.raw_basis is not None
    assert projection.raw_basis.grad is not None
    assert torch.isfinite(projection.raw_basis.grad).all()


def test_qr_init_orthonormalizes_once_without_forward_reparameterization() -> None:
    generator = torch.Generator().manual_seed(17)
    initial_basis = torch.randn(2, 2, 4, generator=generator)
    projection = OrthogonalFactorProjection(
        state_dim=4,
        num_factors=2,
        factor_dim=2,
        learnable=True,
        initial_basis=initial_basis,
        orthogonality_mode="qr_init",
    )

    assert is_orthonormal_basis(projection.analysis_basis())
    with torch.no_grad():
        assert projection.raw_basis is not None
        projection.raw_basis[0, 0, 0].add_(0.25)
    assert not is_orthonormal_basis(projection.analysis_basis())
    assert projection.after_optimizer_step(1) is False


def test_qr_init_can_prepare_a_fixed_full_rank_basis() -> None:
    basis = torch.tensor(
        [
            [[2.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
        ]
    )
    projection = OrthogonalFactorProjection(
        4,
        2,
        2,
        initial_basis=basis,
        orthogonality_mode="qr_init",
    )

    assert is_orthonormal_basis(projection.analysis_basis())


def test_qr_retraction_respects_post_step_frequency() -> None:
    generator = torch.Generator().manual_seed(23)
    projection = OrthogonalFactorProjection(
        4,
        2,
        2,
        learnable=True,
        initial_basis=torch.randn(2, 2, 4, generator=generator),
        orthogonality_mode="qr_retraction",
        qr_retraction_frequency=2,
    )
    assert is_orthonormal_basis(projection.analysis_basis())

    with torch.no_grad():
        assert projection.raw_basis is not None
        projection.raw_basis[0, 0, 0].add_(0.5)
    assert not is_orthonormal_basis(projection.analysis_basis())
    assert projection.after_optimizer_step(1) is False
    assert not is_orthonormal_basis(projection.analysis_basis())
    assert projection.after_optimizer_step(2) is True
    assert is_orthonormal_basis(projection.analysis_basis())


def test_qr_modes_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="orthogonality_mode"):
        OrthogonalFactorProjection(4, 2, 2, orthogonality_mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        OrthogonalFactorProjection(4, 2, 2, qr_retraction_frequency=0)
    with pytest.raises(ValueError, match="only in qr_retraction"):
        OrthogonalFactorProjection(4, 2, 2, qr_retraction_frequency=2)
    with pytest.raises(ValueError, match="requires a learnable"):
        OrthogonalFactorProjection(4, 2, 2, orthogonality_mode="qr_retraction")

    projection = OrthogonalFactorProjection(4, 2, 2, learnable=True)
    with pytest.raises(RuntimeError, match="qr_retraction mode"):
        projection.retract_orthogonal_()
    with pytest.raises(ValueError, match="positive integer"):
        projection.after_optimizer_step(0)


def test_general_pseudoinverse_composition_and_transpose_audit_path() -> None:
    basis = torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]]])
    state = torch.tensor([[2.0, 3.0]])

    factors = decompose_state(state, basis)
    reconstructed = compose_state(factors, basis)
    transpose_reconstructed = transpose_synthesize_state(factors, basis)

    torch.testing.assert_close(reconstructed, state)
    torch.testing.assert_close(transpose_reconstructed, torch.tensor([[7.0, 5.0]]))


def test_standalone_decompose_compose_supports_double_precision() -> None:
    raw_basis = torch.randn(2, 2, 4, dtype=torch.float64)
    basis = orthonormalize_basis(raw_basis)
    state = torch.randn(3, 4, dtype=torch.float64)

    factors = decompose_state(state, basis)
    reconstructed = compose_state(factors, basis)

    assert reconstructed.dtype == torch.float64
    torch.testing.assert_close(reconstructed, state, atol=1e-10, rtol=1e-10)


def test_factorization_and_shapes_fail_fast() -> None:
    with pytest.raises(ValueError, match=r"K \* r != d|num_factors \* factor_dim"):
        validate_factorization(7, 2, 3)
    with pytest.raises(ValueError, match="cannot be inferred"):
        OrthogonalFactorProjection(state_dim=5, num_factors=2)
    with pytest.raises(TypeError, match="must be integers"):
        validate_factorization(4.0, 2, 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be integers"):
        validate_factorization(4, True, 4)  # type: ignore[arg-type]

    basis = torch.eye(4).reshape(2, 2, 4)
    with pytest.raises(ValueError, match="state must end"):
        decompose_state(torch.randn(3, 5), basis)
    with pytest.raises(ValueError, match="factors must end"):
        compose_state(torch.randn(3, 4, 1), basis)


def test_fixed_basis_rejects_nonorthogonal_geometry() -> None:
    invalid_basis = torch.ones(2, 2, 4)
    with pytest.raises(ValueError, match="orthonormal"):
        OrthogonalFactorProjection(4, 2, 2, initial_basis=invalid_basis)


def test_fixed_basis_is_detached_from_callers_autograd_graph() -> None:
    source = torch.eye(4, requires_grad=True)
    basis = source.reshape(2, 2, 4)
    projection = OrthogonalFactorProjection(4, 2, 2, initial_basis=basis)
    state = torch.randn(3, 4, requires_grad=True)

    projection(state).sum().backward()

    assert source.grad is None
    assert projection.fixed_basis is not None
    assert not projection.fixed_basis.requires_grad


def test_float64_rank_check_preserves_small_nonzero_singular_values() -> None:
    basis = torch.diag(torch.tensor([1.0, 1.0e-8], dtype=torch.float64)).reshape(1, 2, 2)

    projection = OrthogonalFactorProjection(
        2,
        1,
        2,
        learnable=True,
        initial_basis=basis,
    )

    assert projection.raw_basis is not None
    assert projection.raw_basis.dtype == torch.float64


def test_initial_basis_rejects_complex_values_instead_of_dropping_imaginary_part() -> None:
    basis = torch.eye(2, dtype=torch.complex64).reshape(1, 2, 2)

    with pytest.raises(TypeError, match="real-valued"):
        OrthogonalFactorProjection(2, 1, 2, initial_basis=basis)
