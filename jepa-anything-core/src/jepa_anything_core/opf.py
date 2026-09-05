"""Orthogonal predictive-factor (OPF) state geometry.

The module deliberately gives factors no semantic names.  It only constructs a
set of orthogonal predictive coordinates that downstream experiments may test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn


OrthogonalityMode = Literal["soft_gram", "qr_init", "qr_retraction"]
ORTHOGONALITY_MODES = frozenset({"soft_gram", "qr_init", "qr_retraction"})


def default_geometry_tolerance(dtype: torch.dtype) -> float:
    """Return the default absolute geometry tolerance for a floating dtype.

    The values account for low-precision basis geometry and state round trips:
    ``1e-10`` for float64, ``1e-5`` for float32, ``1e-2`` for float16, and
    ``5e-2`` for bfloat16.  Unsupported floating dtypes use ``1e-5``.
    """

    probe = torch.empty((), dtype=dtype)
    if not probe.is_floating_point():
        raise TypeError("geometry dtype must be floating point")
    if dtype == torch.float64:
        return 1e-10
    if dtype == torch.float16:
        return 1e-2
    if dtype == torch.bfloat16:
        return 5e-2
    return 1e-5


def validate_factorization(state_dim: int, num_factors: int, factor_dim: int) -> None:
    """Validate the complete-state contract ``state_dim = num_factors * factor_dim``.

    Args:
        state_dim: Width of the complete latent state.
        num_factors: Number of predictive coordinate groups, ``K``.
        factor_dim: Width of each group, ``r``.

    Raises:
        ValueError: If a dimension is non-positive or ``K * r != d``.
    """

    dimensions = {
        "state_dim": state_dim,
        "num_factors": num_factors,
        "factor_dim": factor_dim,
    }
    invalid_types = [
        name
        for name, value in dimensions.items()
        if isinstance(value, bool) or not isinstance(value, int)
    ]
    if invalid_types:
        raise TypeError(f"dimensions must be integers; invalid: {', '.join(invalid_types)}")
    invalid = [name for name, value in dimensions.items() if value <= 0]
    if invalid:
        raise ValueError(f"dimensions must be positive; invalid: {', '.join(invalid)}")
    if num_factors * factor_dim != state_dim:
        raise ValueError(
            "complete OPF state requires num_factors * factor_dim == state_dim "
            f"({num_factors} * {factor_dim} != {state_dim})"
        )


def _validate_basis_shape(basis: Tensor) -> tuple[int, int, int]:
    if basis.ndim != 3:
        raise ValueError(f"basis must have shape (K, r, d), got {tuple(basis.shape)}")
    num_factors, factor_dim, state_dim = basis.shape
    validate_factorization(state_dim, num_factors, factor_dim)
    if not basis.is_floating_point():
        raise TypeError("basis must be floating point")
    return num_factors, factor_dim, state_dim


def _validate_same_device_dtype(value: Tensor, basis: Tensor, name: str) -> None:
    if value.device != basis.device:
        raise ValueError(f"{name} and basis must be on the same device")
    if value.dtype != basis.dtype:
        raise ValueError(f"{name} and basis must have the same dtype")


def orthonormalize_basis(raw_basis: Tensor) -> Tensor:
    """Map a full-rank raw basis to an orthonormal basis with differentiable QR.

    Args:
        raw_basis: Tensor of shape ``(K, r, d)`` satisfying ``K * r = d``.

    Returns:
        A tensor of the same shape whose flattened rows are orthonormal.

    Raises:
        ValueError: If ``raw_basis`` is rank deficient.

    Notes:
        QR is evaluated in float32 for CPU half-precision inputs and cast back.
        This utility is intended for initialization and explicit diagnostics.
        Learnable OPF modules optimize their analysis rows directly and do not
        call it in the training forward path.
    """

    _, _, state_dim = _validate_basis_shape(raw_basis)
    if not torch.isfinite(raw_basis).all():
        raise ValueError("raw_basis contains non-finite values")
    flat = raw_basis.reshape(state_dim, state_dim)
    work = flat.float() if flat.dtype in (torch.float16, torch.bfloat16) else flat
    if torch.linalg.matrix_rank(work.detach()).item() < state_dim:
        raise ValueError("raw_basis must be full rank")
    q, upper = torch.linalg.qr(work.transpose(0, 1), mode="reduced")
    # Canonicalize QR's column-sign ambiguity so repeated retractions do not
    # arbitrarily flip predictive-coordinate signs. Full rank guarantees a
    # nonzero diagonal in exact arithmetic; treat a numerical zero as positive.
    diagonal = torch.diagonal(upper)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q = q * signs.unsqueeze(0)
    return q.transpose(0, 1).reshape_as(raw_basis).to(dtype=raw_basis.dtype)


def is_orthonormal_basis(basis: Tensor, *, atol: float | None = None) -> bool:
    """Return whether flattened basis rows form an orthonormal complete basis.

    If ``atol`` is omitted, :func:`default_geometry_tolerance` selects a
    dtype-aware threshold.  Low-precision products are accumulated in float32.
    """

    _, _, state_dim = _validate_basis_shape(basis)
    if not torch.isfinite(basis).all():
        return False
    tolerance = default_geometry_tolerance(basis.dtype) if atol is None else atol
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("atol must be finite and non-negative")
    flat = basis.reshape(state_dim, state_dim)
    work = flat.float() if flat.dtype in (torch.float16, torch.bfloat16) else flat
    identity = torch.eye(state_dim, device=basis.device, dtype=work.dtype)
    return bool(
        torch.allclose(work @ work.transpose(0, 1), identity, atol=tolerance, rtol=0.0)
    )


def decompose_state(state: Tensor, basis: Tensor) -> Tensor:
    """Analyze complete states into predictive factor coordinates.

    Args:
        state: Tensor with trailing shape ``(d,)``.
        basis: Analysis rows with shape ``(K, r, d)``.

    Returns:
        Tensor with trailing shape ``(K, r)``.
    """

    _, _, state_dim = _validate_basis_shape(basis)
    if state.ndim < 1 or state.shape[-1] != state_dim:
        raise ValueError(f"state must end in dimension {state_dim}, got {tuple(state.shape)}")
    _validate_same_device_dtype(state, basis, "state")
    return torch.einsum("...d,krd->...kr", state, basis)


def compose_state(factors: Tensor, basis: Tensor) -> Tensor:
    """Synthesize complete states with the Moore--Penrose inverse.

    Args:
        factors: Tensor with trailing shape ``(K, r)``.
        basis: Analysis rows with shape ``(K, r, d)``.  Flattening the
            leading dimensions gives the analysis map ``P.T``.

    Returns:
        Tensor with trailing shape ``(d,)``.
    """

    num_factors, factor_dim, state_dim = _validate_basis_shape(basis)
    if factors.ndim < 2 or factors.shape[-2:] != (num_factors, factor_dim):
        raise ValueError(
            f"factors must end in ({num_factors}, {factor_dim}), got {tuple(factors.shape)}"
        )
    _validate_same_device_dtype(factors, basis, "factors")
    flat = basis.reshape(state_dim, state_dim)
    work = flat.float() if flat.dtype in (torch.float16, torch.bfloat16) else flat
    synthesis = torch.linalg.pinv(work).to(dtype=basis.dtype)
    return torch.einsum("...a,da->...d", factors.flatten(start_dim=-2), synthesis)


def transpose_synthesize_state(factors: Tensor, basis: Tensor) -> Tensor:
    """Synthesize with ``P`` rather than ``(P.T)^dagger`` for geometry audits.

    This is exact only when the flattened analysis rows are orthonormal.  It is
    exposed separately because transpose-synthesis error is an informative
    audit of learned projector geometry; operational composition should use
    :func:`compose_state`.
    """

    num_factors, factor_dim, _ = _validate_basis_shape(basis)
    if factors.ndim < 2 or factors.shape[-2:] != (num_factors, factor_dim):
        raise ValueError(
            f"factors must end in ({num_factors}, {factor_dim}), got {tuple(factors.shape)}"
        )
    _validate_same_device_dtype(factors, basis, "factors")
    return torch.einsum("...kr,krd->...d", factors, basis)


def project_state(state: Tensor, basis: Tensor) -> Tensor:
    """Return each factor's transpose-synthesized component in state space.

    The result has trailing shape ``(K, d)``.  Its sum is exactly the original
    state only when the flattened analysis rows are orthonormal; use
    :func:`compose_state` for general operational synthesis.
    """

    factors = decompose_state(state, basis)
    return torch.einsum("...kr,krd->...kd", factors, basis)


class OrthogonalFactorProjection(nn.Module):
    """Complete orthogonal factorization of a latent state.

    Args:
        state_dim: Complete state width, ``d``.
        num_factors: Number of predictive coordinate groups, ``K``.
        factor_dim: Coordinates per group, ``r``.  If omitted, it is inferred
            from ``state_dim / num_factors``.
        learnable: If true, store the analysis rows directly as trainable
            parameters. If false, store a fixed already-orthonormal basis.
        initial_basis: Optional tensor with shape ``(K, r, d)``.  A learnable
            basis may be any full-rank matrix. A fixed basis must be orthonormal
            unless ``orthogonality_mode='qr_init'`` initializes it by QR.
        orthogonality_mode: ``soft_gram`` leaves learnable analysis rows
            unconstrained for a projector-Gram objective; ``qr_init`` applies
            QR once before storing them; ``qr_retraction`` additionally permits
            explicit post-optimizer-step retraction. No mode applies QR inside
            :meth:`forward`.
        qr_retraction_frequency: Positive optimizer-step interval used by
            :meth:`after_optimizer_step` in ``qr_retraction`` mode.

    This class enforces ``K * r = d`` at construction.  A learnable analysis
    basis is not guaranteed to be orthogonal during optimization; use the Gram
    loss and geometry audits to verify it.  The class guarantees no semantic
    interpretation: factor indices remain anonymous predictive coordinates
    until experiments establish a meaning.
    """

    state_dim: int
    num_factors: int
    factor_dim: int
    learnable: bool
    orthogonality_mode: OrthogonalityMode
    qr_retraction_frequency: int
    raw_basis: nn.Parameter | None
    fixed_basis: Tensor | None

    def __init__(
        self,
        state_dim: int,
        num_factors: int,
        factor_dim: int | None = None,
        *,
        learnable: bool = False,
        initial_basis: Tensor | Sequence[Sequence[Sequence[float]]] | None = None,
        orthogonality_mode: OrthogonalityMode = "soft_gram",
        qr_retraction_frequency: int = 1,
    ) -> None:
        super().__init__()
        if orthogonality_mode not in ORTHOGONALITY_MODES:
            raise ValueError(
                "orthogonality_mode must be one of soft_gram, qr_init, or qr_retraction"
            )
        if (
            isinstance(qr_retraction_frequency, bool)
            or not isinstance(qr_retraction_frequency, int)
            or qr_retraction_frequency <= 0
        ):
            raise ValueError("qr_retraction_frequency must be a positive integer")
        if orthogonality_mode == "qr_retraction" and not learnable:
            raise ValueError("qr_retraction requires a learnable projection")
        if orthogonality_mode != "qr_retraction" and qr_retraction_frequency != 1:
            raise ValueError(
                "qr_retraction_frequency is configurable only in qr_retraction mode"
            )
        if factor_dim is None:
            if num_factors <= 0 or state_dim % num_factors != 0:
                raise ValueError("factor_dim cannot be inferred unless state_dim is divisible by K")
            factor_dim = state_dim // num_factors
        validate_factorization(state_dim, num_factors, factor_dim)

        self.state_dim = state_dim
        self.num_factors = num_factors
        self.factor_dim = factor_dim
        self.learnable = learnable
        self.orthogonality_mode = orthogonality_mode
        self.qr_retraction_frequency = qr_retraction_frequency

        if initial_basis is None:
            basis = torch.eye(state_dim).reshape(num_factors, factor_dim, state_dim)
        else:
            basis = torch.as_tensor(initial_basis)
            if tuple(basis.shape) != (num_factors, factor_dim, state_dim):
                raise ValueError(
                    "initial_basis must have shape "
                    f"({num_factors}, {factor_dim}, {state_dim}), got {tuple(basis.shape)}"
                )
            if basis.is_complex():
                raise TypeError("initial_basis must be real-valued")
            if not basis.is_floating_point():
                basis = basis.float()

        if orthogonality_mode in {"qr_init", "qr_retraction"}:
            basis = orthonormalize_basis(basis)

        if learnable:
            # Start from a complete analysis map.  Later optimization may move
            # through ill-conditioned states; pseudoinverse synthesis and audits
            # make that state explicit rather than silently rewriting it by QR.
            _, _, checked_state_dim = _validate_basis_shape(basis)
            work = basis.reshape(checked_state_dim, checked_state_dim)
            if not torch.isfinite(work).all():
                raise ValueError("initial_basis contains non-finite values")
            rank_input = work.detach()
            if rank_input.dtype in (torch.float16, torch.bfloat16):
                rank_input = rank_input.float()
            if torch.linalg.matrix_rank(rank_input).item() < checked_state_dim:
                raise ValueError("a learnable initial_basis must be full rank")
            self.raw_basis = nn.Parameter(basis.detach().clone())
            self.register_buffer("fixed_basis", None)
        else:
            if not is_orthonormal_basis(basis):
                raise ValueError("a fixed initial_basis must be orthonormal")
            self.register_parameter("raw_basis", None)
            self.register_buffer("fixed_basis", basis.detach().clone())

    def analysis_basis(self) -> Tensor:
        """Return the current analysis rows with shape ``(K, r, d)``.

        For a learnable projection this is the parameter optimized by prediction,
        projector-Gram, and factor-activity losses.  No QR projection is applied.
        """

        if self.learnable:
            if self.raw_basis is None:  # pragma: no cover - protects corrupted module state
                raise RuntimeError("learnable projection has no raw_basis")
            return self.raw_basis
        if self.fixed_basis is None:  # pragma: no cover - protects corrupted module state
            raise RuntimeError("fixed projection has no fixed_basis")
        return self.fixed_basis

    def current_basis(self) -> Tensor:
        """Alias for :meth:`analysis_basis`."""

        return self.analysis_basis()

    def orthonormal_basis(self) -> Tensor:
        """Return an explicitly QR-orthonormalized diagnostic copy.

        This compatibility helper is not used by :meth:`forward`,
        :meth:`decompose`, :meth:`compose`, or the geometry audit.  Calling it on
        a learnable projection changes the represented analysis map and should
        therefore be reserved for initialization or explicit diagnostics.
        """

        basis = self.analysis_basis()
        if self.learnable:
            return orthonormalize_basis(basis)
        return basis

    def retract_orthogonal_(self) -> "OrthogonalFactorProjection":
        """QR-retract learnable rows in place outside the forward path.

        Call this only after an optimizer step. Optimizer state is deliberately
        not rewritten; callers remain responsible for choosing an optimizer and
        monitoring whether its state is compatible with repeated retraction.
        """

        if self.orthogonality_mode != "qr_retraction":
            raise RuntimeError("orthogonal retraction requires qr_retraction mode")
        if self.raw_basis is None:
            raise RuntimeError("orthogonal retraction requires learnable analysis rows")
        with torch.no_grad():
            self.raw_basis.copy_(orthonormalize_basis(self.raw_basis))
        return self

    def after_optimizer_step(self, step: int) -> bool:
        """Apply a scheduled QR retraction and return whether it ran.

        ``step`` is a positive, one-based optimizer-step count. Other modes
        return ``False`` without modifying the analysis rows.
        """

        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("step must be a positive integer")
        if self.orthogonality_mode != "qr_retraction":
            return False
        if step % self.qr_retraction_frequency != 0:
            return False
        self.retract_orthogonal_()
        return True

    def decompose(self, state: Tensor) -> Tensor:
        """Decompose ``(..., d)`` states into ``(..., K, r)`` coordinates."""

        return decompose_state(state, self.analysis_basis())

    def compose(self, factors: Tensor) -> Tensor:
        """Compose ``(..., K, r)`` coordinates into ``(..., d)`` states."""

        return compose_state(factors, self.analysis_basis())

    def transpose_synthesize(self, factors: Tensor) -> Tensor:
        """Apply transpose synthesis for geometry auditing."""

        return transpose_synthesize_state(factors, self.analysis_basis())

    def project(self, state: Tensor) -> Tensor:
        """Project states into ``K`` components with trailing shape ``(K, d)``."""

        return project_state(state, self.analysis_basis())

    def projectors(self) -> Tensor:
        """Return ``P_k P_k.T`` operators with shape ``(K, d, d)``.

        They are idempotent orthogonal projectors only when the corresponding
        within-block Gram constraints are satisfied.
        """

        basis = self.analysis_basis()
        return torch.einsum("krd,kre->kde", basis, basis)

    def forward(self, state: Tensor) -> Tensor:
        """Alias for :meth:`decompose`."""

        return self.decompose(state)

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, num_factors={self.num_factors}, "
            f"factor_dim={self.factor_dim}, learnable={self.learnable}, "
            f"orthogonality_mode={self.orthogonality_mode!r}, "
            f"qr_retraction_frequency={self.qr_retraction_frequency}"
        )
