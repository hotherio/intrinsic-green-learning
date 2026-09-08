# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportCallIssue=false, reportArgumentType=false
"""Spectral Green's-function kernel — peer of :class:`igl.GreenKernel`.

For each latent dimension ``j``, ``SpectralKernel`` evaluates a 1-D
:class:`SpectralBasis`. The multi-dim Green's function is the
*separable product*

::

    G(z, μ_r) = Π_j Σ_{k ∉ null} φ_k(z_j) · φ_k(μ_{r,j}) / max(λ_k, ε)

plus extra null-space columns from an optional
:class:`igl.types.NullSpaceBasis`. The full design matrix has shape
``[N, n_anchors + null_space.n_columns]``.

Null modes (``basis.null_indices``, ``λ = 0``) are **excluded** from the
sum: a Green's function does not reach the operator's kernel, and flooring
``λ₀`` at ``ε`` would instead hand the constant a weight of ``1/ε`` that
swamps every other mode (rank-1 design matrices). Their contribution comes
through the augmented null-space columns, which the lstsq solve fits
without Tikhonov shrinkage. Every factor is then zero-mean in its
coordinate, so a function of a strict subset of the coordinates (``h(z₁)``
when ``d = 2``) also needs those columns.

Two kinds of basis change the computation:

- a **joint** basis (``is_joint = True``, e.g.
  :class:`igl.spectral.LearnedLaplacianBasis`) is a function of the whole
  latent vector: the kernel is ``Σ_k φ_k(z) φ_k(μ_r) / λ_k`` with no
  per-dimension product, and the basis must be passed alone;
- an **index** basis (``is_index_basis = True``, e.g.
  :class:`igl.spectral.GraphLaplacianBasis`) takes node indices: it needs
  fixed integer ``anchors`` (``learnable_anchors=False``) and is exempt
  from the domain map.

The closed-form bases live on a bounded or half-bounded domain
(``basis.domain``) while the encoder's latent is unbounded. With
``domain_map="auto"`` (default) each coordinate is squashed into its
basis's domain before evaluation — sigmoid onto a bounded interval,
softplus onto a half-line, identity on ℝ — anchors included, so the
anchors keep living in the unbounded pre-map space and stay learnable.
"""

import math
import warnings
from collections.abc import Sequence
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from igl.exceptions import IGLConfigError
from igl.types import DomainMap, DomainMapLike, NullSpaceBasis, SpectralBasis

_EXPECTED_Z_NDIM = 2
_UNBOUNDED: tuple[float, float] = (-math.inf, math.inf)


def _domain_of(basis: nn.Module) -> tuple[float, float]:
    lo, hi = getattr(basis, "domain", _UNBOUNDED)
    return float(lo), float(hi)


def _map_to_domain(z: torch.Tensor, domain: tuple[float, float]) -> torch.Tensor:
    """Squash an unbounded coordinate into ``domain`` (identity on ℝ)."""
    lo, hi = domain
    if math.isfinite(lo) and math.isfinite(hi):
        return lo + (hi - lo) * torch.sigmoid(z)
    if math.isfinite(lo):
        return lo + F.softplus(z)
    if math.isfinite(hi):
        return hi - F.softplus(z)
    return z


def _kept_modes(bases: Sequence[nn.Module], *, null_space_given: bool) -> list[list[int]]:
    """Per basis, the mode indices the kernel expands: everything but ``null_indices``."""
    keeps: list[list[int]] = []
    excluded_any = False
    for b in bases:
        null = {int(i) for i in getattr(b, "null_indices", ())}
        keep = [k for k in range(int(b.n_modes)) if k not in null]
        if not keep:
            raise IGLConfigError("basis has no non-null modes; nothing for the kernel to expand")
        excluded_any = excluded_any or len(keep) < int(b.n_modes)
        keeps.append(keep)
    if excluded_any and not null_space_given:
        warnings.warn(
            "SpectralKernel: a basis has null modes (e.g. the constant) that the kernel excludes; "
            "pass null_space= (e.g. ConstantNullSpace()) so the design matrix can reach them",
            RuntimeWarning,
            stacklevel=3,
        )
    return keeps


def _resolve_anchors(
    anchors: torch.Tensor | None,
    *,
    latent_dim: int,
    n_anchors: int,
    anchor_init_std: float,
    has_index_basis: bool,
    learnable: bool,
) -> torch.Tensor:
    """Explicit anchors (mandatory, fixed, for an index basis) or a Gaussian draw in the pre-map space."""
    if has_index_basis and (anchors is None or learnable):
        raise IGLConfigError(
            "an index basis takes node ids: pass explicit integer anchors and set learnable_anchors=False",
        )
    if anchors is None:
        return torch.randn(n_anchors, latent_dim) * anchor_init_std
    anchor_tensor = torch.as_tensor(anchors, dtype=torch.float32).clone()
    if anchor_tensor.dim() != _EXPECTED_Z_NDIM or anchor_tensor.shape[1] != latent_dim:
        raise IGLConfigError(f"anchors must be [R, {latent_dim}]; got {tuple(anchor_tensor.shape)}")
    return anchor_tensor


class SpectralKernel(nn.Module):
    """Spectral Green's-function kernel.

    Args:
        latent_dim: Latent dimensionality ``d``.
        bases: One :class:`SpectralBasis` (used uniformly across all
            ``latent_dim`` dimensions, or once on the whole latent when it
            is a joint basis) or a sequence of length ``latent_dim``.
            Sub-bases may have different mode counts.
        n_anchors: Number of anchor positions ``R`` in latent space.
            Ignored when ``anchors`` is given.
        null_space: Optional :class:`NullSpaceBasis` — its
            ``evaluate(z)`` output is concatenated to the design matrix.
            Required to reach the null modes the kernel excludes.
        epsilon: Floor applied to the retained eigenvalues (additional
            safety on top of the basis's own clamping).
        anchor_init_std: Std-dev of the Gaussian initialiser for the
            learnable anchor positions (pre-map space).
        anchors: Optional explicit initial anchor positions ``[R, d]``.
            Mandatory (as integer node ids) for an index basis.
        learnable_anchors: When ``False`` the anchors are a fixed buffer.
        domain_map: ``"auto"`` squashes each coordinate into its basis's
            domain before evaluation; ``"none"`` feeds the raw latent.

    Raises:
        IGLConfigError: On shape mismatches, invalid hyperparameters, a
            joint basis inside a per-dimension sequence, or an index basis
            without fixed integer anchors.
    """

    latent_dim: int
    n_anchors: int
    output_dim: int  # n_anchors + null_space.n_columns (if any)
    learnable_anchors: bool

    def __init__(  # noqa: PLR0913
        self,
        latent_dim: int,
        *,
        bases: SpectralBasis | Sequence[SpectralBasis],
        n_anchors: int = 64,
        null_space: NullSpaceBasis | None = None,
        epsilon: float = 1e-4,
        anchor_init_std: float = 0.25,
        anchors: torch.Tensor | None = None,
        learnable_anchors: bool = True,
        domain_map: DomainMapLike = DomainMap.AUTO,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise IGLConfigError(f"latent_dim must be >= 1, got {latent_dim}")
        if n_anchors < 1:
            raise IGLConfigError(f"n_anchors must be >= 1, got {n_anchors}")
        if epsilon <= 0:
            raise IGLConfigError(f"epsilon must be > 0, got {epsilon}")
        domain_mode = DomainMap(domain_map)

        # Resolve `bases` to a list of SpectralBasis modules.
        if isinstance(bases, nn.Module):
            joint = bool(getattr(bases, "is_joint", False))
            per_dim_bases: list[nn.Module] = [bases] if joint else [bases] * latent_dim
        else:
            seq = list(bases)
            if any(getattr(b, "is_joint", False) for b in seq):
                raise IGLConfigError(
                    "a joint basis is a function of the whole latent; pass it alone, not in a per-dimension sequence",
                )
            if len(seq) != latent_dim:
                raise IGLConfigError(
                    f"bases sequence length ({len(seq)}) must equal latent_dim ({latent_dim})",
                )
            for b in seq:
                if not isinstance(b, nn.Module):
                    raise IGLConfigError("every basis must be an nn.Module")
            joint = False
            per_dim_bases = seq

        self._joint = joint
        self._bases = nn.ModuleList(per_dim_bases)
        index_dims = [bool(getattr(b, "is_index_basis", False)) for b in per_dim_bases]
        self._domains: list[tuple[float, float]] = [
            _UNBOUNDED if (domain_mode is DomainMap.NONE or joint or is_index) else _domain_of(b)
            for b, is_index in zip(per_dim_bases, index_dims, strict=True)
        ]

        self._keeps = _kept_modes(per_dim_bases, null_space_given=null_space is not None)
        # Kept-mode indices as device buffers: indexing with a Python list
        # uploads the indices on every call (a host synchronisation on CUDA).
        # Non-persistent, so checkpoints written before this change still load.
        for j, keep in enumerate(self._keeps):
            self.register_buffer(f"_keep_index_{j}", torch.tensor(keep, dtype=torch.long), persistent=False)

        anchor_tensor = _resolve_anchors(
            anchors,
            latent_dim=latent_dim,
            n_anchors=n_anchors,
            anchor_init_std=anchor_init_std,
            has_index_basis=any(index_dims),
            learnable=learnable_anchors,
        )
        if learnable_anchors:
            self.anchor_positions = nn.Parameter(anchor_tensor)
        else:
            self.register_buffer("anchor_positions", anchor_tensor)

        self.latent_dim = latent_dim
        self.n_anchors = int(anchor_tensor.shape[0])
        self.learnable_anchors = learnable_anchors
        self.epsilon = epsilon
        self.domain_map = domain_mode
        self._null_space = null_space
        self.output_dim = n_anchors + (null_space.n_columns if null_space is not None else 0)

    @property
    def is_joint(self) -> bool:
        """Whether the kernel expands one joint basis on the whole latent."""
        return self._joint

    def map_to_domain(self, z: torch.Tensor) -> torch.Tensor:
        """Squash ``[N, d]`` latents into the per-dimension basis domains (identity for ``"none"``)."""
        if self._joint:
            return z
        columns = [_map_to_domain(z[:, j], self._domains[j]) for j in range(self.latent_dim)]
        return torch.stack(columns, dim=-1)

    def _keep_index(self, j: int) -> torch.Tensor:
        """Kept-mode indices of dimension ``j`` as a ``long`` tensor on the module's device."""
        return cast(torch.Tensor, getattr(self, f"_keep_index_{j}"))

    def _factor(self, basis: nn.Module, keep: torch.Tensor, z_col: torch.Tensor, s_col: torch.Tensor) -> torch.Tensor:
        """``Σ_{k∈keep} φ_k(z) φ_k(s) / max(λ_k, ε)`` as an ``[N, R]`` matrix."""
        phi_z = cast(torch.Tensor, basis(z_col)).index_select(1, keep)
        phi_s = cast(torch.Tensor, basis(s_col)).index_select(1, keep)
        eigvals = cast(torch.Tensor, basis.eigenvalues).index_select(0, keep).clamp(min=self.epsilon)
        return phi_z @ (phi_s / eigvals.unsqueeze(0)).T

    def compute_design_matrix(
        self,
        z: torch.Tensor,
        *,
        gate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the design matrix.

        Args:
            z: ``[N, d]`` latent coordinates.
            gate_mask: Optional ``[d]`` binary mask; latent dims with
                ``mask = 0`` contribute a factor of ``1`` (neutral). A joint
                basis sees the mask only through the masked ``z``.

        Returns:
            ``[N, output_dim]`` design matrix.
        """
        if z.dim() != _EXPECTED_Z_NDIM or z.shape[-1] != self.latent_dim:
            raise IGLConfigError(
                f"z must be [N, {self.latent_dim}]; got {tuple(z.shape)}",
            )
        anchors: torch.Tensor = self.anchor_positions  # [R, d]

        if self._joint:
            kernel_main = self._factor(self._bases[0], self._keep_index(0), z, anchors)
        else:
            z_mapped = self.map_to_domain(z)
            anchors_mapped = self.map_to_domain(anchors)
            kernel_main = torch.ones(z.shape[0], self.n_anchors, device=z.device, dtype=z.dtype)
            for j in range(self.latent_dim):
                factor = self._factor(self._bases[j], self._keep_index(j), z_mapped[:, j], anchors_mapped[:, j])
                if gate_mask is not None:
                    # Blend on the device instead of reading the mask on the host:
                    # a masked dimension contributes the neutral factor 1.
                    keep = gate_mask[j].to(factor.dtype)
                    factor = 1.0 + keep * (factor - 1.0)
                kernel_main = kernel_main * factor

        if self._null_space is None:
            return kernel_main
        null_cols = self._null_space.evaluate(z)  # [N, n_columns]
        return torch.cat([kernel_main, null_cols], dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        *,
        gate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.compute_design_matrix(z, gate_mask=gate_mask)


__all__ = ["SpectralKernel"]
