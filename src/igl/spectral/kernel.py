# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportCallIssue=false, reportArgumentType=false
"""Spectral Green's-function kernel — peer of :class:`igl.GreenKernel`.

For each latent dimension ``j``, ``SpectralKernel`` evaluates a 1-D
:class:`SpectralBasis`. The multi-dim Green's function is the
*separable product*

::

    G(z, μ_r) = Π_j Σ_k φ_k(z_j) · φ_k(μ_{r,j}) / max(λ_k, ε)

plus extra null-space columns from an optional
:class:`igl.types.NullSpaceBasis`. The full design matrix has shape
``[N, n_anchors + null_space.n_columns]``.

Anchors ``μ`` are learnable just like in :class:`igl.GreenKernel`.
Null modes intrinsic to the basis (e.g., Fourier cosine's ``n=0``) are
automatically zeroed *inside the product* via the eigenvalue floor —
their contribution comes through the augmented null-space columns
instead, so the lstsq solve can fit them without Tikhonov shrinkage.
"""

from collections.abc import Sequence

import torch
from torch import nn

from igl.exceptions import IGLConfigError
from igl.types import NullSpaceBasis, SpectralBasis

_EXPECTED_Z_NDIM = 2


class SpectralKernel(nn.Module):
    """Spectral Green's-function kernel.

    Args:
        latent_dim: Latent dimensionality ``d``.
        bases: One :class:`SpectralBasis` (used uniformly across all
            ``latent_dim`` dimensions) or a sequence of length
            ``latent_dim``. Sub-bases may have different mode counts.
        n_anchors: Number of anchor positions ``R`` in latent space.
        null_space: Optional :class:`NullSpaceBasis` — its
            ``evaluate(z)`` output is concatenated to the design matrix.
        epsilon: Floor applied to eigenvalues (additional safety on top
            of the basis's own clamping).
        anchor_init_std: Std-dev of the Gaussian initialiser for the
            learnable anchor positions.

    Raises:
        IGLConfigError: On shape mismatches or invalid hyperparameters.
    """

    latent_dim: int
    n_anchors: int
    output_dim: int  # n_anchors + null_space.n_columns (if any)

    def __init__(
        self,
        latent_dim: int,
        *,
        bases: SpectralBasis | Sequence[SpectralBasis],
        n_anchors: int = 64,
        null_space: NullSpaceBasis | None = None,
        epsilon: float = 1e-4,
        anchor_init_std: float = 0.25,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise IGLConfigError(f"latent_dim must be >= 1, got {latent_dim}")
        if n_anchors < 1:
            raise IGLConfigError(f"n_anchors must be >= 1, got {n_anchors}")
        if epsilon <= 0:
            raise IGLConfigError(f"epsilon must be > 0, got {epsilon}")

        # Resolve `bases` to a per-dim list of SpectralBasis modules.
        if isinstance(bases, nn.Module):
            per_dim_bases: list[nn.Module] = [bases] * latent_dim
        else:
            seq = list(bases)
            if len(seq) != latent_dim:
                raise IGLConfigError(
                    f"bases sequence length ({len(seq)}) must equal latent_dim ({latent_dim})",
                )
            for b in seq:
                if not isinstance(b, nn.Module):
                    raise IGLConfigError("every basis must be an nn.Module")
            per_dim_bases = seq

        self._bases = nn.ModuleList(per_dim_bases)
        self.latent_dim = latent_dim
        self.n_anchors = n_anchors
        self.epsilon = epsilon
        self._null_space = null_space
        self.output_dim = n_anchors + (null_space.n_columns if null_space is not None else 0)

        # Per-dim anchor coordinates [R, d].
        anchors = torch.randn(n_anchors, latent_dim) * anchor_init_std
        self.anchor_positions = nn.Parameter(anchors)

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
                ``mask = 0`` contribute a factor of ``1`` (neutral).

        Returns:
            ``[N, output_dim]`` design matrix.
        """
        if z.dim() != _EXPECTED_Z_NDIM or z.shape[-1] != self.latent_dim:
            raise IGLConfigError(
                f"z must be [N, {self.latent_dim}]; got {tuple(z.shape)}",
            )
        n_samples = z.shape[0]
        anchors = self.anchor_positions  # [R, d]

        # Per-dimension product: start with ones and multiply in the
        # spectral Green's function for each active dim.
        kernel_main = torch.ones(n_samples, self.n_anchors, device=z.device, dtype=z.dtype)
        for j in range(self.latent_dim):
            if gate_mask is not None and gate_mask[j].item() == 0:
                continue
            basis = self._bases[j]
            phi_z = basis(z[:, j])  # [N, K_j]
            phi_s = basis(anchors[:, j])  # [R, K_j]
            eigvals = basis.eigenvalues.clamp(min=self.epsilon)
            # G_j(z, μ_r) = Σ_k φ_k(z) φ_k(μ_r) / max(λ_k, ε)
            weighted_phi_s = phi_s / eigvals.unsqueeze(0)  # [R, K_j]
            kernel_j = phi_z @ weighted_phi_s.T  # [N, R]
            kernel_main = kernel_main * kernel_j

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
