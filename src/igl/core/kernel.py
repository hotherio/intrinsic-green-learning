"""Multi-scale, multi-operator Green's-function kernel.

Given latent coordinates ``z ∈ R^{N × d}`` and ``R`` learnable anchors
``μ_r ∈ R^d``, the kernel computes the design matrix

::

    Φ_{n,r} = sigmoid(rho_r) · Σ_k γ_k · Π_j G_op(z_{n,j} - μ_{r,j}, σ_{k,j})^{m_j}

where ``m_j`` is an optional gate mask (used at training time by Matryoshka
truncation) and the operator can vary across scales (multi-operator
configurations). Computation is in log-space with sign-tracking so oscillatory
kernels (helmholtz, gabor, mexican_hat) can be combined safely.

The kernel exposes :func:`forward` (the standard ``nn.Module`` entry point) and
the more explicitly-named :func:`compute_design_matrix`, which are aliases.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from igl.exceptions import IGLConfigError
from igl.kernels._registry import Operator, get_operator


class GreenKernel(nn.Module):
    """Multi-scale, multi-operator Green's-function kernel.

    Args:
        latent_dim: Latent dimension ``d_max``.
        n_anchors: Number of anchor positions ``R``.
        n_scales: Total number of scales ``K`` (split across operators if
            ``operator`` is a sequence).
        operator: Single operator name (e.g. ``"gaussian"``) or a non-empty
            sequence of names for a multi-operator configuration.
        sigma_log_range: ``(low, high)`` log-scale range used to initialise
            ``log_sigma`` linearly across the ``K`` scales (default
            ``(-1.5, 1.5)``).
        anchor_init_std: Standard deviation for the Gaussian initialisation
            of anchor positions (default ``0.5``).
    """

    latent_dim: int
    n_anchors: int
    n_scales: int

    def __init__(
        self,
        latent_dim: int,
        *,
        n_anchors: int = 64,
        n_scales: int = 4,
        operator: str | Sequence[str] = "gaussian",
        sigma_log_range: tuple[float, float] = (-1.5, 1.5),
        anchor_init_std: float = 0.5,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise IGLConfigError(f"latent_dim must be >= 1, got {latent_dim}")
        if n_anchors < 1:
            raise IGLConfigError(f"n_anchors must be >= 1, got {n_anchors}")
        if n_scales < 1:
            raise IGLConfigError(f"n_scales must be >= 1, got {n_scales}")

        op_names: list[str] = [operator] if isinstance(operator, str) else list(operator)
        if not op_names:
            raise IGLConfigError("operator sequence must be non-empty")
        if n_scales < len(op_names):
            raise IGLConfigError(
                f"n_scales ({n_scales}) must be >= number of operators ({len(op_names)})",
            )

        base = n_scales // len(op_names)
        remainder = n_scales % len(op_names)
        op_counts = [base + (1 if i < remainder else 0) for i in range(len(op_names))]
        total_k = sum(op_counts)
        # Resolve operators eagerly so an unknown name fails at construction time.
        operators: list[Operator] = [get_operator(name) for name in op_names]

        self.latent_dim = latent_dim
        self.n_anchors = n_anchors
        self.n_scales = total_k
        self._operators: list[Operator] = operators
        self._op_counts: list[int] = op_counts

        self.anchor_positions = nn.Parameter(torch.randn(n_anchors, latent_dim) * anchor_init_std)
        self.rank_importance = nn.Parameter(torch.ones(n_anchors))
        self.log_sigma = nn.Parameter(
            torch.linspace(sigma_log_range[0], sigma_log_range[1], total_k).unsqueeze(1).expand(total_k, latent_dim).clone(),
        )
        self.log_gamma = nn.Parameter(torch.zeros(total_k))

    @property
    def operator_names(self) -> tuple[str, ...]:
        """Names of the operators in registration order."""
        return tuple(op.name for op in self._operators)

    def compute_design_matrix(
        self,
        z: torch.Tensor,
        *,
        gate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the design matrix from latent coordinates.

        Args:
            z: Latent coordinates of shape ``[N, d]``.
            gate_mask: Optional ``[d]`` binary mask (1 for active dims).
                Used by Matryoshka random truncation during training. Passing
                ``None`` is equivalent to a mask of all ones.

        Returns:
            Design matrix of shape ``[N, R]``.
        """
        n_samples = z.shape[0]
        device = z.device
        dtype = z.dtype

        # Distances [N, R, d].
        dist = z.unsqueeze(1) - self.anchor_positions.unsqueeze(0)

        sigma = torch.exp(self.log_sigma)
        gamma = F.softmax(self.log_gamma, dim=0)

        phi = torch.zeros(n_samples, self.n_anchors, device=device, dtype=dtype)
        k_offset = 0
        for operator, k_count in zip(self._operators, self._op_counts, strict=True):
            sigma_op = sigma[k_offset : k_offset + k_count]  # [K_op, d]
            gamma_op = gamma[k_offset : k_offset + k_count]  # [K_op]

            # Broadcast: dist [N, R, 1, d] vs sigma_op [1, 1, K_op, d].
            log_kvals, signs = operator.fn(
                dist.unsqueeze(2),
                sigma_op.unsqueeze(0).unsqueeze(0),
            )

            if gate_mask is not None:
                mask4 = gate_mask[None, None, None, :]
                log_kvals = log_kvals * mask4
                # For sign-parity, masked-out dimensions must contribute +1.
                effective_signs = torch.where(mask4.bool(), signs, torch.ones_like(signs))
            else:
                effective_signs = signs

            # Sign parity across dimensions.
            neg_count = (effective_signs < 0).sum(dim=-1)  # [N, R, K_op]
            total_sign = torch.where(neg_count % 2 == 1, -1.0, 1.0)

            # Product over dimensions in log-space.
            prod_k = total_sign * torch.exp(log_kvals.sum(dim=-1))  # [N, R, K_op]
            phi = phi + (prod_k * gamma_op[None, None, :]).sum(dim=-1)

            k_offset += k_count

        importance = torch.sigmoid(self.rank_importance)
        return phi * importance.unsqueeze(0)

    def forward(self, z: torch.Tensor, *, gate_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Alias of :meth:`compute_design_matrix`."""
        return self.compute_design_matrix(z, gate_mask=gate_mask)


__all__ = ["GreenKernel"]
