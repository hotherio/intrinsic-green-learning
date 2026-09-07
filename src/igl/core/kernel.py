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

Two formulations compute the same numbers:

- the **reference** path materialises ``log G`` on an ``[N, R, K, d]`` tensor,
  sums over ``d`` and tracks signs for oscillatory operators; it is the CPU
  branch and is kept bit-identical release to release;
- the **fast** path, for operators registered as *separable*
  (``log|G(d, σ)| = -f(d) · g(σ)``: gaussian, laplacian, yukawa), contracts the
  sum over ``d`` as one matrix product ``-f(dist) @ g(σ)ᵀ`` on the
  ``[N, R, d]`` distance tensor, without the ``[N, R, K, d]`` intermediate and
  without sign tracking. Measured 1.6-2x faster forward and 2-4x faster
  forward+backward at 1e-7 relative error; it is the GPU branch. Its
  contraction runs in full precision in both directions so TF32 (the CUDA
  branch's matmul policy) cannot perturb the design matrix.

Sign tracking is skipped on both paths for blocks whose operator is not
oscillatory: the sign is then an exact ``+1`` and the product is unchanged.
"""

from collections.abc import Sequence
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from igl.core.solver import full_precision_matmul
from igl.exceptions import IGLConfigError
from igl.kernels._registry import Operator, get_operator
from igl.types import NullSpaceBasis

KernelPathLiteral = Literal["auto", "reference", "fast"]


class _SeparableLogKernel(torch.autograd.Function):
    """``logk[n, r, k] = -Σ_j f[n, r, j] · g[k, j]`` with a full-precision backward.

    The backward runs inside ``loss.backward()``, outside any forward-time
    precision context, so the two contractions it needs set the precision
    themselves (a no-op off CUDA).
    """

    @staticmethod
    def forward(ctx: Any, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]  # noqa: ANN401
        ctx.save_for_backward(f, g)
        with full_precision_matmul(f.device):
            return -torch.einsum("nrd,kd->nrk", f, g)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]  # noqa: ANN401
        f, g = cast("tuple[torch.Tensor, torch.Tensor]", ctx.saved_tensors)
        with full_precision_matmul(f.device):
            grad_f = -torch.einsum("nrk,kd->nrd", grad, g)
            grad_g = -torch.einsum("nrk,nrd->kd", grad, f)
        return grad_f, grad_g


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
        null_space: Optional :class:`igl.types.NullSpaceBasis` (e.g.
            :class:`igl.spectral.ConstantNullSpace`) whose ``evaluate(z)``
            output is concatenated as extra columns to the design matrix.
            These columns receive their own lstsq weights without Tikhonov
            shrinkage — useful to give the local kernel a DC mode it
            doesn't otherwise carry.
        kernel_path: ``"auto"`` (default) takes the reference formulation on
            CPU and the fast one on MPS and CUDA; ``"reference"`` or ``"fast"``
            force one (the fast path applies only to separable operators;
            the others always use the reference formulation).
    """

    latent_dim: int
    n_anchors: int
    n_scales: int
    output_dim: int

    def __init__(
        self,
        latent_dim: int,
        *,
        n_anchors: int = 64,
        n_scales: int = 4,
        operator: str | Sequence[str] = "gaussian",
        sigma_log_range: tuple[float, float] = (-1.5, 1.5),
        anchor_init_std: float = 0.5,
        null_space: NullSpaceBasis | None = None,
        kernel_path: KernelPathLiteral = "auto",
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise IGLConfigError(f"latent_dim must be >= 1, got {latent_dim}")
        if kernel_path not in ("auto", "reference", "fast"):
            raise IGLConfigError(f"kernel_path must be 'auto', 'reference' or 'fast', got {kernel_path!r}")
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

        n_ops = len(op_names)
        base, remainder = divmod(n_scales, n_ops)
        op_counts = [base + (1 if i < remainder else 0) for i in range(n_ops)]
        total_k = sum(op_counts)
        # Resolve operators eagerly so an unknown name fails at construction time.
        operators: list[Operator] = [get_operator(name) for name in op_names]

        self.latent_dim = latent_dim
        self.n_anchors = n_anchors
        self.n_scales = total_k
        self._operators: list[Operator] = operators
        self._op_counts: list[int] = op_counts
        self._null_space = null_space
        self.kernel_path: KernelPathLiteral = kernel_path
        self.output_dim = n_anchors + (null_space.n_columns if null_space is not None else 0)

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

    def _resolve_path(self, z: torch.Tensor, path: KernelPathLiteral | None) -> Literal["reference", "fast"]:
        chosen = path if path is not None else self.kernel_path
        if chosen == "auto":
            return "reference" if z.device.type == "cpu" else "fast"
        return chosen

    @staticmethod
    def _block_reference(
        operator: Operator,
        dist: torch.Tensor,
        sigma_op: torch.Tensor,
        gate_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """One operator block on the ``[N, R, K_op, d]`` log tensor: ``Π_j G`` as ``[N, R, K_op]``."""
        # Broadcast: dist [N, R, 1, d] vs sigma_op [1, 1, K_op, d].
        log_kvals, signs = operator.fn(dist.unsqueeze(2), sigma_op.unsqueeze(0).unsqueeze(0))
        if gate_mask is not None:
            mask4 = gate_mask[None, None, None, :]
            log_kvals = log_kvals * mask4
        magnitude = torch.exp(log_kvals.sum(dim=-1))  # [N, R, K_op]
        if not operator.is_oscillatory:
            # Every sign is +1: multiplying by it is exact, so it is skipped.
            return magnitude
        if gate_mask is not None:
            # For sign-parity, masked-out dimensions must contribute +1.
            signs = torch.where(gate_mask[None, None, None, :].bool(), signs, torch.ones_like(signs))
        neg_count = (signs < 0).sum(dim=-1)  # [N, R, K_op]
        total_sign = torch.where(neg_count % 2 == 1, -1.0, 1.0)
        return total_sign * magnitude

    @staticmethod
    def _block_fast(
        operator: Operator,
        dist: torch.Tensor,
        sigma_op: torch.Tensor,
        gate_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """One separable block as a contraction on the ``[N, R, d]`` distance tensor."""
        assert operator.separable is not None  # noqa: S101  # guarded by the caller
        f_dist, g_sigma = operator.separable
        f = f_dist(dist)  # [N, R, d]
        g = g_sigma(sigma_op)  # [K_op, d]
        if gate_mask is not None:
            g = g * gate_mask[None, :]  # a masked dimension contributes 0 to the log-sum: factor 1
        logk = cast(torch.Tensor, _SeparableLogKernel.apply(f, g))  # pyright: ignore[reportUnknownMemberType]
        return torch.exp(logk)

    def compute_design_matrix(
        self,
        z: torch.Tensor,
        *,
        gate_mask: torch.Tensor | None = None,
        path: KernelPathLiteral | None = None,
    ) -> torch.Tensor:
        """Build the design matrix from latent coordinates.

        Args:
            z: Latent coordinates of shape ``[N, d]``.
            gate_mask: Optional ``[d]`` binary mask (1 for active dims).
                Used by Matryoshka random truncation during training. Passing
                ``None`` is equivalent to a mask of all ones.
            path: Override of the constructor's ``kernel_path`` for this call.

        Returns:
            Design matrix of shape ``[N, R]``.
        """
        n_samples = z.shape[0]
        device = z.device
        dtype = z.dtype
        use_fast = self._resolve_path(z, path) == "fast"

        # Distances [N, R, d].
        dist = z.unsqueeze(1) - self.anchor_positions.unsqueeze(0)

        sigma = torch.exp(self.log_sigma)
        gamma = F.softmax(self.log_gamma, dim=0)

        phi = torch.zeros(n_samples, self.n_anchors, device=device, dtype=dtype)
        k_offset = 0
        for operator, k_count in zip(self._operators, self._op_counts, strict=True):
            sigma_op = sigma[k_offset : k_offset + k_count]  # [K_op, d]
            gamma_op = gamma[k_offset : k_offset + k_count]  # [K_op]
            if use_fast and operator.separable is not None:
                prod_k = self._block_fast(operator, dist, sigma_op, gate_mask)
            else:
                prod_k = self._block_reference(operator, dist, sigma_op, gate_mask)
            phi = phi + (prod_k * gamma_op[None, None, :]).sum(dim=-1)  # [N, R]
            k_offset += k_count

        importance = torch.sigmoid(self.rank_importance)
        phi = phi * importance.unsqueeze(0)

        if self._null_space is None:
            return phi
        null_cols = self._null_space.evaluate(z)  # [N, n_columns]
        return torch.cat([phi, null_cols], dim=-1)

    def forward(
        self, z: torch.Tensor, *, gate_mask: torch.Tensor | None = None, path: KernelPathLiteral | None = None
    ) -> torch.Tensor:
        """Alias of :meth:`compute_design_matrix`."""
        return self.compute_design_matrix(z, gate_mask=gate_mask, path=path)


__all__ = ["GreenKernel", "KernelPathLiteral"]
