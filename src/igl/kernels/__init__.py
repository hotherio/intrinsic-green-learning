"""Multi-scale Green's-function kernel operator zoo + registry.

Importing this package eagerly imports every bundled operator, which triggers
each module's ``register_operator(...)`` call. Downstream users can add their
own kernels at any time via :func:`register_operator`.
"""

# Import bundled operators to trigger their side-effecting registration.
# Order is alphabetical and does not affect behaviour; the registry indexes
# by name.
from igl.kernels import (
    cauchy,
    gabor,
    gaussian,
    helmholtz,
    laplacian,
    mexican_hat,
    multiquadric,
    soft_box,
    yukawa,
)
from igl.kernels._registry import (
    Operator,
    get_operator,
    list_operators,
    register_operator,
)

__all__ = [
    "Operator",
    "cauchy",
    "gabor",
    "gaussian",
    "get_operator",
    "helmholtz",
    "laplacian",
    "list_operators",
    "mexican_hat",
    "multiquadric",
    "register_operator",
    "soft_box",
    "yukawa",
]
