"""Shared pytest fixtures for the IGL test suite.

The autouse RNG seed fixture is the cornerstone: ``pytest-randomly`` reshuffles
test order on every run, which would otherwise let earlier-tests' RNG state
leak into later tests and break determinism for numerical assertions.
"""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_rngs() -> None:
    """Reseed Python, NumPy, and PyTorch RNGs before every test.

    This runs *per-test* (not session-scope) so each test starts from the same
    deterministic state regardless of the shuffled run order.
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
