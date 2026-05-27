"""Starter property-based tests using Hypothesis.

Delete this file once the library has real tests of its own. The point is
to show the @given pattern and confirm `pytest -n auto` works on a clean
clone.
"""

from hypothesis import given
from hypothesis import strategies as st


@given(st.integers(), st.integers())
def test_addition_is_commutative(a: int, b: int) -> None:
    assert a + b == b + a


@given(st.lists(st.integers()))
def test_reverse_is_involutive(xs: list[int]) -> None:
    assert list(reversed(list(reversed(xs)))) == xs
