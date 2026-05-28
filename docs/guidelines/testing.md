# Testing

How Hother Python libraries are tested, extracted from `cancelable` and `streamblocks`.

## What's enforced

- **pytest** with `xfail_strict = true` (an `xfail` that unexpectedly passes is a test failure).
- **`-n auto`** (pytest-xdist) — tests run in parallel.
- **`pytest-randomly`** — test order is shuffled each run.
- **`--strict-markers --strict-config`** — typos in `@pytest.mark.xxx` or unknown ini options fail loudly.
- **Coverage tracking** via `pytest-cov` writing to `term-missing` + `html`.
- **`hypothesis`** is available in the dev group for property-based tests.
- **`vulture`** is available via `make vulture` for dead-code detection.

## What we expect

### Layout

- Tests live in `tests/` at repo root (single directory, not split into `unit/` vs `integration/`).
- One file per logical area: `test_<module>.py`.
- Shared fixtures and helpers in `tests/conftest.py`.
- Test functions named `test_*`; classes named `Test*` (PEP 8 + pytest convention).

### Fixture scope

**Default to `function` scope.** Both siblings explicitly set `asyncio_default_fixture_loop_scope = "function"`.

Use `session` / `module` only for genuinely expensive resources (DB connections, large fixture data). Document the reason inline.

### Async tests

Two patterns, pick per library:

| Library uses | Test pattern |
|---|---|
| `anyio` (e.g., `cancelable`) | `pytest-anyio`, mark tests `@pytest.mark.anyio`, parametrize the backend via a fixture |
| `asyncio` (e.g., `streamblocks`) | `pytest-asyncio` with `asyncio_mode = "auto"` (no marker needed per test) |

### Coverage threshold

**`fail_under = 100`** is the Hother default — siblings both target 100%.

If you legitimately can't reach 100%, add a `pragma: no cover` comment with a one-line reason directly above the uncovered branch. Don't lower the threshold to dodge a hard case.

The `[tool.coverage.report].exclude_lines` list in `pyproject.toml` already covers the standard exceptions (`pragma: no cover`, `if __name__ == "__main__":`, `raise NotImplementedError`, etc.). Add more entries there rather than scattering comments.

### Property-based tests with Hypothesis

Use `@given(...)` for any pure function with a non-trivial domain:

```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_addition_is_commutative(a: int, b: int) -> None:
    assert a + b == b + a
```

When Hypothesis finds a counterexample, the failing example is appended to `.hypothesis/examples/` and re-tried on every run — don't `.gitignore` that directory.

### Mocking

- **Default to `pytest-mock`'s `mocker` fixture** — cleaner than `unittest.mock` patches and auto-restores after each test.
- **Don't mock the database or the network at unit-test boundaries.** Use integration tests with real services (test DB / `httpx.MockTransport`). Mocked behavior diverges from production and hides bugs.
- **Mock at the boundary you control**, not the boundary you depend on. E.g., mock your own `EmailService` interface, not `smtplib`.

### What NOT to test

- Auto-generated code (Pydantic models with no custom logic).
- Trivial property accessors / one-liners that exist just to satisfy a Protocol.
- Other libraries' behavior (we test our integration with them, not them).

## Examples

### Good

```python
from collections.abc import Iterator

import pytest
from hypothesis import given, strategies as st

from hother.streamblocks import StreamState


@pytest.fixture
def empty_state() -> StreamState:
    return StreamState.new()


def test_initial_state_has_no_events(empty_state: StreamState) -> None:
    assert empty_state.event_count == 0


@given(st.lists(st.integers()))
def test_append_preserves_order(empty_state: StreamState, xs: list[int]) -> None:
    for x in xs:
        empty_state.append(x)
    assert list(empty_state.events()) == xs
```

### Bad

```python
def test_things():  # vague name; tests multiple things
    import unittest.mock  # use mocker fixture
    with unittest.mock.patch("httpx.AsyncClient.get") as m:  # mocks the dep, not our seam
        m.return_value.json.return_value = {"ok": True}
        result = my_function()
        assert result  # no specific assertion
```

## See also

- [`async.md`](async.md) — asyncio vs anyio decision, which drives the test framework choice
- [pytest documentation](https://docs.pytest.org/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [`errors.md`](errors.md) — how exceptions are tested (`pytest.raises` patterns)
