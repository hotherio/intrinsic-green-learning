# API design

Conventions for public-API surface in Hother Python libraries.

## What's enforced

- **basedpyright strict** flags unannotated public symbols and unsafe overload signatures.
- **Ruff `N` rules** enforce PEP 8 naming.
- **Ruff `D` rules** enforce docstring presence (see [`docstrings.md`](docstrings.md)).

## What we expect

### `__all__` is mandatory and comprehensive

Every package's top-level `__init__.py` declares `__all__` exhaustively. Both siblings do — cancelable lists 90 items, streamblocks ~50, grouped by category in comments.

```python
# src/hother/streamblocks/__init__.py
"""Real-time extraction of structured blocks from text streams."""

from .core.processor import StreamProcessor
from .events import BlockEvent, BlockErrorEvent, StreamEvent
from .syntaxes import SyntaxRegistry, register_syntax

__all__ = [
    # Core
    "StreamProcessor",
    # Events
    "BlockEvent",
    "BlockErrorEvent",
    "StreamEvent",
    # Syntaxes
    "SyntaxRegistry",
    "register_syntax",
]
```

Three reasons:

1. **Discoverability** — `from package import *` and IDE autocomplete pick up the listed items.
2. **Stability contract** — anything not in `__all__` is implicitly private (even without `_` prefix).
3. **Refactor safety** — `basedpyright` warns if `__all__` references an undefined symbol, so renames break loudly.

### Re-exports

Top-level `__init__.py` should **flatten the API surface**: import the public symbols from their submodules and re-export them via `__all__`. Consumers should `from hother.streamblocks import StreamProcessor`, never `from hother.streamblocks.core.processor import StreamProcessor`.

Internal modules can stay deeply nested; only the import surface is flat.

### Private = leading underscore

- Module-level: `_internal_helper` is private; consumers may not import it directly.
- Class attributes: `self._lock`, `self._pending_callbacks` — private state.
- Module names: `_internal.py` for whole modules that shouldn't be imported externally.

### Keyword-only arguments

Use `*` to **force kwargs** in constructors and public functions with more than one optional argument:

```python
def from_iterable(
    cls,
    source: Iterable[Event],
    *,
    buffer_size: int = 1024,
    backend: Backend = Backend.AUTO,
) -> Self: ...
```

This:

- Makes call sites self-documenting (`StreamProcessor.from_iterable(events, buffer_size=2048)`).
- Lets us add / reorder kwargs without breaking callers (positional ordering is frozen).
- Catches `from_iterable(events, 2048)` (which `buffer_size` did the caller mean?) at type-check time.

### Positional-only arguments

Use `/` to **forbid kwargs** when the parameter name is implementation detail:

```python
def from_dict(cls, data: Mapping[str, Any], /) -> Self: ...
```

This frees you to rename `data` later without it being a breaking change.

### `@overload`

Use for functions whose return type depends on input type (rare in library code):

```python
from typing import overload

@overload
def get(self, key: str) -> str: ...
@overload
def get(self, key: str, default: T) -> str | T: ...

def get(self, key: str, default: T | _Missing = _MISSING) -> str | T: ...
```

Neither sibling uses `@overload` heavily. Don't reach for it for one-off Union returns; prefer designing the API to have a single return type.

### Deprecation

When removing a public symbol or argument, deprecate it for **at least one minor version** before removal:

```python
import warnings

def old_function(...) -> ...:
    warnings.warn(
        "old_function() is deprecated, use new_function() instead. "
        "Will be removed in 2.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return new_function(...)
```

Then bump the major version when removing (see [`releases.md`](releases.md)).

### Avoiding leaky abstractions

- **Don't return third-party types from the public API** unless they're part of a well-known contract (`pathlib.Path`, `datetime`, `pydantic.BaseModel`). Wrap them.
- **Don't accept `**kwargs` for "future extensibility"** — it bypasses type checking and surfaces no documentation. Add explicit parameters when you need them.
- **Don't expose internal singletons.** If a global state matters to consumers, give them a factory or context manager.

## Examples

### Good

```python
"""Public API for hother.cancelable."""

from .core.cancelable import Cancelable
from .core.token import CancellationToken
from .exceptions import CancelableError, TokenExpiredError
from .registry import OperationRegistry

__all__ = [
    "Cancelable",
    "CancellationToken",
    "CancelableError",
    "OperationRegistry",
    "TokenExpiredError",
]


class Cancelable:
    def __init__(
        self,
        operation_id: str,
        /,
        *,
        token: CancellationToken | None = None,
        timeout_s: float | None = None,
    ) -> None:
        ...
```

### Bad

```python
# No __all__; magic re-exports
from .core import *
from .registry import *

# Positional everything, type-unsafe **kwargs
class Cancelable:
    def __init__(self, operation_id, token=None, timeout=None, **kwargs):
        ...
```

## See also

- [`docstrings.md`](docstrings.md) — public symbols require docstrings
- [`releases.md`](releases.md) — semver discipline ties to API changes
- [PEP 8 — Naming Conventions](https://peps.python.org/pep-0008/#naming-conventions)
