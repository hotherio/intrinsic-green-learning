# Typing

How types are written in Hother Python libraries, extracted from how `cancelable` and `streamblocks` actually use them.

## What's enforced

- **basedpyright in strict mode** (`[tool.basedpyright] typeCheckingMode = "strict"`). Configured in `pyproject.toml`. Runs in pre-commit (`uv run basedpyright src`) and is blocking in CI.
- **`ty` runs in parallel** as a non-blocking CI signal so we track Astral's checker as it approaches 1.0.
- **Ruff `UP` rule set** enforces modern annotation syntax (`list[int]`, not `List[int]`).
- **`py.typed` marker** ships with the package so downstream consumers see our types.

## What we expect

### `from __future__ import annotations`

Use **selectively** — not in every file. Both siblings add it only where it actually buys something:

- Files with generics or forward references.
- Files declaring types that reference themselves.

Adding it everywhere is fine but unnecessary noise. Don't add it to one-line modules.

### Generics

**Use the pre-PEP-695 `TypeVar`/`ParamSpec` form.** Both siblings do:

```python
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

def wrap(fn: Callable[P, R]) -> Callable[P, R]: ...
```

> **Hother decision needed:** PEP 695 syntax (`def foo[T](x: T) -> T:`) is fully supported in Python 3.12+. We could adopt it now. We haven't because basedpyright + pre-695 is what siblings shipped. Decide whether to migrate template-wide and update siblings, or stay on `TypeVar` for consistency.

### Protocols

- **Use Protocols for structural callback typing**, e.g. `ProgressCallback`, `StatusCallback`. Both siblings do this in `types.py`.
- **Don't mark them `@runtime_checkable`** unless you need `isinstance(x, MyProtocol)`. Neither sibling uses it — static typing is enough.

```python
class ProgressCallback(Protocol):
    def __call__(self, *, completed: int, total: int) -> None | Awaitable[None]: ...
```

### Data shapes

Pick the lightest construct that fits:

| Construct | When to use |
|---|---|
| `pydantic.BaseModel` | Domain models with validation (e.g. `OperationContext`, `CancelationToken`). Use when input comes from outside the library. |
| `@dataclass` | Internal config / state objects (e.g. `ProcessorConfig`, `StreamState`). No validation needed; better performance. |
| `TypedDict` | **Not used** in either sibling. Prefer one of the above unless you specifically need a dict-shaped public API. |

### `Self` in returns

Use `from typing import Self` for methods returning the class instance, especially classmethod factories and singleton patterns:

```python
from typing import Self

@classmethod
def get_instance(cls) -> Self: ...
```

### Suppressing the checker

**Never use blanket `# type: ignore`.** Always use scoped suppressions with the specific rule code:

```python
foo: Any = bar  # pyright: ignore[reportUnknownMemberType]
```

If a whole file needs an exception (e.g. an optional integration whose dep is missing in dev), add it to `[tool.basedpyright].exclude` in `pyproject.toml` with a comment explaining why.

### Where `Any` is tolerated

`Any` appears in these contexts in sibling code and is acceptable:

- `PrivateAttr` fields in Pydantic models (`_event: Any = PrivateAttr(default=None)`) — framework-injected.
- Async callback return types (`None | Awaitable[None]`) when the caller is generic over sync/async.
- Heterogeneous metadata dicts (`dict[str, Any]`) at API boundaries.

Anywhere else, justify it in a comment or replace with a concrete type / Protocol / `TypeVar`.

## Examples

### Good

```python
from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol, Self

class Subscriber(Protocol):
    def __call__(self, event: Event) -> None | Awaitable[None]: ...

class Stream:
    @classmethod
    def from_iterable(cls, source: Iterable[Event]) -> Self:
        ...
```

### Bad

```python
# Module docstring missing; from __future__ used pointlessly in a leaf module
from __future__ import annotations
from typing import Any, List  # pre-3.9 syntax; use list[...]

def process(items: List[Any]):  # type: ignore  # blanket suppression
    return items
```

## See also

- [basedpyright strict-mode rule list](https://docs.basedpyright.com/dev/configuration/config-files/)
- [PEP 695 — Type Parameter Syntax](https://peps.python.org/pep-0695/) (not yet adopted)
- [PEP 692 — TypedDict for kwargs](https://peps.python.org/pep-0692/) (not used; mentioned for awareness)
- [`api-design.md`](api-design.md) — public-surface conventions that interact with typing
