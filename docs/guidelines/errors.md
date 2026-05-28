# Errors

How exceptions are designed and used in Hother Python libraries.

## What's enforced

- **basedpyright strict** flags bare `except:` and unreachable except clauses.
- **Ruff** enforces `raise ... from <cause>` in `try`/`except` rewrites via the `B904` rule.

## What we expect

### Custom base exception class

Every library defines a **single base exception class** that all library-raised exceptions inherit from. This lets downstream consumers catch only "errors from this library" without catching everything.

```python
# In src/<package>/exceptions.py
class CancelableError(Exception):
    """Base class for all exceptions raised by hother.cancelable."""
```

Cancelable does this (`CancelationError(Exception)`). Streamblocks chose a different model — see "Errors as events" below — appropriate to its event-stream domain.

### Exception hierarchy

Inherit specific exceptions from the base. Use one level deep by default; add a third level only for genuinely distinct cases consumers will want to catch differently:

```python
class CancelableError(Exception): ...

class TokenExpiredError(CancelableError): ...
class CancellationSourceError(CancelableError): ...

class TimeoutCancellationError(CancellationSourceError): ...
class SignalCancellationError(CancellationSourceError): ...
```

### Errors as events (alternative)

For event-stream / pipeline libraries (e.g. `streamblocks`), surface errors as **typed events** rather than exceptions:

```python
from enum import StrEnum

class BlockErrorCode(StrEnum):
    PARSE_FAILED = "parse_failed"
    INVALID_CONTEXT = "invalid_context"

class BlockErrorEvent(BaseModel):
    code: BlockErrorCode
    message: str
    block_index: int
```

This keeps the async control flow linear instead of having every consumer wrap iterators in `try`/`except`.

**Pick exceptions OR events per library, not both.** Mixing them confuses consumers about which error path they're on.

### Chaining

When re-raising or wrapping, **always use `raise ... from`** so the cause is preserved:

```python
try:
    payload = json.loads(body)
except json.JSONDecodeError as exc:
    raise InvalidPayloadError(f"could not parse body: {body[:40]!r}") from exc
```

Use `from None` to **deliberately suppress** the cause (rare; document why):

```python
except KeyError:
    raise ConfigError("required key 'api_url' missing") from None
```

### Don't swallow exceptions

A bare `except: pass` or `except Exception: pass` hides bugs. If you genuinely need to ignore an exception:

```python
try:
    ...
except SpecificError:
    logger.debug("ignoring known-benign condition", exc_info=True)
```

The narrowest possible `except` class. A `logger.debug` (not `pass`). And `exc_info=True` so the stack is reachable if the assumption turns out wrong.

### Structured exception attributes

For exceptions consumers will introspect, give them typed attributes — not just a message string:

```python
class RateLimitedError(ApiError):
    def __init__(self, *, retry_after: float, request_id: str) -> None:
        super().__init__(f"rate-limited; retry after {retry_after:.1f}s")
        self.retry_after = retry_after
        self.request_id = request_id
```

This is more useful than parsing the message in downstream code.

### Validation errors

For input validation, **lean on Pydantic** rather than rolling custom `ValueError`:

```python
class StreamConfig(BaseModel):
    buffer_size: int = Field(gt=0, le=100_000)
    backend: Literal["asyncio", "anyio"]
```

Pydantic produces a `ValidationError` with structured detail — better than a per-field `ValueError`.

## Examples

### Good

```python
class OperationRegistry:
    def cancel(self, operation_id: str) -> None:
        try:
            operation = self._lookup(operation_id)
        except KeyError as exc:
            raise UnknownOperationError(operation_id=operation_id) from exc
        operation.cancel()


class UnknownOperationError(CancelableError):
    def __init__(self, *, operation_id: str) -> None:
        super().__init__(f"no operation registered with id={operation_id!r}")
        self.operation_id = operation_id
```

### Bad

```python
def cancel(self, operation_id):
    try:
        operation = self._lookup(operation_id)
    except:                                            # bare except
        raise Exception("not found")                   # generic Exception, lost cause
    try:
        operation.cancel()
    except Exception:                                  # silent swallow
        pass
```

## See also

- [`testing.md`](testing.md) — patterns for `pytest.raises(MyError, match=...)`
- [PEP 3134 — Exception Chaining](https://peps.python.org/pep-3134/)
