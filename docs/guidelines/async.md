# Async

How to handle async code (or whether to) in Hother Python libraries.

## What's enforced

- **basedpyright strict** flags missing `await`, calling sync code in async contexts, and missing `Awaitable` annotations.
- **Ruff `ASYNC` rule set** would flag common async pitfalls (blocking I/O in async, `time.sleep` instead of `await asyncio.sleep`) — enable in `pyproject.toml` if your library is async.

## What we expect

### Async or sync?

Decide **once per library**, not per function. Most Hother libraries fall into one of three buckets:

| Library shape | API style |
|---|---|
| Pure computation, no I/O (e.g. parsing, validation) | **Sync only.** Async adds friction without benefit. |
| Network / file I/O that's part of normal use | **Async-first, sync wrappers if useful.** |
| Mixed: long-running operations + simple helpers | **Async core, sync helpers.** Don't make sync callers `asyncio.run()` everything. |

Avoid "rainbow-coloured" APIs that ship both sync and async versions of the same function — they double the surface area and rot independently.

### asyncio vs anyio

| | asyncio | anyio |
|---|---|---|
| Pros | Standard library, no dep, large ecosystem (httpx, etc.) | Supports both asyncio and trio backends; cleaner cancellation; structured concurrency primitives (`TaskGroup`) feel right |
| Cons | Cancellation semantics are subtle; verbose `gather` patterns | Extra dep; some libs are asyncio-only |
| When | Simple async code; integrating with asyncio-only deps | Cancellation-aware libraries; mixing trio + asyncio users; structured concurrency |

**Sibling pattern:** `cancelable` uses **anyio** (it's a cancellation library — needed anyio's first-class cancellation). `streamblocks` uses **asyncio** (just stream processing, no fancy cancellation).

### Async context managers

Use `@asynccontextmanager` or `__aenter__` / `__aexit__` for any async resource that needs cleanup:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def open_stream(uri: str) -> AsyncIterator[Stream]:
    stream = await Stream.connect(uri)
    try:
        yield stream
    finally:
        await stream.aclose()
```

Consumers then write:

```python
async with open_stream(uri) as stream:
    async for event in stream:
        handle(event)
```

Cleaner than try/finally everywhere, and `__aexit__` runs even on exception.

### Cancellation

When using `anyio` / `asyncio`, **never swallow `CancelledError`** (or anyio's `Cancelled`). Re-raise it after any cleanup:

```python
try:
    await long_running_op()
except (asyncio.CancelledError, ...) as exc:
    cleanup()
    raise  # re-raise so the task group knows we're done
```

Catching cancellation silently leaves the task group hung; this is one of the most common async bugs.

### Structured concurrency

Prefer **task groups** over loose `asyncio.create_task()`:

```python
# asyncio (3.11+)
async with asyncio.TaskGroup() as tg:
    t1 = tg.create_task(fetch(url1))
    t2 = tg.create_task(fetch(url2))
# Both tasks awaited / cancelled together. If one raises, the other is cancelled.

# anyio
async with anyio.create_task_group() as tg:
    tg.start_soon(fetch, url1)
    tg.start_soon(fetch, url2)
```

Loose tasks (`asyncio.create_task(...)` without an `await`) are easy to forget and leak — even worse, they swallow exceptions silently until they're garbage-collected.

### Don't call sync I/O from async code

The eternal foot-gun. If you must call a sync API from an async function, push it to a thread:

```python
# Bad — blocks the event loop
def get_config():
    return requests.get(...).json()  # blocking, in async context

# Good — same, off the loop
result = await asyncio.to_thread(requests.get, ...)
# Or use anyio:
result = await anyio.to_thread.run_sync(requests.get, ...)
```

Better still: use an async-native library (`httpx.AsyncClient` instead of `requests`).

### Don't `asyncio.run()` inside library code

`asyncio.run()` opens and closes an event loop. Calling it from inside an async-aware library means consumers can't compose with their own loop.

```python
# Bad — library forces its own loop
def fetch(url: str) -> Response:
    return asyncio.run(_async_fetch(url))

# Good — let consumers decide
async def fetch(url: str) -> Response:
    ...

# If you need a sync wrapper, document it clearly:
def fetch_sync(url: str) -> Response:
    """Sync wrapper around fetch(). Don't call from async code."""
    return asyncio.run(fetch(url))
```

### Testing async code

See [`testing.md`](testing.md) for the framework-specific patterns. Briefly:

- `asyncio_mode = "auto"` lets you write `async def test_foo()` without markers.
- For anyio, use `@pytest.mark.anyio` + a backend-parametrizing fixture.
- For timing assertions in async tests, use context-manager helpers (cancelable's pattern: `assert_completes_within(2.0)`) rather than wall-clock comparisons.

## Examples

### Good

```python
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


@asynccontextmanager
async def stream_events(uri: str) -> AsyncIterator[Event]:
    stream = await Stream.connect(uri)
    try:
        async with asyncio.TaskGroup() as tg:
            heartbeat = tg.create_task(_heartbeat(stream))
            yield stream
            heartbeat.cancel()
    finally:
        await stream.aclose()
```

### Bad

```python
async def stream_events(uri):
    stream = await Stream.connect(uri)
    asyncio.create_task(_heartbeat(stream))  # leaked task, swallowed exceptions
    return stream  # caller has to remember to .aclose() — error-prone

async def fetch(url):
    return requests.get(url).json()  # blocks the event loop
```

## See also

- [`testing.md`](testing.md) — async test framework choice
- [`errors.md`](errors.md) — cancellation as an exception path
- [anyio documentation](https://anyio.readthedocs.io/)
- [PEP 654 — Exception Groups (TaskGroup)](https://peps.python.org/pep-0654/)
