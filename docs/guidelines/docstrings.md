# Docstrings

How to document Python symbols in Hother libraries.

## What's enforced

- **Ruff `D` rules** enforce docstring presence and style.
- **Google-style docstrings** are the project default — set via `[tool.ruff.lint.pydocstyle] convention = "google"`. This also matches mkdocstrings' default rendering, so the same string powers both the IDE hover and the published docs site.

## What we expect

### Which symbols need a docstring

| Symbol | Required? |
|---|---|
| Public module | No (`D100` ignored) |
| Package (`__init__.py`) | No (`D104` ignored) |
| Public class | **Yes** |
| Public function / method | **Yes** |
| `__init__` | No (`D107` ignored) — describe in the class docstring instead |
| Dunder methods (`__repr__`, `__len__`, etc.) | No (`D105` ignored) |
| Private (`_foo`) | No, but encouraged when behavior is non-obvious |
| Tests | No — name should be self-documenting |

Both siblings configure these ignores. When in doubt, write the docstring — it costs nothing and helps the next reader.

### Google style

```python
def process_stream(source: AsyncIterable[bytes], *, buffer_size: int = 1024) -> StreamProcessor:
    """Build a StreamProcessor over the given byte source.

    The processor consumes `source` lazily — bytes are read only when the
    consumer iterates over the resulting events.

    Args:
        source: The async byte iterator to process.
        buffer_size: How many bytes to read per chunk. Higher values
            reduce overhead at the cost of latency.

    Returns:
        A StreamProcessor ready to be iterated.

    Raises:
        InvalidSourceError: If `source` is not async-iterable.

    Example:
        ```python
        async for event in process_stream(source):
            handle(event)
        ```
    """
```

Sections used:

- **One-line summary** — imperative mood, ends with period, fits in ~80 chars.
- **Optional blank line + extended description** — why this exists, non-obvious behavior, performance notes.
- **`Args:`** — one entry per parameter that needs explanation. Skip self-evident ones.
- **`Returns:`** — what it returns and what shape (only when not obvious from the type hint).
- **`Raises:`** — every exception this function will deliberately raise. Don't list `Exception` — be specific.
- **`Example:`** — a runnable snippet for non-trivial APIs.

Skip sections that have nothing to say. A function with self-documenting types and no surprises is fine with just the summary line.

### Examples in docstrings

Use **fenced Markdown code blocks** with `python` syntax for examples (matches the published docs site):

```python
"""...
Example:
    ```python
    bridge = AnyioBridge(buffer_size=5000)
    async with bridge:
        ...
    ```
"""
```

> **Hother decision needed:** doctest opt-in. Sibling repos use `>>>` REPL examples in some places (streamblocks's `ProcessorConfig`) but don't run them via doctest. Decide whether the template should wire `pytest --doctest-modules` into the test config — currently no.

### Don't repeat the type hints

```python
# Bad — type info is duplicated and rots independently
def foo(name: str) -> int:
    """Return the length.

    Args:
        name (str): A string. (← redundant)

    Returns:
        int: The length. (← redundant)
    """
```

```python
# Good — types live in the signature; the docstring explains intent
def foo(name: str) -> int:
    """Return the byte length of `name` encoded as UTF-8."""
```

### Don't repeat the function name

```python
# Bad
def parse_iso_timestamp(s: str) -> datetime:
    """Parses an ISO timestamp."""  # adds nothing
```

```python
# Good
def parse_iso_timestamp(s: str) -> datetime:
    """Parse an ISO-8601 timestamp into a timezone-aware datetime.

    Naive timestamps (no offset) are treated as UTC.
    """
```

### Module docstrings

The template's `__init__.py` files don't require a docstring (`D104` is ignored), but a one-paragraph one is encouraged for the top-level package:

```python
"""Real-time extraction of structured blocks from text streams.

The main entry point is `StreamProcessor`. See `docs/` for full API
reference.
"""
```

This text shows up on the package's PyPI page (via README) and in IDE hovers.

### What NOT to write

- `# TODO: docstring` placeholders — write a real one or omit.
- `"""This file contains the Foo class."""` — say something useful or skip.
- Long change-log style comments in docstrings — those belong in `CHANGELOG.md`.

## Examples

### Good

```python
class CancellationToken:
    """A cancellation signal shared between producer and consumers.

    Tokens are created by an OperationRegistry and passed into the
    cancellable operation. Consumers check `is_cancelled` periodically
    or register a callback via `add_callback`.

    Tokens are not reusable — once cancelled they remain cancelled.
    """

    def add_callback(self, callback: CancellationCallback) -> None:
        """Register `callback` to fire when the token is cancelled.

        If the token is already cancelled, `callback` runs immediately
        in the caller's context.

        Raises:
            TokenExpiredError: If the token has been garbage-collected.
        """
```

### Bad

```python
class CancellationToken:
    """CancellationToken class.

    Args:
        Nothing.
    """  # adds no information

    def add_callback(self, callback):
        """add_callback method."""  # repeats the name; no detail
```

## See also

- [Google Python Style Guide — Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [PEP 257 — Docstring Conventions](https://peps.python.org/pep-0257/)
- [mkdocstrings Python handler](https://mkdocstrings.github.io/python/) — how these docstrings render in our docs
