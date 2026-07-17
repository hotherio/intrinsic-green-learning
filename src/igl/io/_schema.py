"""Checkpoint schema for :func:`igl.save` / :func:`igl.load`."""

from typing import cast

from igl.exceptions import IGLSerializationError

SCHEMA_VERSION = 1

_KINDS = frozenset({"igl_module", "classifier", "regressor", "autoencoder", "distiller"})
_REQUIRED_KEYS = frozenset({"schema_version", "kind", "dims", "config", "state_dict", "preprocessing", "provenance"})


def validate_payload(payload: object) -> dict[str, object]:
    """Check the loaded payload against schema v1 and return it typed as a dict.

    Raises:
        IGLSerializationError: If the payload is not a schema-v1 checkpoint.
    """
    if not isinstance(payload, dict):
        raise IGLSerializationError(f"not an igl checkpoint: expected a dict payload, got {type(payload).__name__}")
    typed = cast(dict[str, object], payload)
    missing = _REQUIRED_KEYS - typed.keys()
    if missing:
        raise IGLSerializationError(f"not an igl checkpoint: missing keys {sorted(missing)}")
    version = typed["schema_version"]
    if version != SCHEMA_VERSION:
        raise IGLSerializationError(f"unsupported checkpoint schema version {version!r} (this build reads {SCHEMA_VERSION})")
    kind = typed["kind"]
    if kind not in _KINDS:
        raise IGLSerializationError(f"unknown checkpoint kind {kind!r} (expected one of {sorted(_KINDS)})")
    return typed
