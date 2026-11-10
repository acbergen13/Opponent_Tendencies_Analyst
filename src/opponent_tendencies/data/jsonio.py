"""Canonical JSON serialization and crash-safe writes.

Every store in this project is a single JSON document that is replaced
atomically. Keeping the serializer and the write sequence in one place means
the play store, the media store, and any later store share the same guarantees:

* keys and records are serialized deterministically, so identical data produces
  identical bytes;
* a write goes to a temporary file in the destination directory, is flushed and
  fsynced, and is then renamed over the target;
* the parent directory is fsynced so the rename survives a crash.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "atomic_write_text",
    "canonical_dump",
]


def canonical_dump(payload: Mapping[str, object]) -> str:
    """Serialize ``payload`` to stable, compact JSON with a trailing newline."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync so the rename survives a crash."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(descriptor)
