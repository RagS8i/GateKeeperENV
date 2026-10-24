"""APPEND_FILE handling, matching the reference harness contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


def load_append_text(
    append_file: str | None,
    *,
    inputs_dir: Path = Path("inputs"),
) -> tuple[str, dict[str, Any] | None]:
    """Return (text, info).

    An empty name means inputs/append.txt if it exists.  A named file that is
    missing, empty or non-UTF8 is a configuration error.  Paths are confined to
    the mounted inputs/ directory.
    """

    explicit = bool((append_file or "").strip())
    name = (append_file or "").strip() or "append.txt"
    for prefix in ("./inputs/", "inputs/", ".\\inputs\\", "inputs\\"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    base = inputs_dir.resolve()
    path = (inputs_dir / name).resolve()
    if base not in path.parents:
        raise ConfigError(
            f"APPEND_FILE must be a file inside inputs/, got {name!r}"
        )
    if not path.is_file():
        if explicit:
            raise ConfigError(
                f"APPEND_FILE inputs/{name} not found; put the file in inputs/"
            )
        return "", None
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"APPEND_FILE inputs/{name} is not UTF-8 text ({exc.reason})"
        ) from exc
    if not text.strip():
        if explicit:
            raise ConfigError(f"APPEND_FILE inputs/{name} is empty")
        return "", None
    info = {
        "file": f"inputs/{name}",
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return text, info


def with_append(system_prompt: str, append_text: str) -> str:
    if not append_text:
        return system_prompt
    return system_prompt + "\n\n" + append_text
