"""JSONL and console logging for reproducible trajectories."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts", utc_now())
        self._handle.write(
            json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def print_line(text: str) -> None:
    print(text, file=sys.stdout, flush=True)


# Template-compatible alias.
JsonlLog = JsonlWriter


def print_step(
    *,
    task_id: str,
    step: int,
    tools: list[str],
    state_hash: str | None,
    shadow_score: float | None,
    elapsed: float,
) -> None:
    score_text = "n/a" if shadow_score is None else f"{shadow_score:.3f}"
    print_line(
        f"[{task_id}] step {step:02d} tools={','.join(tools) or '-'} "
        f"state={state_hash or '-'} shadow={score_text} elapsed={elapsed:.1f}s"
    )


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n...[log-truncated]"
