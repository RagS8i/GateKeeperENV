"""Hard limits enforced by the harness."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Limits:
    max_steps: int = 40
    max_seconds: float = 600.0
    max_tool_calls_per_reply: int = 10
    max_output_chars: int = 4000
    context_char_budget: int = 144_000
    model_timeout_seconds: float = 180.0
    tool_timeout_seconds: float = 60.0
    verifier_timeout_seconds: float = 120.0
    max_assistant_chars: int = 12_000


DEFAULT_LIMITS = Limits()
