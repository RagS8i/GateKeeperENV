"""Model clients: a thin OpenAI-compatible client and deterministic mocks."""

from __future__ import annotations

import json
from typing import Any

from .parser import Reply, ToolCall, parse_assistant_message


def _reply_from_actions(actions: list[dict[str, Any]]) -> Reply:
    calls: list[ToolCall] = []
    for index, action in enumerate(actions):
        arguments = action.get("args", {})
        raw = json.dumps(arguments)
        calls.append(
            ToolCall(
                name=str(action["tool"]),
                arguments=arguments,
                raw_arguments=raw,
                call_id=f"mock_{index}",
            )
        )
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments,
                },
            }
            for call in calls
        ],
    }
    return Reply(
        text="",
        calls=calls,
        assistant_message=assistant_message,
        usage={},
        raw={"mock": True},
    )


class OpenAIClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RuntimeError(
                "the openai package is required for non-mock models"
            ) from exc
        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
        )

    def start(self, _task_context: dict[str, Any]) -> None:
        return

    def finish(self) -> None:
        return

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout: float,
    ) -> Reply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "timeout": timeout,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = self._client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return Reply(text="", calls=[], assistant_message={"role": "assistant", "content": ""})
        message = choices[0].message
        reply = parse_assistant_message(message)
        usage = getattr(response, "usage", None)
        if usage is not None:
            if hasattr(usage, "model_dump"):
                reply.usage = usage.model_dump()
            elif isinstance(usage, dict):
                reply.usage = usage
        if hasattr(response, "model_dump"):
            reply.raw = response.model_dump()
        return reply


class MockClient:
    """Deterministic clients used by the baseline gate and offline tests."""

    def __init__(
        self,
        mode: str,
        *,
        batch_size: int = 8,
    ) -> None:
        self.mode = mode
        self.batch_size = max(1, batch_size)
        self.actions: list[dict[str, Any]] = []
        self.cursor = 0

    def start(self, task_context: dict[str, Any]) -> None:
        self.cursor = 0
        self.actions = []
        if self.mode == "solver":
            self.actions = list(task_context.get("oracle_actions", []))
        elif self.mode in {"allow_all", "deny_all"}:
            bundle = task_context.get("bundle") or {}
            broken = bundle.get("broken_policy", {}).get("statements", [])
            actions: list[dict[str, Any]] = [
                {"tool": "delete_policy", "args": {"id": statement["id"]}}
                for statement in broken
            ]
            effect = "allow" if self.mode == "allow_all" else "deny"
            actions.append(
                {
                    "tool": "put_policy",
                    "args": {
                        "statement": {
                            "id": "s999",
                            "effect": effect,
                            "principals": ["*"],
                            "actions": ["*"],
                            "resources": ["*"],
                            "condition": None,
                            "exceptions": [],
                        }
                    },
                }
            )
            actions.append({"tool": "submit", "args": {}})
            self.actions = actions
        elif self.mode == "idle":
            self.actions = [{"tool": "submit", "args": {}}]
        else:
            raise ValueError(f"unknown mock mode: {self.mode}")

    def finish(self) -> None:
        return

    def complete(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
        *,
        timeout: float,
    ) -> Reply:
        del timeout
        if self.cursor >= len(self.actions):
            return _reply_from_actions([{"tool": "submit", "args": {}}])
        batch = self.actions[self.cursor : self.cursor + self.batch_size]
        self.cursor += len(batch)
        return _reply_from_actions(batch)


def make_client(
    model: str,
    api_key: str,
    base_url: str | None,
    max_tokens: int | None = None,
) -> OpenAIClient | MockClient:
    """Build the model client used for a run. Mock names need no API key."""
    if model.startswith("mock/"):
        mode = model.split("/", 1)[1]
        return MockClient(mode)
    if not api_key:
        raise ValueError(f"API_KEY is required for model {model!r}")
    return OpenAIClient(
        model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
    )


def oracle_actions(env: Any) -> list[dict[str, Any]]:
    """The reference-solution baseline for the current task and seed."""
    return env.oracle()


def noop_actions() -> list[dict[str, Any]]:
    """The no-op baseline: give up immediately by submitting the broken policy."""
    return [{"tool": "submit", "args": {}}]


def baseline_actions(bundle: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """Scripted actions for the no-op, allow-all and deny-all baselines."""
    if kind == "noop":
        return noop_actions()
    statements = bundle.get("broken_policy", {}).get("statements", [])
    actions: list[dict[str, Any]] = [
        {"tool": "delete_policy", "args": {"id": statement["id"]}}
        for statement in statements
    ]
    effect = "allow" if kind == "allow_all" else "deny"
    actions.append(
        {
            "tool": "put_policy",
            "args": {
                "statement": {
                    "id": "s999",
                    "effect": effect,
                    "principals": ["*"],
                    "actions": ["*"],
                    "resources": ["*"],
                    "condition": None,
                    "exceptions": [],
                }
            },
        }
    )
    actions.append({"tool": "submit", "args": {}})
    return actions
