"""Lenient parsing of native and text-embedded tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str
    call_id: str
    parse_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.parse_error is None and bool(self.name)


@dataclass(slots=True)
class Reply:
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None


def _to_argument_string(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    if hasattr(content, "model_dump"):
        try:
            return _content_to_text(content.model_dump())
        except Exception:
            pass
    if isinstance(content, (list, tuple)):
        parts = [
            part
            for part in (_content_to_text(item) for item in content)
            if part
        ]
        return "\n".join(parts)
    if isinstance(content, dict):
        block_type = content.get("type")
        if block_type not in (None, "text", "input_text", "output_text"):
            return ""
        for key in ("text", "content", "value"):
            if key not in content:
                continue
            value = content[key]
            if isinstance(value, str):
                if value:
                    return value
            elif isinstance(value, (dict, list, tuple)):
                text = _content_to_text(value)
                if text:
                    return text
            elif value is not None:
                return str(value)
        return ""
    return str(content)


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str, str | None]:
    text = _to_argument_string(raw).strip()
    if not text:
        return {}, "", None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, text, f"arguments are not valid JSON: {exc.msg}"
    if not isinstance(value, dict):
        return None, text, "arguments must be a JSON object"
    return value, text, None


def tool_call_from_mapping(
    value: dict[str, Any],
    *,
    fallback_id: str,
) -> ToolCall | None:
    call_id = str(value.get("id") or fallback_id)
    function = value.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    elif isinstance(function, str):
        name = function
        arguments = value.get("arguments", {})
    else:
        name = value.get("name") or value.get("tool")
        arguments = value.get("arguments", value.get("args", value.get("parameters", {})))
        if isinstance(name, dict):
            arguments = name.get("arguments", name.get("args", arguments))
            name = name.get("name") or name.get("tool")
    if not isinstance(name, str) or not name.strip():
        return None
    parsed, raw_text, error = parse_arguments(arguments)
    return ToolCall(
        name=name.strip(),
        arguments=parsed,
        raw_arguments=raw_text,
        call_id=call_id,
        parse_error=error,
    )


def _extract_json_spans(text: str) -> list[tuple[int, int, Any]]:
    values: list[tuple[int, int, Any]] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if start is None:
            if char in "{[":
                start = index
                depth = 1
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    parsed = None
                if parsed is not None:
                    values.append((start, index + 1, parsed))
                start = None
                depth = 0

    values.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    return values


def _calls_from_value(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, list):
        candidates = [item for item in value if isinstance(item, dict)]
        if candidates and all(_looks_like_call(item) for item in candidates):
            return candidates
        return None
    if isinstance(value, dict):
        if "tool_calls" in value and isinstance(value["tool_calls"], list):
            candidates = [
                item for item in value["tool_calls"] if isinstance(item, dict)
            ]
            if candidates and all(_looks_like_call(item) for item in candidates):
                return candidates
        if _looks_like_call(value):
            return [value]
    return None


def _looks_like_call(value: dict[str, Any]) -> bool:
    if "tool_calls" in value:
        return False
    if isinstance(value.get("function"), (dict, str)):
        return True
    name = value.get("name") or value.get("tool")
    if isinstance(name, dict):
        return bool(name.get("name") or name.get("tool"))
    return isinstance(name, str) and bool(name.strip())


def parse_text_tool_calls(text: Any) -> list[ToolCall]:
    text = _content_to_text(text)
    if not text or not text.strip():
        return []

    spans = _extract_json_spans(text)
    accepted_spans: list[tuple[int, int]] = []
    calls: list[ToolCall] = []
    for index, (start, end, value) in enumerate(spans):
        if any(outer_start <= start and end <= outer_end for outer_start, outer_end in accepted_spans):
            continue
        candidates = _calls_from_value(value)
        if not candidates:
            continue
        accepted_spans.append((start, end))
        for position, candidate in enumerate(candidates):
            call = tool_call_from_mapping(
                candidate,
                fallback_id=f"text_call_{index}_{position}",
            )
            if call is not None:
                calls.append(call)
    return calls


def parse_assistant_message(message: Any) -> Reply:
    text = getattr(message, "content", None)
    if text is None and isinstance(message, dict):
        text = message.get("content")
    text = _content_to_text(text)

    raw_calls = getattr(message, "tool_calls", None)
    if raw_calls is None and isinstance(message, dict):
        raw_calls = message.get("tool_calls")
    calls: list[ToolCall] = []
    if raw_calls:
        for position, raw_call in enumerate(raw_calls):
            if hasattr(raw_call, "model_dump"):
                call_mapping = raw_call.model_dump()
            elif isinstance(raw_call, dict):
                call_mapping = raw_call
            else:
                call_mapping = {
                    "id": getattr(raw_call, "id", f"native_{position}"),
                    "function": {
                        "name": getattr(
                            getattr(raw_call, "function", None), "name", ""
                        ),
                        "arguments": getattr(
                            getattr(raw_call, "function", None), "arguments", "{}"
                        ),
                    },
                }
            call_mapping = dict(call_mapping)
            call_mapping.setdefault("id", f"native_{position}")
            call = tool_call_from_mapping(call_mapping, fallback_id=f"native_{position}")
            if call is not None:
                calls.append(call)

    if not calls:
        legacy = getattr(message, "function_call", None)
        if legacy is None and isinstance(message, dict):
            legacy = message.get("function_call")
        if legacy:
            if hasattr(legacy, "model_dump"):
                legacy = legacy.model_dump()
            if isinstance(legacy, dict):
                call = tool_call_from_mapping(
                    {"function": legacy, "id": "legacy_0"},
                    fallback_id="legacy_0",
                )
                if call is not None:
                    calls.append(call)

    if not calls:
        calls = parse_text_tool_calls(text)

    assistant_message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        assistant_message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments or "{}",
                },
            }
            for call in calls
        ]
    return Reply(text=text, calls=calls, assistant_message=assistant_message)


def extract_text_tool_calls(text: str) -> list[ToolCall]:
    return parse_text_tool_calls(text)
