"""Tiny OpenAI-compatible fake model used by offline end-to-end tests."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _native_call(name: str, arguments: dict[str, Any] | str, index: int) -> dict[str, Any]:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": raw},
    }


def _message_for(model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    turns = sum(1 for message in messages if message.get("role") == "assistant")
    submit_call = _native_call("submit", {}, 1)

    if model in {"fake/native_submit", "fake/submit"}:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [submit_call],
        }
    if model == "fake/text_submit":
        return {
            "role": "assistant",
            "content": '{"tool": "submit", "args": {}}',
        }
    if model == "fake/legacy_submit":
        return {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "submit", "arguments": "{}"},
        }
    if model == "fake/malformed":
        if turns == 0:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _native_call("put_policy", "{not valid json", 1)
                ],
            }
        return {"role": "assistant", "content": "", "tool_calls": [submit_call]}
    if model == "fake/25calls":
        if turns == 0:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _native_call("get_requirements", {}, index)
                    for index in range(1, 26)
                ],
            }
        return {"role": "assistant", "content": "", "tool_calls": [submit_call]}
    if model == "fake/multi":
        if turns == 0:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _native_call("get_requirements", {}, index)
                    for index in range(1, 4)
                ],
            }
        return {"role": "assistant", "content": "", "tool_calls": [submit_call]}
    if model == "fake/text_multi":
        if turns == 0:
            return {
                "role": "assistant",
                "content": (
                    "```json\n"
                    '[{"tool":"get_requirements","args":{}},'
                    '{"tool":"list_policies","args":{"page":1}}]\n'
                    "```"
                ),
            }
        return {"role": "assistant", "content": '{"tool":"submit","args":{}}'}
    if model == "fake/allow_all":
        if turns == 0:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _native_call(
                        "put_policy",
                        {
                            "statement": {
                                "id": "s999",
                                "effect": "allow",
                                "principals": ["*"],
                                "actions": ["*"],
                                "resources": ["*"],
                                "condition": None,
                                "exceptions": [],
                            }
                        },
                        1,
                    )
                ],
            }
        return {"role": "assistant", "content": "", "tool_calls": [submit_call]}
    return {"role": "assistant", "content": "I am thinking about the policy."}


class FakeModelHandler(BaseHTTPRequestHandler):
    server_version = "FakeModel/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] in {"/health", "/v1/health"}:
            self._send(200, {"ok": True})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route not in {"/v1/chat/completions", "/chat/completions"}:
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            model = str(body.get("model", ""))
            messages = body.get("messages") or []
            message = _message_for(model, messages)
        except Exception as exc:
            self._send(400, {"error": str(exc)})
            return
        self._send(
            200,
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls"
                        if message.get("tool_calls") or message.get("function_call")
                        else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )


def start_fake_model(
    port: int = 0,
    host: str = "127.0.0.1",
) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer((host, port), FakeModelHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://{host}:{server.server_address[1]}/v1"


def main() -> None:
    port = int(os.environ.get("PORT", "9000"))
    server, thread, url = start_fake_model(port=port, host="0.0.0.0")
    print(f"fake-model listening on {url}", flush=True)
    try:
        thread.join()
    except KeyboardInterrupt:  # pragma: no cover
        server.shutdown()


if __name__ == "__main__":
    main()
