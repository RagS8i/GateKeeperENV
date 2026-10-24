"""Small stdlib HTTP helpers shared by the services, harness and tests."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

MAX_BODY_BYTES = 1_000_000


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class JsonHandler(BaseHTTPRequestHandler):
    """Base handler with JSON parsing and consistent error responses."""

    server_version = "Gatekeeper/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _read_json(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if not length_header:
            return {}
        try:
            length = int(length_header)
        except ValueError:
            raise ValueError("invalid Content-Length") from None
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON body: {exc}") from None
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": "not found"})


def start_http_server(
    handler_class: type[JsonHandler],
    host: str = "0.0.0.0",
    port: int = 0,
) -> tuple[ThreadingHTTPServer, threading.Thread, int]:
    server = ThreadingHTTPServer((host, port), handler_class)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, int(server.server_address[1])


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HttpError(exc.code, body) from None
    except urllib.error.URLError as exc:
        raise ConnectionError(str(exc.reason)) from None
    except TimeoutError:
        raise TimeoutError(f"request to {url} timed out") from None
    if not raw:
        return {}
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("response body must be a JSON object")
    return decoded


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    return _request_json("POST", url, payload, timeout=timeout)


def get_json(url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    return _request_json("GET", url, None, timeout=timeout)


def wait_for_health(
    base_url: str,
    *,
    timeout: float = 60.0,
    interval: float = 0.25,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = get_json(f"{base_url.rstrip('/')}/health", timeout=2.0)
            if response.get("ok") is True:
                return
        except Exception as exc:  # pragma: no cover - retry branch
            last_error = exc
        time.sleep(interval)
    raise TimeoutError(
        f"{base_url} did not become healthy within {timeout}s"
        + (f": {last_error}" if last_error else "")
    )


def make_handler(
    routes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
) -> type[JsonHandler]:
    class RoutedHandler(JsonHandler):
        def do_POST(self) -> None:  # noqa: N802
            route = self.path.split("?", 1)[0]
            handler = routes.get(route)
            if handler is None:
                self._send_json(404, {"error": "not found"})
                return
            try:
                body = self._read_json()
                response = handler(body)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except KeyError as exc:
                self._send_json(400, {"error": f"missing field: {exc}"})
                return
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._send_json(200, response)

    return RoutedHandler
