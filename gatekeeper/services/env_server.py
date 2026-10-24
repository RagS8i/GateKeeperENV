"""Environment service.  It only ever sees the sanitized task bundle."""

from __future__ import annotations

import os
import threading
from typing import Any

from ..env_core import Environment
from ..http_util import make_handler, start_http_server


class EnvService:
    def __init__(self) -> None:
        self._env: Environment | None = None
        self._lock = threading.Lock()

    def reset(self, bundle: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._env = Environment(bundle)
            return {
                "observation": self._env.observation(),
                "state": self._env.state(),
            }

    def tool(self, tool: str, args: dict[str, Any] | None) -> dict[str, Any]:
        with self._lock:
            if self._env is None:
                raise ValueError("environment has not been reset")
            result = self._env.step(tool, args)
            result["state_hash"] = self._env.state_hash
            result["submitted"] = self._env.submitted
            result["statement_count"] = len(self._env.current_policy.statements)
            return result

    def state(self) -> dict[str, Any]:
        with self._lock:
            if self._env is None:
                raise ValueError("environment has not been reset")
            return self._env.state()


def build_service() -> EnvService:
    return EnvService()


def main() -> None:
    service = build_service()
    routes = {
        "/reset": lambda body: service.reset(body),
        "/tool": lambda body: service.tool(
            str(body.get("tool", "")), body.get("args")
        ),
        "/state": lambda _body: service.state(),
    }
    handler = make_handler(routes)
    port = int(os.environ.get("PORT", "8000"))
    server, thread, actual_port = start_http_server(handler, port=port)
    print(f"env-service listening on {actual_port}", flush=True)
    try:
        thread.join()
    except KeyboardInterrupt:  # pragma: no cover
        server.shutdown()


if __name__ == "__main__":
    main()
