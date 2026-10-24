"""Trusted verifier service.  Hidden answer keys stay inside this process."""

from __future__ import annotations

import os
from typing import Any

from common.http_util import make_handler, start_http_server
from .core import TaskService


class VerifierService:
    def __init__(self, tasks: TaskService | None = None) -> None:
        self.tasks = tasks or TaskService()

    def reset(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.tasks.reset(
            str(body["task_id"]), _seed(body)
        )

    def score(self, body: dict[str, Any]) -> dict[str, Any]:
        policy = body.get("policy")
        if not isinstance(policy, dict):
            raise ValueError("score requires a policy object")
        return self.tasks.score(str(body["task_id"]), policy, _seed(body))

    def shadow(self, body: dict[str, Any]) -> dict[str, Any]:
        policy = body.get("policy")
        if not isinstance(policy, dict):
            raise ValueError("shadow requires a policy object")
        return self.tasks.shadow(str(body["task_id"]), policy, _seed(body))

    def oracle(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.tasks.oracle(str(body["task_id"]), _seed(body))

    def list_tasks(self, _body: dict[str, Any]) -> dict[str, Any]:
        return self.tasks.list_tasks()


def _seed(body: dict[str, Any]) -> int | None:
    seed = body.get("seed")
    return None if seed is None else int(seed)


def main() -> None:
    service = VerifierService()
    routes = {
        "/reset": service.reset,
        "/score": service.score,
        "/shadow": service.shadow,
        "/oracle": service.oracle,
        "/tasks": service.list_tasks,
    }
    handler = make_handler(routes)
    port = int(os.environ.get("PORT", "8000"))
    server, thread, actual_port = start_http_server(handler, port=port)
    print(f"verifier-service listening on {actual_port}", flush=True)
    try:
        thread.join()
    except KeyboardInterrupt:  # pragma: no cover
        server.shutdown()


if __name__ == "__main__":
    main()
