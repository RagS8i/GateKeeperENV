"""Clients for the environment and verifier HTTP services."""

from __future__ import annotations

from typing import Any

from ..http_util import get_json, post_json


class EnvClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self, timeout: float = 2.0) -> dict[str, Any]:
        return get_json(f"{self.base_url}/health", timeout=timeout)

    def reset(self, bundle: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
        return post_json(f"{self.base_url}/reset", bundle, timeout=timeout)

    def tool(
        self,
        name: str,
        args: dict[str, Any] | None,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        return post_json(
            f"{self.base_url}/tool",
            {"tool": name, "args": args or {}},
            timeout=timeout,
        )

    def state(self, *, timeout: float = 10.0) -> dict[str, Any]:
        return post_json(f"{self.base_url}/state", {}, timeout=timeout)


class VerifierClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self, timeout: float = 2.0) -> dict[str, Any]:
        return get_json(f"{self.base_url}/health", timeout=timeout)

    def reset(
        self, task_id: str, seed: int | None = None, *, timeout: float = 60.0
    ) -> dict[str, Any]:
        return post_json(
            f"{self.base_url}/reset",
            {"task_id": task_id, "seed": seed},
            timeout=timeout,
        )

    def score(
        self,
        task_id: str,
        policy: dict[str, Any],
        seed: int | None = None,
        *,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        return post_json(
            f"{self.base_url}/score",
            {"task_id": task_id, "seed": seed, "policy": policy},
            timeout=timeout,
        )

    def shadow(
        self,
        task_id: str,
        policy: dict[str, Any],
        seed: int | None = None,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return post_json(
            f"{self.base_url}/shadow",
            {"task_id": task_id, "seed": seed, "policy": policy},
            timeout=timeout,
        )

    def oracle(
        self, task_id: str, seed: int | None = None, *, timeout: float = 60.0
    ) -> dict[str, Any]:
        return post_json(
            f"{self.base_url}/oracle",
            {"task_id": task_id, "seed": seed},
            timeout=timeout,
        )

    def tasks(self, *, timeout: float = 10.0) -> list[dict[str, Any]]:
        response = post_json(f"{self.base_url}/tasks", {}, timeout=timeout)
        return list(response.get("tasks", []))
