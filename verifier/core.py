"""Trusted verifier logic: task generation, hidden scoring and oracle actions."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .generator import (
    GeneratedTask,
    TaskConfig,
    generate_task,
    load_task_configs,
)
from common.models import Policy


class TaskService:
    """Holds the hidden answer keys and never exposes them to the environment."""

    def __init__(
        self,
        configs: tuple[TaskConfig, ...] | None = None,
        *,
        tasks_path: Path | None = None,
    ) -> None:
        self.configs = configs or load_task_configs(tasks_path)
        self._by_id = {config.task_id: config for config in self.configs}
        self._cache: dict[tuple[str, int], GeneratedTask] = {}
        self._lock = threading.Lock()

    def reset(self, task_id: str, seed: int | None = None) -> dict[str, Any]:
        return self.get_task(task_id, seed).sanitized_bundle()

    def get_task(self, task_id: str, seed: int | None = None) -> GeneratedTask:
        config = self._config(task_id)
        effective_seed = config.seed if seed is None else int(seed)
        key = (task_id, effective_seed)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            task = generate_task(config, seed_override=effective_seed)
            self._cache[key] = task
            return task

    def _config(self, task_id: str) -> TaskConfig:
        try:
            return self._by_id[task_id]
        except KeyError:
            raise KeyError(f"unknown task: {task_id!r}") from None

    def score(
        self,
        task_id: str,
        policy_payload: dict[str, Any],
        seed: int | None = None,
    ) -> dict[str, Any]:
        policy = Policy.model_validate(policy_payload)
        return self.get_task(task_id, seed).score(policy)

    def shadow(
        self,
        task_id: str,
        policy_payload: dict[str, Any],
        seed: int | None = None,
    ) -> dict[str, Any]:
        policy = Policy.model_validate(policy_payload)
        return self.get_task(task_id, seed).shadow_score(policy)

    def oracle(self, task_id: str, seed: int | None = None) -> dict[str, Any]:
        task = self.get_task(task_id, seed)
        return {
            "task_id": task.task_id,
            "seed": task.seed,
            "actions": task.oracle_actions(),
        }

    def list_tasks(self) -> dict[str, Any]:
        return {
            "count": len(self.configs),
            "tasks": [
                {
                    "task_id": config.task_id,
                    "tier": config.tier,
                    "seed": config.seed,
                }
                for config in self.configs
            ]
        }
