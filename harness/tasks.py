"""Task configuration for the harness.

The task data lives in ``tasks.json`` and is loaded by ``common.tasks``.  This
module is the harness-facing import point, mirroring the template's
``harness/tasks.py`` while keeping the generator in the trusted verifier.
"""

from __future__ import annotations

from common.tasks import (
    CONTEXTS_BY_TIER,
    DEMO_TASK,
    MUTATION_KINDS,
    TASKS_PATH,
    TaskConfig,
    load_task_configs,
)

TASKS = {config.task_id: config for config in load_task_configs()}

__all__ = [
    "CONTEXTS_BY_TIER",
    "DEMO_TASK",
    "MUTATION_KINDS",
    "TASKS",
    "TASKS_PATH",
    "TaskConfig",
    "load_task_configs",
]
