"""Task configuration: ids, tiers, dimensions and seeds.

The generation logic lives in ``verifier/generator.py``. This module only owns
the data contract, so the harness and the verifier can both read ``tasks.json``
without importing the hidden generator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MUTATION_KINDS: tuple[str, ...] = (
    "drop_statement",
    "widen_wildcard",
    "remove_condition",
    "remove_deny",
    "change_group",
    "flip_effect",
    "remove_exception",
    "swap_condition",
)

CONTEXTS_BY_TIER: dict[str, tuple[dict[str, str], ...]] = {
    "easy": (
        {},
        {"network": "corp"},
        {"network": "guest"},
        {"mfa": "true"},
        {"network": "corp", "mfa": "true"},
    ),
    "medium": (
        {},
        {"network": "corp"},
        {"network": "vpn"},
        {"network": "guest"},
        {"mfa": "true"},
        {"network": "corp", "mfa": "true"},
    ),
    "hard": (
        {},
        {"network": "corp"},
        {"network": "vpn"},
        {"network": "guest"},
        {"mfa": "true"},
        {"mfa": "false"},
        {"device": "managed"},
        {"device": "unmanaged"},
        {"network": "vpn", "mfa": "false"},
        {"network": "corp", "mfa": "true", "device": "managed"},
        {"network": "guest", "mfa": "false", "device": "unmanaged"},
    ),
}

TASKS_PATH = Path(__file__).resolve().parent.parent / "tasks.json"
DEMO_TASK = "med_01"


@dataclass(frozen=True, slots=True)
class TaskConfig:
    task_id: str
    tier: str
    seed: int
    users: int
    groups: int
    resources: int
    statements: int
    mutation_types: tuple[str, ...]
    visible_examples: int = 6

    def __post_init__(self) -> None:
        if self.tier not in CONTEXTS_BY_TIER:
            raise ValueError(f"unknown task tier: {self.tier!r}")
        if min(self.users, self.groups, self.resources, self.statements) <= 0:
            raise ValueError("task dimensions must be positive")
        if self.groups < 2:
            raise ValueError("tasks need at least two groups")
        if self.statements < len(self.mutation_types) + 2:
            raise ValueError("not enough statements for the requested mutations")
        unknown = set(self.mutation_types) - set(MUTATION_KINDS)
        if unknown:
            raise ValueError(f"unknown mutation types: {sorted(unknown)}")
        if not self.mutation_types:
            raise ValueError("each task needs at least one mutation")
        if not 3 <= self.visible_examples <= 12:
            raise ValueError("visible_examples must be between 3 and 12")

    @property
    def contexts(self) -> tuple[dict[str, str], ...]:
        return CONTEXTS_BY_TIER[self.tier]


def load_task_configs(path: Path | None = None) -> tuple[TaskConfig, ...]:
    source = path or TASKS_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError(f"{source} must contain a non-empty 'tasks' list")
    configs: list[TaskConfig] = []
    for raw in raw_tasks:
        configs.append(
            TaskConfig(
                task_id=str(raw["task_id"]),
                tier=str(raw["tier"]),
                seed=int(raw["seed"]),
                users=int(raw["users"]),
                groups=int(raw["groups"]),
                resources=int(raw["resources"]),
                statements=int(raw["statements"]),
                mutation_types=tuple(str(item) for item in raw["mutation_types"]),
                visible_examples=int(raw.get("visible_examples", 6)),
            )
        )
    ids = [config.task_id for config in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("task ids must be unique")
    return tuple(configs)
