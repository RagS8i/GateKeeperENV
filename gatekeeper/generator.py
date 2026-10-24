"""Deterministic task generator for Gatekeeper.

The generator creates a coherent world, a correct access policy ``P*`` and a
broken policy produced by a small number of semantic mutations.  The broken
policy, the world, the rendered requirements and a few visible examples are
safe to hand to the environment.  Everything else (``P*``, mutations, the
hidden decision matrix and the scoring weights) stays in the verifier.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .matrix import HiddenMatrix
from .models import (
    ALL_ACTIONS,
    Group,
    Policy,
    Resource,
    Statement,
    User,
    World,
    canonical_statement,
    state_hash,
)
from .requirements import render_requirements

MUTATION_KINDS: tuple[str, ...] = (
    "drop_statement",
    "widen_wildcard",
    "remove_condition",
    "remove_deny",
    "change_group",
    "flip_effect",
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
        {"network": "vpn", "mfa": "false"},
    ),
}

CONDITION_POOL: tuple[tuple[str, str], ...] = (
    ("network", "corp"),
    ("network", "vpn"),
    ("network", "guest"),
    ("mfa", "true"),
    ("mfa", "false"),
)

TASKS_PATH = Path(__file__).resolve().parent.parent / "tasks.json"
DEMO_TASK = "med_01"


class GenerationError(RuntimeError):
    """The generator could not produce a valid task for a seed."""


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


@dataclass(frozen=True, slots=True)
class MutationRecord:
    kind: str
    statement_id: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    sample_impact: int
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "statement_id": self.statement_id,
            "before": self.before,
            "after": self.after,
            "sample_impact": self.sample_impact,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class GeneratedTask:
    config: TaskConfig
    seed: int
    world: World
    correct_policy: Policy
    broken_policy: Policy
    requirements: str
    visible_examples: list[dict[str, object]]
    matrix: HiddenMatrix
    reference_decisions: bytes
    e_start: int
    sample_indices: tuple[int, ...]
    reference_sample: bytes
    broken_sample: bytes
    shadow_indices: tuple[int, ...]
    shadow_e_start: int
    mutations: tuple[MutationRecord, ...]

    @property
    def task_id(self) -> str:
        return self.config.task_id

    @property
    def tier(self) -> str:
        return self.config.tier

    @property
    def state_hash(self) -> str:
        return state_hash(self.world, self.broken_policy)

    def sanitized_bundle(self) -> dict[str, Any]:
        """The only task data the environment is allowed to receive."""

        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "seed": self.seed,
            "world": self.world.model_dump(mode="json"),
            "broken_policy": self.broken_policy.model_dump(mode="json"),
            "requirements": self.requirements,
            "visible_examples": self.visible_examples,
            "state_hash": self.state_hash,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "seed": self.seed,
            "total_requests": self.matrix.total,
            "statement_count": len(self.broken_policy.statements),
            "state_hash": self.state_hash,
        }

    def score(self, policy: Policy) -> dict[str, Any]:
        candidate = self.matrix.decisions(policy)
        false_allows, false_denies = HiddenMatrix.error_counts(
            candidate, self.reference_decisions
        )
        return {
            "task_id": self.task_id,
            "score": HiddenMatrix.score_from_errors(
                false_allows, false_denies, self.e_start
            ),
            "false_allows": false_allows,
            "false_denies": false_denies,
            "weighted_errors": HiddenMatrix.weighted_errors(
                false_allows, false_denies
            ),
            "e_start": self.e_start,
            "total_requests": self.matrix.total,
        }

    def shadow_score(self, policy: Policy) -> dict[str, Any]:
        candidate = self.matrix.decisions_at(self.shadow_indices, policy)
        result = self.matrix.shadow_score(
            self.shadow_indices,
            candidate,
            self.reference_decisions,
            self.shadow_e_start,
        )
        result["task_id"] = self.task_id
        result["sample_size"] = len(self.shadow_indices)
        return result

    def oracle_actions(self) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for statement in sorted(self.broken_policy.statements, key=lambda item: item.id):
            actions.append(
                {"tool": "delete_policy", "args": {"id": statement.id}}
            )
        for statement in sorted(
            self.correct_policy.statements, key=lambda item: item.id
        ):
            actions.append(
                {
                    "tool": "put_policy",
                    "args": {"statement": statement.model_dump(mode="json")},
                }
            )
        actions.append({"tool": "submit", "args": {}})
        return actions


def _statement_key(statement: Statement) -> str:
    payload = canonical_statement(statement)
    payload.pop("id")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _rebuild(statement: Statement, **updates: Any) -> Statement:
    payload = statement.model_dump()
    payload.update(updates)
    return Statement(**payload)


def _replace_statement(policy: Policy, statement: Statement) -> Policy:
    return Policy(
        statements=[
            statement if existing.id == statement.id else existing
            for existing in policy.statements
        ]
    )


def _remove_statement(policy: Policy, statement_id: str) -> Policy:
    return Policy(
        statements=[
            existing
            for existing in policy.statements
            if existing.id != statement_id
        ]
    )


def _build_world(rng: random.Random, config: TaskConfig) -> World:
    user_ids = [f"u{number:03d}" for number in range(1, config.users + 1)]
    group_ids = [f"g{number:02d}" for number in range(1, config.groups + 1)]
    resource_ids = [
        f"r{number:03d}" for number in range(1, config.resources + 1)
    ]

    members: dict[str, set[str]] = {group_id: set() for group_id in group_ids}
    ungrouped_probability = 0.08 if config.tier != "easy" else 0.03
    for user_id in user_ids:
        if rng.random() < ungrouped_probability:
            continue
        count = 1 if rng.random() < 0.82 else 2
        chosen = rng.sample(group_ids, min(count, len(group_ids)))
        for group_id in chosen:
            members[group_id].add(f"user:{user_id}")
    for group_id in group_ids:
        if not members[group_id]:
            members[group_id].add(f"user:{rng.choice(user_ids)}")

    nested_probability = {"easy": 0.20, "medium": 0.38, "hard": 0.50}[
        config.tier
    ]
    for index in range(2, len(group_ids)):
        if rng.random() < nested_probability:
            parent = group_ids[rng.randrange(0, index)]
            members[group_ids[index]].add(f"group:{parent}")

    groups = {
        group_id: Group(id=group_id, members=sorted(members[group_id]))
        for group_id in group_ids
    }

    departments = ("eng", "sales", "finance", "ops", "legal")
    sensitivities = ("public", "internal", "secret")
    projects = ("atlas", "beacon", "cobalt", "delta")
    all_tags = [
        *(f"dept:{value}" for value in departments),
        *(f"sens:{value}" for value in sensitivities),
        *(f"project:{value}" for value in projects),
    ]

    resource_tags: dict[str, list[str]] = {}
    for index, resource_id in enumerate(resource_ids):
        if index < len(all_tags):
            tags = [all_tags[index]]
            if rng.random() < 0.45:
                extra = rng.choice(all_tags)
                if extra not in tags:
                    tags.append(extra)
        else:
            tags = rng.sample(all_tags, rng.randint(1, 2))
        if index % 4 == 0 and "scope:restricted" not in tags:
            tags.append("scope:restricted")
        resource_tags[resource_id] = sorted(tags)

    resources = {
        resource_id: Resource(id=resource_id, tags=resource_tags[resource_id])
        for resource_id in resource_ids
    }

    return World(
        users={user_id: User(id=user_id) for user_id in user_ids},
        groups=groups,
        resources=resources,
        actions=ALL_ACTIONS,
    )


def _principal_selectors(
    rng: random.Random,
    world: World,
    *,
    tier: str,
    force_group: bool = False,
) -> list[str]:
    if force_group:
        group_id = rng.choice(sorted(world.groups))
        return [f"group:{group_id}"]
    if tier == "hard" and rng.random() < 0.08:
        return ["*"]

    user_ids = sorted(world.users)
    group_ids = sorted(world.groups)
    count = 1 if rng.random() < 0.65 else 2
    chosen: list[str] = []
    while len(chosen) < count:
        if group_ids and (not user_ids or rng.random() < 0.58):
            selector = f"group:{rng.choice(group_ids)}"
        else:
            selector = f"user:{rng.choice(user_ids)}"
        if selector not in chosen:
            chosen.append(selector)
    return chosen


def _resource_selectors(
    rng: random.Random,
    world: World,
    *,
    tier: str,
    force_specific: bool = False,
) -> list[str]:
    resource_ids = sorted(world.resources)
    if force_specific:
        return [f"resource:{rng.choice(resource_ids)}"]
    if tier == "hard" and rng.random() < 0.08:
        return ["*"]

    tags = sorted(
        {
            tag
            for resource in world.resources.values()
            for tag in resource.tags
        }
    )
    count = rng.randint(1, 3)
    chosen: list[str] = []
    while len(chosen) < count:
        if tags and rng.random() < 0.38:
            selector = f"tag:{rng.choice(tags)}"
        else:
            selector = f"resource:{rng.choice(resource_ids)}"
        if selector not in chosen:
            chosen.append(selector)
    return chosen


def _action_selectors(rng: random.Random, *, tier: str) -> list[str]:
    if tier == "hard" and rng.random() < 0.07:
        return ["*"]
    count = 1 if rng.random() < 0.72 else 2
    return rng.sample(list(ALL_ACTIONS), count)


def _condition(rng: random.Random, *, force: bool, tier: str) -> dict[str, str] | None:
    probability = {"easy": 0.16, "medium": 0.36, "hard": 0.50}[tier]
    if not force and rng.random() >= probability:
        return None
    count = 1 if rng.random() < 0.60 else 2
    entries: dict[str, str] = {}
    while len(entries) < count:
        key, value = rng.choice(CONDITION_POOL)
        if key not in entries:
            entries[key] = value
    return entries


def _exceptions(
    rng: random.Random,
    world: World,
    principals: list[str],
    *,
    tier: str,
) -> list[str]:
    probability = {"easy": 0.0, "medium": 0.16, "hard": 0.24}[tier]
    if rng.random() >= probability:
        return []
    principal_set = set(principals)
    candidates = [
        f"user:{user_id}"
        for user_id in sorted(world.users)
        if f"user:{user_id}" not in principal_set
    ] + [
        f"group:{group_id}"
        for group_id in sorted(world.groups)
        if f"group:{group_id}" not in principal_set
    ]
    if not candidates:
        return []
    count = 1 if rng.random() < 0.8 else 2
    return rng.sample(candidates, min(count, len(candidates)))


def _make_statement(
    rng: random.Random,
    world: World,
    *,
    forced_kind: str | None,
    tier: str,
    serial: int,
    used_keys: set[str],
    safe_resources: tuple[str, ...],
) -> Statement | None:
    broad_forced = forced_kind in {
        "remove_condition",
        "remove_deny",
        "change_group",
        "flip_effect",
        "drop_statement",
    }

    for _ in range(30):
        if forced_kind == "widen_wildcard":
            principals = [f"user:{rng.choice(sorted(world.users))}"]
            actions = [rng.choice(ALL_ACTIONS)]
            resources = [
                f"resource:{rng.choice(safe_resources)}"
            ]
            condition = None
            exceptions: list[str] = []
        elif broad_forced:
            principals = [
                f"group:{rng.choice(sorted(world.groups))}"
            ]
            actions = rng.sample(list(ALL_ACTIONS), 2)
            resource_ids = rng.sample(
                safe_resources, min(2, len(safe_resources))
            )
            resources = [f"resource:{resource_id}" for resource_id in resource_ids]
            condition = (
                _condition(rng, force=True, tier=tier)
                if forced_kind == "remove_condition"
                else None
            )
            exceptions = []
        else:
            principals = _principal_selectors(rng, world, tier=tier)
            actions = _action_selectors(rng, tier=tier)
            resources = _resource_selectors(rng, world, tier=tier)
            condition = _condition(rng, force=False, tier=tier)
            exceptions = _exceptions(rng, world, principals, tier=tier)

        statement = Statement(
            id=f"s{serial:03d}",
            effect="deny",
            principals=principals,
            actions=actions,
            resources=resources,
            condition=condition,
            exceptions=exceptions,
        )
        key = _statement_key(statement)
        if key not in used_keys:
            used_keys.add(key)
            return statement
    return None


def _ensure_effects(
    statements: list[Statement],
    forced_types: list[str | None],
) -> list[Statement]:
    statements = list(statements)
    if not any(statement.effect == "deny" for statement in statements):
        statements[0] = _rebuild(statements[0], effect="deny")

    if not any(statement.effect == "allow" for statement in statements):
        candidates = [
            index
            for index, statement in enumerate(statements)
            if statement.effect == "deny" and forced_types[index] != "remove_deny"
        ]
        if candidates:
            index = candidates[0]
        else:
            index = 1 if len(statements) > 1 else 0
        statements[index] = _rebuild(statements[index], effect="allow")
    return statements


def _build_correct_policy(
    rng: random.Random,
    world: World,
    config: TaskConfig,
) -> tuple[Policy, dict[str, list[str]], set[str]] | None:
    safe_resources = tuple(
        resource_id
        for resource_id in sorted(world.resources)
        if "scope:restricted"
        not in world.resources[resource_id].tags
    )
    if not safe_resources:
        return None

    # A broad allow baseline plus a broad protected deny gives every task a
    # realistic deny-list shape and guarantees that both trivial policies and
    # an empty policy score badly.
    statements: list[Statement] = [
        Statement(
            id="s001",
            effect="allow",
            principals=["*"],
            actions=["*"],
            resources=["*"],
        ),
        Statement(
            id="s002",
            effect="deny",
            principals=["*"],
            actions=["*"],
            resources=["tag:scope:restricted"],
        ),
    ]
    protected_ids = {"s001", "s002"}
    forced_types: list[str | None] = [None, None]
    used_keys = {_statement_key(statement) for statement in statements}

    feature_targets: dict[str, list[str]] = {}
    for index, forced_kind in enumerate(config.mutation_types):
        statement = _make_statement(
            rng,
            world,
            forced_kind=forced_kind,
            tier=config.tier,
            serial=index + 3,
            used_keys=used_keys,
            safe_resources=safe_resources,
        )
        if statement is None:
            return None
        statements.append(statement)
        forced_types.append(forced_kind)
        feature_targets.setdefault(forced_kind, []).append(statement.id)

    if len(statements) > config.statements:
        return None
    while len(statements) < config.statements:
        index = len(statements)
        statement = _make_statement(
            rng,
            world,
            forced_kind=None,
            tier=config.tier,
            serial=index + 1,
            used_keys=used_keys,
            safe_resources=safe_resources,
        )
        if statement is None:
            return None
        statements.append(statement)
        forced_types.append(None)

    statements = _ensure_effects(statements, forced_types)

    keys: set[str] = set()
    for statement in statements:
        key = _statement_key(statement)
        if key in keys:
            return None
        keys.add(key)

    return Policy(statements=statements), feature_targets, protected_ids


def _supports_mutation(kind: str, statement: Statement, world: World) -> bool:
    if kind == "drop_statement":
        return True
    if kind == "remove_deny":
        return statement.effect == "deny"
    if kind == "remove_condition":
        return bool(statement.condition)
    if kind == "flip_effect":
        return True
    if kind == "widen_wildcard":
        return any(resource != "*" for resource in statement.resources) or any(
            action != "*" for action in statement.actions
        )
    if kind == "change_group":
        return len(world.groups) >= 2 and any(
            principal.startswith("group:") for principal in statement.principals
        )
    return False


def _mutate_statement(
    rng: random.Random,
    world: World,
    statement: Statement,
    kind: str,
) -> Statement | None:
    if kind == "remove_condition":
        return _rebuild(statement, condition=None)
    if kind == "flip_effect":
        effect = "deny" if statement.effect == "allow" else "allow"
        return _rebuild(statement, effect=effect)
    if kind == "widen_wildcard":
        if any(resource != "*" for resource in statement.resources):
            return _rebuild(statement, resources=["*"])
        if any(action != "*" for action in statement.actions):
            return _rebuild(statement, actions=["*"])
        return None
    if kind == "change_group":
        group_principals = [
            principal
            for principal in statement.principals
            if principal.startswith("group:")
        ]
        if not group_principals or len(world.groups) < 2:
            return None
        old_selector = rng.choice(group_principals)
        old_group = old_selector.removeprefix("group:")
        alternatives = [
            group_id
            for group_id in sorted(world.groups)
            if group_id != old_group
            and f"group:{group_id}" not in statement.principals
        ]
        if not alternatives:
            return None
        replacement = f"group:{rng.choice(alternatives)}"
        principals = [
            replacement if principal == old_selector else principal
            for principal in statement.principals
        ]
        return _rebuild(statement, principals=principals)
    return None


def _apply_mutations(
    rng: random.Random,
    world: World,
    matrix: HiddenMatrix,
    correct_policy: Policy,
    feature_targets: dict[str, list[str]],
    protected_ids: set[str],
    mutation_types: Iterable[str],
    sample_indices: tuple[int, ...],
    reference_sample: bytes,
) -> tuple[Policy, list[MutationRecord]] | None:
    current = correct_policy
    current_sample = reference_sample
    used_ids: set[str] = set()
    records: list[MutationRecord] = []

    ordered_types = [
        kind for kind in mutation_types if kind not in {"drop_statement", "remove_deny"}
    ] + [
        kind for kind in mutation_types if kind == "drop_statement"
    ] + [
        kind for kind in mutation_types if kind == "remove_deny"
    ]

    min_impact = max(1, len(sample_indices) // 1000)
    for kind in ordered_types:
        dedicated = [
            statement_id
            for statement_id in feature_targets.get(kind, [])
            if statement_id not in used_ids
        ]
        others = [
            statement.id
            for statement in current.statements
            if statement.id not in used_ids
            and statement.id not in protected_ids
            and statement.id not in dedicated
            and _supports_mutation(kind, statement, world)
        ]
        rng.shuffle(others)
        candidates = dedicated + others

        accepted = False
        for statement_id in candidates:
            statement = next(
                (
                    item
                    for item in current.statements
                    if item.id == statement_id
                ),
                None,
            )
            if statement is None or not _supports_mutation(kind, statement, world):
                continue

            if kind in {"drop_statement", "remove_deny"}:
                new_policy = _remove_statement(current, statement.id)
                mutated = None
            else:
                mutated = _mutate_statement(rng, world, statement, kind)
                if mutated is None:
                    continue
                new_policy = _replace_statement(current, mutated)

            new_sample = matrix.decisions_at(sample_indices, new_policy)
            impact = sum(
                1
                for before, after in zip(current_sample, new_sample)
                if before != after
            )
            if impact < min_impact:
                continue

            records.append(
                MutationRecord(
                    kind=kind,
                    statement_id=statement.id,
                    before=canonical_statement(statement),
                    after=canonical_statement(mutated) if mutated is not None else None,
                    sample_impact=impact,
                    explanation=f"{kind} on statement {statement.id}",
                )
            )
            current = new_policy
            current_sample = new_sample
            used_ids.add(statement.id)
            accepted = True
            break

        if not accepted:
            return None

    return current, records


def generate_task(
    config: TaskConfig,
    *,
    seed_override: int | None = None,
    max_attempts: int = 30,
) -> GeneratedTask:
    """Generate one task deterministically for ``config.seed``."""

    seed = config.seed if seed_override is None else int(seed_override)
    rng = random.Random(seed)
    last_reason = "no attempt completed"

    for _ in range(max_attempts):
        world = _build_world(rng, config)
        built = _build_correct_policy(rng, world, config)
        if built is None:
            last_reason = "could not build a sufficiently diverse correct policy"
            continue
        correct_policy, feature_targets, protected_ids = built

        matrix = HiddenMatrix.build(world, config.contexts)
        reference = matrix.decisions(correct_policy)
        allow_count, deny_count = matrix.counts(reference)
        if allow_count == 0 or deny_count == 0:
            last_reason = "correct policy does not exercise both allow and deny"
            continue
        if min(allow_count, deny_count) < matrix.total // 4:
            last_reason = "correct policy decisions are too imbalanced"
            continue

        sample_indices = matrix.sample_indices(
            min(4000, matrix.total), rng
        )
        reference_sample = matrix.decisions_at(sample_indices, correct_policy)
        mutated = _apply_mutations(
            rng,
            world,
            matrix,
            correct_policy,
            feature_targets,
            protected_ids,
            config.mutation_types,
            sample_indices,
            reference_sample,
        )
        if mutated is None:
            last_reason = "could not apply every mutation with meaningful impact"
            continue
        broken_policy, records = mutated

        broken = matrix.decisions(broken_policy)
        false_allows, false_denies = HiddenMatrix.error_counts(broken, reference)
        e_start = HiddenMatrix.weighted_errors(false_allows, false_denies)
        floor = max(20, matrix.total // 500)
        if e_start < floor:
            last_reason = f"E_start {e_start} is below floor {floor}"
            continue

        allow_all_weighted = 3 * deny_count
        deny_all_weighted = allow_count
        if allow_all_weighted < e_start or deny_all_weighted < e_start:
            last_reason = "allow-all or deny-all would not score zero"
            continue

        broken_sample = matrix.decisions_at(sample_indices, broken_policy)
        sample_false_allows, sample_false_denies = HiddenMatrix.error_counts(
            broken_sample, reference_sample
        )
        sample_e_start = HiddenMatrix.weighted_errors(
            sample_false_allows, sample_false_denies
        )
        if sample_e_start <= 0:
            last_reason = "sample E_start is zero"
            continue

        sample_floor = max(10, len(sample_indices) // 200)
        if sample_e_start < sample_floor:
            last_reason = "sample E_start is too small"
            continue

        shadow_rng = random.Random(seed ^ 0x5A17E)
        shadow_indices = matrix.sample_indices(
            min(4096, matrix.total), shadow_rng
        )
        shadow_reference = bytes(reference[index] for index in shadow_indices)
        shadow_broken = matrix.decisions_at(shadow_indices, broken_policy)
        shadow_false_allows, shadow_false_denies = HiddenMatrix.error_counts(
            shadow_broken, shadow_reference
        )
        shadow_e_start = HiddenMatrix.weighted_errors(
            shadow_false_allows, shadow_false_denies
        )
        visible_rng = random.Random(seed ^ 0xC0FFEE)
        visible_examples = matrix.visible_examples(
            reference, config.visible_examples, visible_rng
        )
        requirements = render_requirements(
            world,
            correct_policy,
            task_id=config.task_id,
            tier=config.tier,
        )
        return GeneratedTask(
            config=config,
            seed=seed,
            world=world,
            correct_policy=correct_policy,
            broken_policy=broken_policy,
            requirements=requirements,
            visible_examples=visible_examples,
            matrix=matrix,
            reference_decisions=reference,
            e_start=e_start,
            sample_indices=sample_indices,
            reference_sample=reference_sample,
            broken_sample=broken_sample,
            shadow_indices=shadow_indices,
            shadow_e_start=shadow_e_start,
            mutations=tuple(records),
        )

    raise GenerationError(
        f"could not generate {config.task_id} seed={seed}: {last_reason}"
    )
