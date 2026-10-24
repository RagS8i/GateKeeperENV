"""Agent-facing environment tools.

This module receives only the sanitized task bundle (world, broken policy,
requirements and visible examples).  It has no access to the hidden correct
policy, mutation list or decision matrix.
"""

from __future__ import annotations

import json
from typing import Any

from .evaluator import Evaluator
from .models import (
    Policy,
    Statement,
    World,
    canonical_statement,
    state_hash,
)

MAX_OUTPUT_CHARS = 4000
MAX_POLICY_JSON_CHARS = 20_000
PAGE_SIZE = 5


class EnvironmentError(ValueError):
    pass


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n...[truncated]"


def _normalize_context(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise EnvironmentError("context must be an object of string values")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise EnvironmentError("context keys must be non-empty strings")
        if isinstance(item, bool):
            normalized[key] = "true" if item else "false"
        elif isinstance(item, (str, int, float)):
            normalized[key] = str(item)
        else:
            raise EnvironmentError(
                f"context value for {key!r} must be a string"
            )
    return normalized


class Environment:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.task_id = str(bundle["task_id"])
        self.tier = str(bundle["tier"])
        self.seed = int(bundle["seed"])
        self.world = World.model_validate(bundle["world"])
        self.initial_policy = Policy.model_validate(bundle["broken_policy"])
        self.current_policy = self.initial_policy
        self.requirements = str(bundle["requirements"])
        self.visible_examples = list(bundle["visible_examples"])
        self.evaluator = Evaluator(self.world)
        self.history: list[Policy] = []
        self.submitted = False
        self.initial_state_hash = state_hash(self.world, self.initial_policy)
        declared_hash = bundle.get("state_hash")
        if declared_hash is not None and declared_hash != self.initial_state_hash:
            raise EnvironmentError("task state hash does not match the bundle")

    @property
    def state_hash(self) -> str:
        return state_hash(self.world, self.current_policy)

    def observation(self) -> str:
        policy_json = json.dumps(
            self.current_policy.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        return (
            f"Task {self.task_id} ({self.tier}).\n\n"
            f"{self.requirements}\n\n"
            "Current access policy (this policy is broken):\n"
            f"{policy_json}\n\n"
            "Tools: get_requirements, list_policies, lookup, simulate, "
            "put_policy, delete_policy, undo, diff, check_examples, submit. "
            "Call submit only when you believe the policy is repaired."
        )

    def state(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state_hash": self.state_hash,
            "statement_count": len(self.current_policy.statements),
            "submitted": self.submitted,
            "policy": self.current_policy.model_dump(mode="json"),
        }

    def step(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(tool, str) or not tool:
            return {"ok": False, "content": "ERROR: tool name must be a string"}
        arguments = {} if args is None else args
        if not isinstance(arguments, dict):
            return {
                "ok": False,
                "content": "ERROR: tool arguments must be a JSON object",
            }

        if tool == "submit":
            self.submitted = True
            return {
                "ok": True,
                "content": "Submitted. The episode is complete.",
            }
        if self.submitted:
            return {
                "ok": False,
                "content": "ERROR: the episode has already been submitted",
            }

        handlers = {
            "get_requirements": self.get_requirements,
            "list_policies": self.list_policies,
            "lookup": self.lookup,
            "simulate": self.simulate,
            "put_policy": self.put_policy,
            "delete_policy": self.delete_policy,
            "undo": self.undo,
            "diff": self.diff,
            "check_examples": self.check_examples,
        }
        handler = handlers.get(tool)
        if handler is None:
            return {
                "ok": False,
                "content": f"ERROR: unknown tool {tool!r}. Available: "
                + ", ".join(sorted([*handlers, "submit"])),
            }
        try:
            content = handler(arguments)
        except Exception as exc:
            return {
                "ok": False,
                "content": f"ERROR: {type(exc).__name__}: {exc}",
            }
        return {"ok": True, "content": _clip(content)}

    def get_requirements(self, _args: dict[str, Any]) -> str:
        return self.requirements

    def list_policies(self, args: dict[str, Any]) -> str:
        page = int(args.get("page", 1))
        if page < 1:
            raise EnvironmentError("page must be at least 1")
        statements = sorted(self.current_policy.statements, key=lambda item: item.id)
        pages = max(1, (len(statements) + PAGE_SIZE - 1) // PAGE_SIZE)
        start = (page - 1) * PAGE_SIZE
        selected = statements[start : start + PAGE_SIZE]
        payload = {
            "page": page,
            "pages": pages,
            "statement_count": len(statements),
            "statements": [
                canonical_statement(statement) for statement in selected
            ],
        }
        if not selected:
            payload["note"] = "page is past the end of the policy"
        return json.dumps(payload, indent=2)

    def _parent_groups(self, group_id: str) -> list[str]:
        selector = f"group:{group_id}"
        return sorted(
            group.id
            for group in self.world.groups.values()
            if selector in group.members
        )

    def _available(self, kind: str, limit: int = 12) -> list[str]:
        if kind == "user":
            values = sorted(self.world.users)
        elif kind == "group":
            values = sorted(self.world.groups)
        elif kind == "resource":
            values = sorted(self.world.resources)
        elif kind == "tag":
            values = sorted(
                {
                    tag
                    for resource in self.world.resources.values()
                    for tag in resource.tags
                }
            )
        elif kind == "action":
            values = list(self.world.actions)
        else:
            values = []
        return values[:limit]

    def lookup(self, args: dict[str, Any]) -> str:
        kind = str(args.get("kind", "")).strip().lower()
        raw_id = str(args.get("id", "")).strip()
        if not raw_id:
            raise EnvironmentError("lookup requires an id")
        if kind not in {"user", "group", "resource", "tag", "action"}:
            raise EnvironmentError(
                "kind must be one of user, group, resource, tag, action"
            )
        identifier = raw_id
        if identifier.startswith(f"{kind}:"):
            identifier = identifier.split(":", 1)[1]

        if kind == "user":
            if identifier not in self.world.users:
                return json.dumps(
                    {
                        "kind": kind,
                        "id": identifier,
                        "exists": False,
                        "available": self._available(kind),
                    }
                )
            groups = sorted(self.evaluator.effective_groups(identifier))
            direct = sorted(
                group.id
                for group in self.world.groups.values()
                if f"user:{identifier}" in group.members
            )
            return json.dumps(
                {
                    "kind": kind,
                    "id": identifier,
                    "exists": True,
                    "direct_groups": direct,
                    "effective_groups": groups,
                },
                indent=2,
            )
        if kind == "group":
            if identifier not in self.world.groups:
                return json.dumps(
                    {
                        "kind": kind,
                        "id": identifier,
                        "exists": False,
                        "available": self._available(kind),
                    }
                )
            group = self.world.groups[identifier]
            return json.dumps(
                {
                    "kind": kind,
                    "id": identifier,
                    "exists": True,
                    "members": group.members,
                    "parent_groups": self._parent_groups(identifier),
                },
                indent=2,
            )
        if kind == "resource":
            if identifier not in self.world.resources:
                return json.dumps(
                    {
                        "kind": kind,
                        "id": identifier,
                        "exists": False,
                        "available": self._available(kind),
                    }
                )
            return json.dumps(
                {
                    "kind": kind,
                    "id": identifier,
                    "exists": True,
                    "tags": self.world.resources[identifier].tags,
                },
                indent=2,
            )
        if kind == "tag":
            resources = sorted(
                resource_id
                for resource_id, resource in self.world.resources.items()
                if identifier in resource.tags
            )
            return json.dumps(
                {
                    "kind": kind,
                    "id": identifier,
                    "exists": bool(resources),
                    "resource_count": len(resources),
                    "resources": resources[:50],
                },
                indent=2,
            )
        if identifier not in self.world.actions:
            return json.dumps(
                {
                    "kind": kind,
                    "id": identifier,
                    "exists": False,
                    "available": self._available(kind),
                }
            )
        return json.dumps(
            {"kind": kind, "id": identifier, "exists": True}, indent=2
        )

    def _validate_statement_references(self, statement: Statement) -> None:
        tags = {
            tag
            for resource in self.world.resources.values()
            for tag in resource.tags
        }
        for selector in [*statement.principals, *statement.exceptions]:
            if selector == "*":
                continue
            prefix, _, identifier = selector.partition(":")
            if prefix == "user" and identifier not in self.world.users:
                raise EnvironmentError(f"unknown user selector {selector!r}")
            if prefix == "group" and identifier not in self.world.groups:
                raise EnvironmentError(f"unknown group selector {selector!r}")
        for selector in statement.resources:
            if selector == "*":
                continue
            prefix, _, identifier = selector.partition(":")
            if prefix == "resource" and identifier not in self.world.resources:
                raise EnvironmentError(f"unknown resource selector {selector!r}")
            if prefix == "tag" and identifier not in tags:
                raise EnvironmentError(f"unknown tag selector {selector!r}")
        for action in statement.actions:
            if action != "*" and action not in self.world.actions:
                raise EnvironmentError(f"unknown action {action!r}")

    def put_policy(self, args: dict[str, Any]) -> str:
        payload = args.get("statement")
        if not isinstance(payload, dict):
            raise EnvironmentError("put_policy requires a statement object")
        if len(json.dumps(payload)) > MAX_POLICY_JSON_CHARS:
            raise EnvironmentError("statement payload is too large")
        statement = Statement.model_validate(payload)
        self._validate_statement_references(statement)

        ids = {item.id for item in self.current_policy.statements}
        if statement.id not in ids and len(ids) >= 200:
            raise EnvironmentError("policy already has the maximum 200 statements")
        self.history.append(self.current_policy)
        replaced = statement.id in ids
        self.current_policy = Policy(
            statements=[
                statement if item.id == statement.id else item
                for item in self.current_policy.statements
                if replaced or item.id != statement.id
            ]
            + ([] if replaced else [statement])
        )
        action = "replaced" if replaced else "added"
        return json.dumps(
            {
                "updated": True,
                "action": action,
                "statement_id": statement.id,
                "statement_count": len(self.current_policy.statements),
            }
        )

    def delete_policy(self, args: dict[str, Any]) -> str:
        statement_id = str(args.get("id", "")).strip()
        if not statement_id:
            raise EnvironmentError("delete_policy requires an id")
        existing = {item.id for item in self.current_policy.statements}
        if statement_id not in existing:
            raise EnvironmentError(f"unknown statement id {statement_id!r}")
        self.history.append(self.current_policy)
        self.current_policy = Policy(
            statements=[
                item
                for item in self.current_policy.statements
                if item.id != statement_id
            ]
        )
        return json.dumps(
            {
                "deleted": True,
                "statement_id": statement_id,
                "statement_count": len(self.current_policy.statements),
            }
        )

    def undo(self, _args: dict[str, Any]) -> str:
        if not self.history:
            raise EnvironmentError("there is no edit to undo")
        self.current_policy = self.history.pop()
        return json.dumps(
            {
                "undone": True,
                "statement_count": len(self.current_policy.statements),
                "state_hash": self.state_hash,
            }
        )

    def diff(self, _args: dict[str, Any]) -> str:
        initial = {
            statement.id: canonical_statement(statement)
            for statement in self.initial_policy.statements
        }
        current = {
            statement.id: canonical_statement(statement)
            for statement in self.current_policy.statements
        }
        added = [
            current[statement_id]
            for statement_id in sorted(set(current) - set(initial))
        ]
        removed = sorted(set(initial) - set(current))
        changed = []
        for statement_id in sorted(set(initial) & set(current)):
            if initial[statement_id] != current[statement_id]:
                changed.append(
                    {
                        "id": statement_id,
                        "before": initial[statement_id],
                        "after": current[statement_id],
                    }
                )
        return json.dumps(
            {
                "added": added,
                "removed": removed,
                "changed": changed,
                "added_count": len(added),
                "removed_count": len(removed),
                "changed_count": len(changed),
            },
            indent=2,
        )

    def simulate(self, args: dict[str, Any]) -> str:
        principal = str(args.get("principal", "")).strip()
        resource = str(args.get("resource", "")).strip()
        action = str(args.get("action", "")).strip()
        if not principal or not resource or not action:
            raise EnvironmentError(
                "simulate requires principal, action and resource"
            )
        if principal.startswith("group:"):
            raise EnvironmentError(
                "simulate expects a user principal, not a group"
            )
        if principal.startswith("user:"):
            principal = principal.split(":", 1)[1]
        if resource.startswith("resource:"):
            resource = resource.split(":", 1)[1]
        if principal not in self.world.users:
            raise EnvironmentError(f"unknown user {principal!r}")
        if resource not in self.world.resources:
            raise EnvironmentError(f"unknown resource {resource!r}")
        if action not in self.world.actions:
            raise EnvironmentError(f"unknown action {action!r}")
        context = _normalize_context(args.get("context"))
        decision = self.evaluator.decide(
            self.current_policy, principal, action, resource, context
        )
        return json.dumps(
            {
                "request": {
                    "principal": f"user:{principal}",
                    "action": action,
                    "resource": f"resource:{resource}",
                    "context": context,
                },
                "effect": decision.effect,
                "allowed": decision.allowed,
                "statement_id": decision.statement_id,
                "default_deny": decision.statement_id is None,
            },
            indent=2,
        )

    def check_examples(self, _args: dict[str, Any]) -> str:
        examples = []
        correct = 0
        for example in self.visible_examples:
            decision = self.evaluator.decide(
                self.current_policy,
                example["user"],
                example["action"],
                example["resource"],
                example.get("context") or {},
            )
            expected = example["expected"]
            is_correct = decision.effect == expected
            correct += int(is_correct)
            examples.append(
                {
                    "user": example["user"],
                    "action": example["action"],
                    "resource": example["resource"],
                    "context": example.get("context") or {},
                    "expected": expected,
                    "actual": decision.effect,
                    "statement_id": decision.statement_id,
                    "correct": is_correct,
                }
            )
        return json.dumps(
            {
                "correct": correct,
                "total": len(examples),
                "examples": examples,
            },
            indent=2,
        )
