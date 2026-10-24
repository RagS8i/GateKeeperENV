"""Lossless natural-language rendering of a correct access policy."""

from __future__ import annotations

from collections.abc import Iterable

from .models import Policy, Statement, World


def _join_alternatives(items: list[str]) -> str:
    if not items:
        return "nothing"
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def _principal_phrase(selectors: Iterable[str]) -> str:
    parts: list[str] = []
    for selector in selectors:
        if selector == "*":
            parts.append("all users")
        elif selector.startswith("user:"):
            parts.append(f"user {selector.removeprefix('user:')}")
        else:
            parts.append(f"members of group {selector.removeprefix('group:')}")
    return _join_alternatives(parts)


def _action_phrase(selectors: Iterable[str]) -> str:
    parts: list[str] = []
    for selector in selectors:
        parts.append("any action" if selector == "*" else selector)
    return _join_alternatives(parts)


def _resource_phrase(selectors: Iterable[str]) -> str:
    parts: list[str] = []
    for selector in selectors:
        if selector == "*":
            parts.append("all resources")
        elif selector.startswith("resource:"):
            parts.append(f"resource {selector.removeprefix('resource:')}")
        else:
            parts.append(
                f"resources tagged {selector.removeprefix('tag:')}"
            )
    return _join_alternatives(parts)


def _condition_phrase(condition: dict[str, str] | None) -> str:
    if not condition:
        return ""
    clauses = [f"{key} is {value}" for key, value in sorted(condition.items())]
    return " only when " + " and ".join(clauses)


def _statement_sentence(statement: Statement) -> str:
    principals = _principal_phrase(statement.principals)
    actions = _action_phrase(statement.actions)
    resources = _resource_phrase(statement.resources)
    condition = _condition_phrase(statement.condition)

    if statement.effect == "allow":
        sentence = f"Allow {principals} to {actions} {resources}{condition}."
    else:
        sentence = (
            f"Deny {principals} permission to {actions} {resources}{condition}."
        )
    if statement.exceptions:
        sentence += (
            " This rule does not apply to "
            + _principal_phrase(statement.exceptions)
            + "."
        )
    return sentence


def render_requirements(
    world: World,
    policy: Policy,
    *,
    task_id: str,
    tier: str,
) -> str:
    lines = [
        f"Access requirements for Gatekeeper task {task_id} ({tier}).",
        "",
        "The company uses the actions read, write, delete and share.",
        "A request is allowed only when an allow rule applies and no deny rule",
        "applies. An explicit deny always overrides an allow. Every condition",
        "on a rule must match. Requests that no rule allows are denied by",
        "default. Within a rule, the listed principals, actions and resources",
        "are alternatives; they do not all need to match.",
        "",
        "Groups (members can be users or nested groups):",
    ]
    for group_id in sorted(world.groups):
        group = world.groups[group_id]
        members = _principal_phrase(group.members) if group.members else "no members"
        lines.append(f"- group {group_id}: {members}")

    lines.extend(["", "Rules:"])
    for number, statement in enumerate(
        sorted(policy.statements, key=lambda item: item.id), start=1
    ):
        lines.append(f"{number}. {_statement_sentence(statement)}")

    lines.extend(
        [
            "",
            "The hidden check covers every user, action, resource and supported",
            "context variant. Some checked users, groups, resources and condition",
            "keys are intentionally not mentioned above; inspect the directory",
            "rather than assuming the visible names are exhaustive.",
        ]
    )
    return "\n".join(lines)
