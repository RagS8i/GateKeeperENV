"""Core data models for the Gatekeeper access-policy environment.

Terminology note: an access-policy statement is part of the customer's
permission ruleset.  It is deliberately never called a "policy" on its own, to
avoid confusion with the reinforcement-learning notion of an agent policy.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ALL_ACTIONS: tuple[str, ...] = ("read", "write", "delete", "share")

Effect = Literal["allow", "deny"]

_PRINCIPAL_PREFIXES = ("user:", "group:")
_RESOURCE_PREFIXES = ("resource:", "tag:")

_MAX_ID = 128
_MAX_SELECTOR = 160
_MAX_SELECTORS = 64
_MAX_ACTIONS = 8
_MAX_CONDITION_KEYS = 16
_MAX_TAGS = 32
_MAX_STATEMENTS = 200
_MAX_MEMBERS = 256


def _validate_principal_selector(selector: str) -> str:
    if selector == "*":
        return selector
    prefix, separator, value = selector.partition(":")
    if not separator or f"{prefix}:" not in _PRINCIPAL_PREFIXES or not value:
        raise ValueError(
            "principal selector must be '*', 'user:<user-id>' or 'group:<group-id>'"
        )
    return selector


def _validate_resource_selector(selector: str) -> str:
    if selector == "*":
        return selector
    prefix, separator, value = selector.partition(":")
    if not separator or f"{prefix}:" not in _RESOURCE_PREFIXES or not value:
        raise ValueError(
            "resource selector must be '*', 'resource:<resource-id>' or 'tag:<tag>'"
        )
    return selector


def _validate_action_selector(selector: str) -> str:
    if selector == "*" or selector in ALL_ACTIONS:
        return selector
    raise ValueError(
        f"action selector must be '*' or one of {', '.join(ALL_ACTIONS)}"
    )


class Statement(BaseModel):
    """One allow or deny statement in an access policy."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=1, max_length=_MAX_ID)
    effect: Effect
    principals: list[str] = Field(min_length=1, max_length=_MAX_SELECTORS)
    actions: list[str] = Field(min_length=1, max_length=_MAX_ACTIONS)
    resources: list[str] = Field(min_length=1, max_length=_MAX_SELECTORS)
    condition: dict[str, str] | None = None
    exceptions: list[str] = Field(default_factory=list, max_length=_MAX_SELECTORS)

    @field_validator("principals", "exceptions")
    @classmethod
    def _validate_principals(cls, selectors: list[str]) -> list[str]:
        for selector in selectors:
            if len(selector) > _MAX_SELECTOR:
                raise ValueError("principal selector is too long")
        return [_validate_principal_selector(selector) for selector in selectors]

    @field_validator("actions")
    @classmethod
    def _validate_actions(cls, selectors: list[str]) -> list[str]:
        return [_validate_action_selector(selector) for selector in selectors]

    @field_validator("resources")
    @classmethod
    def _validate_resources(cls, selectors: list[str]) -> list[str]:
        for selector in selectors:
            if len(selector) > _MAX_SELECTOR:
                raise ValueError("resource selector is too long")
        return [_validate_resource_selector(selector) for selector in selectors]

    @field_validator("condition")
    @classmethod
    def _validate_condition(
        cls, condition: dict[str, str] | None
    ) -> dict[str, str] | None:
        if condition is None:
            return None
        if len(condition) > _MAX_CONDITION_KEYS:
            raise ValueError("condition has too many keys")
        for key, value in condition.items():
            if not key:
                raise ValueError("condition keys must not be empty")
            if len(key) > _MAX_SELECTOR or len(value) > _MAX_SELECTOR:
                raise ValueError("condition key or value is too long")
            if not isinstance(value, str):
                raise ValueError("condition values must be strings")
        return condition


class Policy(BaseModel):
    """An ordered collection of access-policy statements.

    Order is not semantically meaningful: the evaluator sorts statements by id
    so that the deciding statement is deterministic across equivalent policies.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    statements: list[Statement] = Field(
        default_factory=list, max_length=_MAX_STATEMENTS
    )

    @model_validator(mode="after")
    def _unique_statement_ids(self) -> "Policy":
        statement_ids = [statement.id for statement in self.statements]
        if len(statement_ids) != len(set(statement_ids)):
            raise ValueError("statement ids must be unique within an access policy")
        return self


class User(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=_MAX_ID)


class Group(BaseModel):
    """A group whose members are users or nested groups.

    Members use the same selector syntax as statements: ``user:<id>`` or
    ``group:<id>``.  Wildcards are not allowed in membership lists.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=_MAX_ID)
    members: list[str] = Field(default_factory=list, max_length=_MAX_MEMBERS)

    @field_validator("members")
    @classmethod
    def _validate_members(cls, selectors: list[str]) -> list[str]:
        for selector in selectors:
            if len(selector) > _MAX_SELECTOR:
                raise ValueError("group member selector is too long")
            _validate_principal_selector(selector)
            if selector == "*":
                raise ValueError("group membership cannot use a wildcard principal")
        return selectors


class Resource(BaseModel):
    """A protected resource.  Tags may encode department, sensitivity, etc."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=_MAX_ID)
    tags: list[str] = Field(default_factory=list, max_length=_MAX_TAGS)

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, tags: list[str]) -> list[str]:
        for tag in tags:
            if not tag:
                raise ValueError("resource tags must not be empty")
            if len(tag) > _MAX_SELECTOR:
                raise ValueError("resource tag is too long")
        return tags


class World(BaseModel):
    """The people, groups and resources an access policy applies to."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    users: dict[str, User]
    groups: dict[str, Group] = Field(default_factory=dict)
    resources: dict[str, Resource]
    actions: tuple[str, ...] = ALL_ACTIONS

    @model_validator(mode="after")
    def _validate_world(self) -> "World":
        if not self.actions:
            raise ValueError("world must define at least one action")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("world action names must be unique")
        for action in self.actions:
            if action not in ALL_ACTIONS:
                raise ValueError(f"unknown world action: {action!r}")

        for key, user in self.users.items():
            if key != user.id:
                raise ValueError(f"user key {key!r} does not match id {user.id!r}")
        for key, group in self.groups.items():
            if key != group.id:
                raise ValueError(f"group key {key!r} does not match id {group.id!r}")
        for key, resource in self.resources.items():
            if key != resource.id:
                raise ValueError(
                    f"resource key {key!r} does not match id {resource.id!r}"
                )

        for group in self.groups.values():
            for member in group.members:
                if member.startswith("user:"):
                    user_id = member.removeprefix("user:")
                    if user_id not in self.users:
                        raise ValueError(
                            f"group {group.id!r} references unknown user {user_id!r}"
                        )
                else:
                    nested_group_id = member.removeprefix("group:")
                    if nested_group_id not in self.groups:
                        raise ValueError(
                            f"group {group.id!r} references unknown group "
                            f"{nested_group_id!r}"
                        )
        return self


def canonical_statement(statement: Statement) -> dict[str, object]:
    """Return a stable, order-insensitive representation of one statement."""

    condition = dict(sorted((statement.condition or {}).items()))
    return {
        "id": statement.id,
        "effect": statement.effect,
        "principals": sorted(statement.principals),
        "actions": sorted(statement.actions),
        "resources": sorted(statement.resources),
        "condition": condition or None,
        "exceptions": sorted(statement.exceptions),
    }


def canonical_policy(policy: Policy) -> dict[str, object]:
    return {
        "statements": [
            canonical_statement(statement)
            for statement in sorted(policy.statements, key=lambda item: item.id)
        ]
    }


def canonical_world(world: World) -> dict[str, object]:
    return {
        "users": sorted(world.users),
        "groups": {
            group_id: sorted(world.groups[group_id].members)
            for group_id in sorted(world.groups)
        },
        "resources": {
            resource_id: sorted(world.resources[resource_id].tags)
            for resource_id in sorted(world.resources)
        },
        "actions": sorted(world.actions),
    }


def canonical_state(world: World, policy: Policy) -> dict[str, object]:
    return {"world": canonical_world(world), "policy": canonical_policy(policy)}


def state_hash(world: World, policy: Policy) -> str:
    """Short stable hash used to prove that two runs saw the same state."""

    payload = json.dumps(
        canonical_state(world, policy), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
