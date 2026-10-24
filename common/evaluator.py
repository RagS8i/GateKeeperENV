"""Deterministic access-policy evaluator.

Semantics (also exposed to the agent in the system prompt):

* default deny;
* every condition on a statement must match the request context;
* an exception principal is not subject to that statement;
* an explicit deny beats any allow;
* statements are considered in ascending id order, so the deciding statement
  id is deterministic and independent of statement insertion order.

The evaluator is deliberately free of any dependency on the generator, hidden
answer key or reward code.  It is shared by the environment (for the agent's
``simulate`` tool) and the verifier (for the hidden decision matrix).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .models import Effect, Policy, Statement, World


@dataclass(frozen=True, slots=True)
class Decision:
    effect: Effect
    statement_id: str | None = None

    @property
    def allowed(self) -> bool:
        return self.effect == "allow"


@dataclass(frozen=True, slots=True)
class _CompiledStatement:
    id: str
    effect: Effect
    principal_users: frozenset[str]
    principal_groups: frozenset[str]
    principal_wildcard: bool
    actions: frozenset[str]
    action_wildcard: bool
    resource_ids: frozenset[str]
    resource_tags: frozenset[str]
    resource_wildcard: bool
    condition: tuple[tuple[str, str], ...]
    exception_users: frozenset[str]
    exception_groups: frozenset[str]
    exception_wildcard: bool

    def applies(
        self,
        *,
        user_id: str,
        user_groups: frozenset[str],
        action: str,
        resource_id: str,
        resource_tags: frozenset[str],
        context: Mapping[str, str],
    ) -> bool:
        if not (
            self.principal_wildcard
            or user_id in self.principal_users
            or not self.principal_groups.isdisjoint(user_groups)
        ):
            return False

        if self.exception_wildcard or (
            user_id in self.exception_users
            or not self.exception_groups.isdisjoint(user_groups)
        ):
            return False

        if not (self.action_wildcard or action in self.actions):
            return False

        if not (
            self.resource_wildcard
            or resource_id in self.resource_ids
            or not self.resource_tags.isdisjoint(resource_tags)
        ):
            return False

        for key, expected in self.condition:
            if context.get(key) != expected:
                return False

        return True


@dataclass(frozen=True, slots=True)
class CompiledPolicy:
    """A policy indexed for repeated evaluation."""

    denies: tuple[_CompiledStatement, ...]
    allows: tuple[_CompiledStatement, ...]
    statement_count: int


def _compile_principal_selectors(
    selectors: Iterable[str],
) -> tuple[frozenset[str], frozenset[str], bool]:
    users: set[str] = set()
    groups: set[str] = set()
    wildcard = False
    for selector in selectors:
        if selector == "*":
            wildcard = True
        elif selector.startswith("user:"):
            users.add(selector.removeprefix("user:"))
        else:
            groups.add(selector.removeprefix("group:"))
    return frozenset(users), frozenset(groups), wildcard


def _compile_resource_selectors(
    selectors: Iterable[str],
) -> tuple[frozenset[str], frozenset[str], bool]:
    ids: set[str] = set()
    tags: set[str] = set()
    wildcard = False
    for selector in selectors:
        if selector == "*":
            wildcard = True
        elif selector.startswith("resource:"):
            ids.add(selector.removeprefix("resource:"))
        else:
            tags.add(selector.removeprefix("tag:"))
    return frozenset(ids), frozenset(tags), wildcard


def _compile_statement(statement: Statement) -> _CompiledStatement:
    principal_users, principal_groups, principal_wildcard = (
        _compile_principal_selectors(statement.principals)
    )
    resource_ids, resource_tags, resource_wildcard = _compile_resource_selectors(
        statement.resources
    )
    exception_users, exception_groups, exception_wildcard = (
        _compile_principal_selectors(statement.exceptions)
    )
    action_wildcard = "*" in statement.actions
    actions = frozenset(
        selector for selector in statement.actions if selector != "*"
    )
    condition = tuple(sorted((statement.condition or {}).items()))

    return _CompiledStatement(
        id=statement.id,
        effect=statement.effect,
        principal_users=principal_users,
        principal_groups=principal_groups,
        principal_wildcard=principal_wildcard,
        actions=actions,
        action_wildcard=action_wildcard,
        resource_ids=resource_ids,
        resource_tags=resource_tags,
        resource_wildcard=resource_wildcard,
        condition=condition,
        exception_users=exception_users,
        exception_groups=exception_groups,
        exception_wildcard=exception_wildcard,
    )


class Evaluator:
    """Fast, deterministic evaluator bound to one immutable world."""

    def __init__(self, world: World) -> None:
        self.world = world
        self._user_ids = frozenset(world.users)
        self._resource_ids = frozenset(world.resources)
        self._actions = frozenset(world.actions)
        self._resource_tags = {
            resource_id: frozenset(resource.tags)
            for resource_id, resource in world.resources.items()
        }
        self._user_groups = self._compute_user_groups(world)

    def _compute_user_groups(self, world: World) -> dict[str, frozenset[str]]:
        direct_user_groups: dict[str, set[str]] = defaultdict(set)
        group_parents: dict[str, set[str]] = defaultdict(set)

        for group_id, group in world.groups.items():
            for member in group.members:
                if member.startswith("user:"):
                    direct_user_groups[member.removeprefix("user:")].add(group_id)
                else:
                    group_parents[member.removeprefix("group:")].add(group_id)

        effective: dict[str, frozenset[str]] = {}
        for user_id in world.users:
            reached: set[str] = set()
            stack = list(direct_user_groups.get(user_id, ()))
            while stack:
                group_id = stack.pop()
                if group_id in reached:
                    continue
                reached.add(group_id)
                stack.extend(group_parents.get(group_id, ()))
            effective[user_id] = frozenset(reached)
        return effective

    def effective_groups(self, user_id: str) -> frozenset[str]:
        try:
            return self._user_groups[user_id]
        except KeyError:
            raise KeyError(f"unknown user: {user_id}") from None

    def compile(self, policy: Policy) -> CompiledPolicy:
        compiled = [_compile_statement(statement) for statement in policy.statements]
        denies = tuple(
            sorted(
                (statement for statement in compiled if statement.effect == "deny"),
                key=lambda statement: statement.id,
            )
        )
        allows = tuple(
            sorted(
                (statement for statement in compiled if statement.effect == "allow"),
                key=lambda statement: statement.id,
            )
        )
        return CompiledPolicy(
            denies=denies,
            allows=allows,
            statement_count=len(compiled),
        )

    def decide(
        self,
        policy: Policy | CompiledPolicy,
        user_id: str,
        action: str,
        resource_id: str,
        context: Mapping[str, str] | None = None,
    ) -> Decision:
        """Return the policy's decision for one request.

        Pass a :class:`CompiledPolicy` when scoring many requests; passing a
        :class:`Policy` compiles it on every call.
        """

        compiled = policy if isinstance(policy, CompiledPolicy) else self.compile(policy)
        user_groups = self.effective_groups(user_id)

        if action not in self._actions:
            raise ValueError(f"unknown action: {action}")
        try:
            resource_tags = self._resource_tags[resource_id]
        except KeyError:
            raise KeyError(f"unknown resource: {resource_id}") from None

        request_context = {} if context is None else context

        for statement in compiled.denies:
            if statement.applies(
                user_id=user_id,
                user_groups=user_groups,
                action=action,
                resource_id=resource_id,
                resource_tags=resource_tags,
                context=request_context,
            ):
                return Decision(effect="deny", statement_id=statement.id)

        for statement in compiled.allows:
            if statement.applies(
                user_id=user_id,
                user_groups=user_groups,
                action=action,
                resource_id=resource_id,
                resource_tags=resource_tags,
                context=request_context,
            ):
                return Decision(effect="allow", statement_id=statement.id)

        return Decision(effect="deny", statement_id=None)
