"""Independent, deliberately simple reference implementation.

This exists only for the differential test.  It shares the data models with
the production evaluator but none of its indexing or matching logic.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

from gatekeeper.models import Policy, World


class NaiveDecision(NamedTuple):
    effect: str
    statement_id: str | None


class NaiveEvaluator:
    def __init__(self, world: World) -> None:
        self.world = world
        self._token_cache: dict[str, frozenset[str]] = {}

    def _principal_tokens(self, user_id: str) -> frozenset[str]:
        cached = self._token_cache.get(user_id)
        if cached is not None:
            return cached

        tokens = {f"user:{user_id}"}
        changed = True
        while changed:
            changed = False
            for group_id, group in self.world.groups.items():
                group_token = f"group:{group_id}"
                if group_token in tokens:
                    continue
                if any(member in tokens for member in group.members):
                    tokens.add(group_token)
                    changed = True

        result = frozenset(tokens)
        self._token_cache[user_id] = result
        return result

    @staticmethod
    def _principal_matches(selector: str, user_id: str, tokens: frozenset[str]) -> bool:
        if selector == "*":
            return True
        if selector.startswith("user:"):
            return selector.removeprefix("user:") == user_id
        if selector.startswith("group:"):
            return selector in tokens
        return False

    @staticmethod
    def _resource_matches(
        selector: str, resource_id: str, resource_tags: list[str]
    ) -> bool:
        if selector == "*":
            return True
        if selector.startswith("resource:"):
            return selector.removeprefix("resource:") == resource_id
        if selector.startswith("tag:"):
            return selector.removeprefix("tag:") in resource_tags
        return False

    def _applies(
        self,
        statement,
        user_id: str,
        tokens: frozenset[str],
        action: str,
        resource_id: str,
        resource_tags: list[str],
        context: Mapping[str, str],
    ) -> bool:
        if not any(
            self._principal_matches(selector, user_id, tokens)
            for selector in statement.principals
        ):
            return False

        if any(
            self._principal_matches(selector, user_id, tokens)
            for selector in statement.exceptions
        ):
            return False

        if "*" not in statement.actions and action not in statement.actions:
            return False

        if not any(
            self._resource_matches(selector, resource_id, resource_tags)
            for selector in statement.resources
        ):
            return False

        for key, expected in (statement.condition or {}).items():
            if context.get(key) != expected:
                return False

        return True

    def decide(
        self,
        policy: Policy,
        user_id: str,
        action: str,
        resource_id: str,
        context: Mapping[str, str] | None = None,
    ) -> NaiveDecision:
        resource_tags = self.world.resources[resource_id].tags
        tokens = self._principal_tokens(user_id)
        request_context = {} if context is None else context

        first_allow: str | None = None
        for statement in sorted(policy.statements, key=lambda item: item.id):
            if not self._applies(
                statement,
                user_id,
                tokens,
                action,
                resource_id,
                resource_tags,
                request_context,
            ):
                continue
            if statement.effect == "deny":
                return NaiveDecision("deny", statement.id)
            if first_allow is None:
                first_allow = statement.id

        if first_allow is not None:
            return NaiveDecision("allow", first_allow)
        return NaiveDecision("deny", None)
