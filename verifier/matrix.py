"""Hidden exhaustive decision matrix used by the verifier.

The matrix is the Cartesian product of users, actions, resources and a small set
of context variants.  It is represented sparsely (one byte per decision) and
requests are decoded on demand, so even the hard tasks stay cheap to score.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from common.evaluator import Evaluator
from common.models import Policy, World
from common.request import AccessRequest

MatrixContext = Mapping[str, str] | Sequence[tuple[str, str]]


@dataclass(slots=True)
class HiddenMatrix:
    world: World
    contexts: tuple[tuple[tuple[str, str], ...], ...]
    users: tuple[str, ...] = field(init=False)
    actions: tuple[str, ...] = field(init=False)
    resources: tuple[str, ...] = field(init=False)
    evaluator: Evaluator = field(init=False)

    def __post_init__(self) -> None:
        if not self.contexts:
            raise ValueError("hidden matrix needs at least one context variant")
        self.users = tuple(sorted(self.world.users))
        self.actions = tuple(self.world.actions)
        self.resources = tuple(sorted(self.world.resources))
        self.evaluator = Evaluator(self.world)

    @classmethod
    def build(
        cls, world: World, contexts: Iterable[MatrixContext]
    ) -> "HiddenMatrix":
        normalized = tuple(
            tuple(sorted(dict(context).items())) for context in contexts
        )
        return cls(world=world, contexts=normalized)

    @property
    def context_count(self) -> int:
        return len(self.contexts)

    @property
    def total(self) -> int:
        return (
            len(self.users)
            * len(self.actions)
            * len(self.resources)
            * len(self.contexts)
        )

    def index_of(
        self,
        user_id: str,
        action: str,
        resource_id: str,
        context: MatrixContext | None = None,
    ) -> int:
        context_items = tuple(sorted(dict(context or {}).items()))
        try:
            user_index = self.users.index(user_id)
            action_index = self.actions.index(action)
            resource_index = self.resources.index(resource_id)
            context_index = self.contexts.index(context_items)
        except ValueError as exc:
            raise KeyError(f"request is outside the hidden matrix: {exc}") from None
        return (
            user_index * len(self.actions) * len(self.resources) * len(self.contexts)
            + action_index * len(self.resources) * len(self.contexts)
            + resource_index * len(self.contexts)
            + context_index
        )

    def request_at(self, index: int) -> AccessRequest:
        if not 0 <= index < self.total:
            raise IndexError(index)
        context_count = len(self.contexts)
        resource_count = len(self.resources)
        action_count = len(self.actions)

        context_index = index % context_count
        index //= context_count
        resource_index = index % resource_count
        index //= resource_count
        action_index = index % action_count
        user_index = index // action_count
        return AccessRequest(
            user_id=self.users[user_index],
            action=self.actions[action_index],
            resource_id=self.resources[resource_index],
            context=self.contexts[context_index],
        )

    def decide_at(self, index: int, policy: Policy) -> bool:
        request = self.request_at(index)
        decision = self.evaluator.decide(
            policy,
            request.user_id,
            request.action,
            request.resource_id,
            request.context_dict(),
        )
        return decision.allowed

    def decisions(self, policy: Policy) -> bytearray:
        """Evaluate one policy over the whole matrix (1 = allow, 0 = deny)."""

        compiled = self.evaluator.compile(policy)
        out = bytearray(self.total)
        position = 0
        for user_id in self.users:
            for action in self.actions:
                for resource_id in self.resources:
                    for context in self.contexts:
                        decision = self.evaluator.decide(
                            compiled,
                            user_id,
                            action,
                            resource_id,
                            dict(context),
                        )
                        out[position] = 1 if decision.allowed else 0
                        position += 1
        return out

    def decisions_at(
        self, indices: Iterable[int], policy: Policy
    ) -> bytearray:
        compiled = self.evaluator.compile(policy)
        out = bytearray()
        for index in indices:
            request = self.request_at(index)
            decision = self.evaluator.decide(
                compiled,
                request.user_id,
                request.action,
                request.resource_id,
                request.context_dict(),
            )
            out.append(1 if decision.allowed else 0)
        return out

    @staticmethod
    def error_counts(candidate: bytes, reference: bytes) -> tuple[int, int]:
        if len(candidate) != len(reference):
            raise ValueError("decision arrays must have the same length")
        false_allows = 0
        false_denies = 0
        for got, want in zip(candidate, reference):
            if got == want:
                continue
            if got:
                false_allows += 1
            else:
                false_denies += 1
        return false_allows, false_denies

    @staticmethod
    def weighted_errors(false_allows: int, false_denies: int) -> int:
        return 3 * false_allows + false_denies

    @staticmethod
    def score_from_errors(
        false_allows: int, false_denies: int, e_start: int
    ) -> float:
        if e_start <= 0:
            return 1.0 if false_allows == 0 and false_denies == 0 else 0.0
        error = 3 * false_allows + false_denies
        return max(0.0, 1.0 - (error / e_start))

    def counts(self, decisions: bytes) -> tuple[int, int]:
        allows = sum(decisions)
        return allows, len(decisions) - allows

    def sample_indices(
        self, size: int, rng: random.Random
    ) -> tuple[int, ...]:
        if size >= self.total:
            return tuple(range(self.total))
        return tuple(sorted(rng.sample(range(self.total), size)))

    def visible_examples(
        self,
        reference: bytes,
        count: int,
        rng: random.Random,
    ) -> list[dict[str, object]]:
        """Choose a deterministic, balanced set of visible examples."""

        if count >= self.total:
            indices = list(range(self.total))
        else:
            allow_indices = [i for i, value in enumerate(reference) if value]
            deny_indices = [i for i, value in enumerate(reference) if not value]
            chosen: set[int] = set()
            if allow_indices:
                chosen.add(rng.choice(allow_indices))
            if deny_indices:
                chosen.add(rng.choice(deny_indices))
            while len(chosen) < count:
                chosen.add(rng.randrange(self.total))
            indices = sorted(chosen)

        examples: list[dict[str, object]] = []
        for index in indices:
            request = self.request_at(index)
            example = request.as_visible()
            example["expected"] = "allow" if reference[index] else "deny"
            examples.append(example)
        return examples

    def shadow_score(
        self,
        indices: Sequence[int],
        candidate: bytes,
        reference: bytes,
        e_start_sample: int,
    ) -> dict[str, object]:
        if len(indices) != len(candidate):
            raise ValueError("candidate shadow decisions do not match indices")
        sample_reference = bytes(reference[index] for index in indices)
        false_allows, false_denies = self.error_counts(candidate, sample_reference)
        return {
            "false_allows": false_allows,
            "false_denies": false_denies,
            "weighted_errors": self.weighted_errors(false_allows, false_denies),
            "e_start": e_start_sample,
            "score": self.score_from_errors(
                false_allows, false_denies, e_start_sample
            ),
        }
