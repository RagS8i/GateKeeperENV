"""Differential test: fast evaluator vs. an independent naive implementation.

The random generator is fully seeded.  Every run checks 2,000 distinct random
access policies across 40 random worlds and 400,000 individual requests.
"""

from __future__ import annotations

import random

import pytest

from common.evaluator import Evaluator
from common.models import (
    ALL_ACTIONS,
    Group,
    Policy,
    Resource,
    Statement,
    User,
    World,
)
from tests.naive_evaluator import NaiveEvaluator

N_WORLDS = 40
POLICIES_PER_WORLD = 50
REQUESTS_PER_POLICY = 200
TOTAL_POLICIES = N_WORLDS * POLICIES_PER_WORLD
TOTAL_REQUESTS = TOTAL_POLICIES * REQUESTS_PER_POLICY

TAGS = ("finance", "hr", "eng", "public", "secret", "ops")
NETWORKS = ("corp", "vpn", "public")
MFA_VALUES = ("true", "false")
REGIONS = ("eu", "us", "apac")

CONTEXTS = (
    {},
    {"network": "corp"},
    {"network": "vpn", "mfa": "true"},
    {"network": "public", "mfa": "false"},
    {"network": "corp", "mfa": "true", "region": "eu"},
    {"network": "vpn", "mfa": "false", "region": "us"},
)


def random_world(rng: random.Random) -> World:
    n_users = rng.randint(1, 8)
    user_ids = [f"u{index}" for index in range(n_users)]
    users = {user_id: User(id=user_id) for user_id in user_ids}

    n_groups = rng.randint(0, 5)
    group_ids = [f"g{index}" for index in range(n_groups)]
    groups: dict[str, Group] = {}
    principal_pool = [f"user:{user_id}" for user_id in user_ids] + [
        f"group:{group_id}" for group_id in group_ids
    ]
    for group_id in group_ids:
        members = [
            rng.choice(principal_pool) for _ in range(rng.randint(0, 4))
        ]
        groups[group_id] = Group(id=group_id, members=members)

    n_resources = rng.randint(1, 8)
    resources = {
        f"r{index}": Resource(
            id=f"r{index}", tags=list(rng.sample(TAGS, rng.randint(0, 3)))
        )
        for index in range(n_resources)
    }

    return World(users=users, groups=groups, resources=resources)


def random_principal_selector(rng: random.Random, world: World) -> str:
    choices: list[str] = [f"user:{user_id}" for user_id in world.users]
    choices.extend(f"group:{group_id}" for group_id in world.groups)
    choices.append("*")
    return rng.choice(choices)


def random_action_selector(rng: random.Random) -> str:
    return "*" if rng.random() < 0.15 else rng.choice(ALL_ACTIONS)


def random_resource_selector(rng: random.Random, world: World) -> str:
    roll = rng.random()
    if roll < 0.15:
        return "*"
    if roll < 0.60:
        return f"resource:{rng.choice(list(world.resources))}"
    return f"tag:{rng.choice(TAGS)}"


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def random_condition(rng: random.Random) -> dict[str, str] | None:
    roll = rng.random()
    if roll < 0.45:
        return None
    if roll < 0.55:
        return {}
    keys = rng.sample(("network", "mfa", "region"), rng.randint(1, 2))
    values = {
        "network": NETWORKS,
        "mfa": MFA_VALUES,
        "region": REGIONS,
    }
    return {key: rng.choice(values[key]) for key in keys}


def random_policy(rng: random.Random, world: World) -> Policy:
    statements: list[Statement] = []
    for index in range(rng.randint(0, 12)):
        principals = unique(
            [
                random_principal_selector(rng, world)
                for _ in range(rng.randint(1, 2))
            ]
        )
        actions = unique(
            [random_action_selector(rng) for _ in range(rng.randint(1, 2))]
        )
        resources = unique(
            [
                random_resource_selector(rng, world)
                for _ in range(rng.randint(1, 2))
            ]
        )
        exceptions = unique(
            [
                random_principal_selector(rng, world)
                for _ in range(rng.randint(0, 2))
            ]
        )
        statements.append(
            Statement(
                id=f"s{index:02d}",
                effect=rng.choice(("allow", "deny")),
                principals=principals,
                actions=actions,
                resources=resources,
                condition=random_condition(rng),
                exceptions=exceptions,
            )
        )
    return Policy(statements=statements)


def test_fast_evaluator_matches_naive_evaluator_on_thousands_of_policies() -> None:
    rng = random.Random(0x6A7E2026)
    checked_requests = 0
    checked_policies = 0

    for world_index in range(N_WORLDS):
        world = random_world(rng)
        fast = Evaluator(world)
        naive = NaiveEvaluator(world)
        users = list(world.users)
        resources = list(world.resources)
        actions = list(world.actions)

        for policy_index in range(POLICIES_PER_WORLD):
            policy = random_policy(rng, world)
            compiled = fast.compile(policy)
            checked_policies += 1

            for request_index in range(REQUESTS_PER_POLICY):
                user_id = rng.choice(users)
                action = rng.choice(actions)
                resource_id = rng.choice(resources)
                context = CONTEXTS[request_index % len(CONTEXTS)]

                fast_decision = fast.decide(
                    compiled, user_id, action, resource_id, context
                )
                naive_decision = naive.decide(
                    policy, user_id, action, resource_id, context
                )
                checked_requests += 1

                if (
                    fast_decision.effect,
                    fast_decision.statement_id,
                ) != naive_decision:
                    pytest.fail(
                        "evaluator mismatch\n"
                        f"world_index={world_index} policy_index={policy_index} "
                        f"request_index={request_index}\n"
                        f"request=({user_id!r}, {action!r}, {resource_id!r}, "
                        f"{context!r})\n"
                        f"fast={fast_decision}\n"
                        f"naive={naive_decision}\n"
                        f"world={world.model_dump_json()}\n"
                        f"policy={policy.model_dump_json()}"
                    )

    assert checked_policies == TOTAL_POLICIES == 2000
    assert checked_requests == TOTAL_REQUESTS == 400000
