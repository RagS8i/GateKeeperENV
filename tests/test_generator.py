from __future__ import annotations

import json

import pytest

from sandbox.env import Environment
from verifier.generator import generate_task, load_task_configs
from common.models import Policy, Statement


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_task_configs_are_complete(configs):
    assert len(configs) == 12
    assert len({config.task_id for config in configs}) == 12
    assert {config.tier for config in configs} == {"easy", "medium", "hard"}
    for config in configs:
        assert config.statements >= len(config.mutation_types) + 2
        assert config.users > 0 and config.resources > 0
        assert config.contexts


@pytest.mark.parametrize("task_id", ["easy_01", "med_01", "hard_01"])
def test_generated_task_invariants(configs, task_id):
    config = next(item for item in configs if item.task_id == task_id)
    task = generate_task(config, seed_override=0)

    assert task.score(task.correct_policy)["score"] == 1.0
    assert task.score(task.broken_policy)["score"] == 0.0

    empty = task.score(Policy(statements=[]))
    assert empty["score"] == 0.0
    assert empty["false_denies"] > 0

    allow_all = task.score(
        Policy(
            statements=[
                Statement(
                    id="allow-all",
                    effect="allow",
                    principals=["*"],
                    actions=["*"],
                    resources=["*"],
                )
            ]
        )
    )
    assert allow_all["score"] == 0.0
    assert allow_all["false_allows"] > 0

    deny_all = task.score(
        Policy(
            statements=[
                Statement(
                    id="deny-all",
                    effect="deny",
                    principals=["*"],
                    actions=["*"],
                    resources=["*"],
                )
            ]
        )
    )
    assert deny_all["score"] == 0.0
    assert deny_all["false_denies"] > 0

    assert task.shadow_score(task.correct_policy)["score"] >= 0.95
    assert task.shadow_score(task.broken_policy)["score"] == 0.0
    assert task.e_start >= 20
    assert len(task.mutations) == len(config.mutation_types)
    assert all(record.sample_impact > 0 for record in task.mutations)
    assert task.sample_indices
    assert len(task.reference_sample) == len(task.sample_indices)


def test_generation_is_deterministic(configs):
    config = next(item for item in configs if item.task_id == "easy_02")
    first = generate_task(config, seed_override=123)
    second = generate_task(config, seed_override=123)

    assert first.state_hash == second.state_hash
    assert first.requirements == second.requirements
    assert first.visible_examples == second.visible_examples
    assert first.e_start == second.e_start
    assert [r.as_dict() for r in first.mutations] == [
        r.as_dict() for r in second.mutations
    ]
    assert first.correct_policy == second.correct_policy
    assert first.broken_policy == second.broken_policy


def test_sanitized_bundle_contains_no_answers(configs):
    config = next(item for item in configs if item.task_id == "easy_03")
    task = generate_task(config, seed_override=0)
    bundle = task.sanitized_bundle()

    assert set(bundle) == {
        "task_id",
        "tier",
        "seed",
        "world",
        "broken_policy",
        "requirements",
        "visible_examples",
        "state_hash",
    }
    forbidden = {
        "correct_policy",
        "reference_decisions",
        "reference_sample",
        "mutations",
        "matrix",
        "e_start",
        "sample_indices",
        "shadow_indices",
        "oracle_actions",
    }

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key not in forbidden, key
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(bundle)
    encoded = json.dumps(bundle).lower()
    for needle in ("oracle", "mutation", "reference_decision", "shadow"):
        assert needle not in encoded


def test_equivalent_policy_scores_identically(configs):
    config = next(item for item in configs if item.task_id == "easy_04")
    task = generate_task(config, seed_override=0)
    reordered = [
        statement.model_copy(
            update={
                "principals": list(reversed(statement.principals)),
                "actions": list(reversed(statement.actions)),
                "resources": list(reversed(statement.resources)),
            }
        )
        for statement in reversed(task.correct_policy.statements)
    ]
    equivalent = Policy(statements=reordered)
    assert task.score(equivalent) == task.score(task.correct_policy)


def test_requirements_cover_policy_details(configs):
    config = next(item for item in configs if item.task_id == "med_02")
    task = generate_task(config, seed_override=0)
    text = task.requirements
    lowered = " ".join(text.lower().split())

    for group_id in task.world.groups:
        assert group_id in text
    for statement in task.correct_policy.statements:
        if statement.condition:
            for key in statement.condition:
                assert key in lowered
        if statement.exceptions:
            assert "does not apply" in lowered
    assert "default deny" in lowered or "denied by default" in lowered
    assert "deny" in lowered and "allow" in lowered


def test_visible_example_overfit_policy_fails(configs):
    config = next(item for item in configs if item.task_id == "easy_01")
    task = generate_task(config, seed_override=0)
    statements = []
    for index, example in enumerate(task.visible_examples):
        statements.append(
            Statement(
                id=f"v{index:03d}",
                effect="allow" if example["expected"] == "allow" else "deny",
                principals=[f"user:{example['user']}"],
                actions=[str(example["action"])],
                resources=[f"resource:{example['resource']}"],
                condition=dict(example["context"]) or None,
            )
        )
    overfit = Policy(statements=statements)
    result = task.score(overfit)
    assert result["score"] < 0.95
    assert result["false_allows"] + result["false_denies"] > 0


def test_inverting_policy_is_not_a_shortcut(configs):
    config = next(item for item in configs if item.task_id == "easy_01")
    task = generate_task(config, seed_override=0)
    inverted = Policy(
        statements=[
            statement.model_copy(
                update={
                    "id": f"i{index:03d}",
                    "effect": "deny" if statement.effect == "allow" else "allow",
                }
            )
            for index, statement in enumerate(task.correct_policy.statements)
        ]
    )
    assert task.score(inverted)["score"] < 0.5


def test_state_hash_matches_environment(configs):
    config = next(item for item in configs if item.task_id == "easy_02")
    task = generate_task(config, seed_override=7)
    env = Environment(task.sanitized_bundle())
    assert env.state_hash == task.state_hash
    assert env.current_policy == task.broken_policy
    assert env.world == task.world
