from __future__ import annotations

import copy
import json

import pytest

from gatekeeper.env_core import Environment
from gatekeeper.generator import generate_task, load_task_configs


@pytest.fixture(scope="module")
def generated():
    config = next(
        item for item in load_task_configs() if item.task_id == "easy_01"
    )
    return generate_task(config, seed_override=0)


@pytest.fixture()
def env(generated):
    return Environment(generated.sanitized_bundle())


def _json_content(result):
    assert result["ok"] is True
    return json.loads(result["content"])


def test_reset_observation_and_state(env, generated):
    assert env.task_id == "easy_01"
    assert env.tier == "easy"
    assert env.state_hash == generated.state_hash
    assert "Access requirements" in env.observation()
    assert env.state()["submitted"] is False


def test_bad_state_hash_is_rejected(generated):
    bundle = copy.deepcopy(generated.sanitized_bundle())
    bundle["state_hash"] = "00000000"
    with pytest.raises(ValueError, match="state hash"):
        Environment(bundle)


def test_requirements_and_policy_listing(env):
    result = env.step("get_requirements")
    assert result["ok"] is True
    assert "default" in result["content"].lower()

    listing = _json_content(env.step("list_policies", {"page": 1}))
    assert listing["page"] == 1
    assert listing["statement_count"] == len(env.current_policy.statements)
    assert listing["statements"]


def test_lookup_user_group_resource_tag_and_action(env):
    user = sorted(env.world.users)[0]
    group = sorted(env.world.groups)[0]
    resource = sorted(env.world.resources)[0]
    action = env.world.actions[0]
    tag = sorted(env.world.resources[resource].tags)[0]

    user_info = _json_content(
        env.step("lookup", {"kind": "user", "id": user})
    )
    assert user_info["exists"] is True
    assert "effective_groups" in user_info

    group_info = _json_content(
        env.step("lookup", {"kind": "group", "id": f"group:{group}"})
    )
    assert group_info["exists"] is True
    assert "members" in group_info

    resource_info = _json_content(
        env.step("lookup", {"kind": "resource", "id": resource})
    )
    assert resource_info["exists"] is True
    assert resource_info["tags"] == env.world.resources[resource].tags

    tag_info = _json_content(env.step("lookup", {"kind": "tag", "id": tag}))
    assert tag_info["exists"] is True

    action_info = _json_content(
        env.step("lookup", {"kind": "action", "id": action})
    )
    assert action_info["exists"] is True

    missing = _json_content(
        env.step("lookup", {"kind": "user", "id": "nobody"})
    )
    assert missing["exists"] is False


def test_simulate_returns_a_valid_decision(env):
    user = sorted(env.world.users)[0]
    resource = sorted(env.world.resources)[0]
    action = env.world.actions[0]
    result = _json_content(
        env.step(
            "simulate",
            {
                "principal": f"user:{user}",
                "action": action,
                "resource": f"resource:{resource}",
                "context": {"mfa": True, "network": "internal"},
            },
        )
    )
    assert result["effect"] in {"allow", "deny"}
    assert result["request"]["principal"] == f"user:{user}"
    assert result["request"]["context"] == {
        "mfa": "true",
        "network": "internal",
    }


def test_put_delete_undo_and_diff(env, generated):
    user = sorted(env.world.users)[0]
    resource = sorted(env.world.resources)[0]
    before = env.state_hash
    statement = {
        "id": "x001",
        "effect": "allow",
        "principals": [f"user:{user}"],
        "actions": ["read"],
        "resources": [f"resource:{resource}"],
        "condition": None,
        "exceptions": [],
    }
    added = _json_content(env.step("put_policy", {"statement": statement}))
    assert added["action"] == "added"
    assert env.state_hash != before

    diff = _json_content(env.step("diff"))
    assert diff["added_count"] == 1
    assert diff["added"][0]["id"] == "x001"

    undone = _json_content(env.step("undo"))
    assert undone["undone"] is True
    assert env.state_hash == before

    statement_id = sorted(env.current_policy.statements, key=lambda s: s.id)[0].id
    deleted = _json_content(env.step("delete_policy", {"id": statement_id}))
    assert deleted["deleted"] is True
    assert deleted["statement_id"] == statement_id
    assert env.state_hash != before
    env.step("undo")
    assert env.state_hash == before


def test_replace_existing_statement(env):
    statement = sorted(env.current_policy.statements, key=lambda s: s.id)[0]
    replacement = statement.model_dump(mode="json")
    replacement["effect"] = "deny" if statement.effect == "allow" else "allow"
    result = _json_content(
        env.step("put_policy", {"statement": replacement})
    )
    assert result["action"] == "replaced"
    assert len(env.current_policy.statements) == len(
        env.initial_policy.statements
    )
    assert env.current_policy.statements != env.initial_policy.statements


def test_invalid_statements_are_rejected(env):
    bad_selector = {
        "id": "bad1",
        "effect": "allow",
        "principals": ["user:does-not-exist"],
        "actions": ["read"],
        "resources": ["*"],
        "condition": None,
        "exceptions": [],
    }
    result = env.step("put_policy", {"statement": bad_selector})
    assert result["ok"] is False
    assert "unknown user" in result["content"]

    bad_context = env.step(
        "simulate",
        {
            "principal": "user:nobody",
            "action": "read",
            "resource": "resource:nothing",
        },
    )
    assert bad_context["ok"] is False


def test_check_examples_returns_per_example_results(env):
    result = _json_content(env.step("check_examples"))
    assert result["total"] == len(env.visible_examples)
    assert 0 <= result["correct"] <= result["total"]
    assert len(result["examples"]) == result["total"]


def test_submit_freezes_the_episode(env):
    assert env.step("submit")["ok"] is True
    assert env.state()["submitted"] is True
    after = env.step("list_policies", {"page": 1})
    assert after["ok"] is False
    assert "already" in after["content"]
    assert "submitted" in after["content"]


def test_unknown_tool_and_bad_arguments(env):
    assert env.step("not_a_tool")["ok"] is False
    assert env.step("lookup", [])["ok"] is False
    assert env.step("undo")["ok"] is False


def test_output_is_clipped(generated):
    bundle = copy.deepcopy(generated.sanitized_bundle())
    bundle["requirements"] = "R" * 5000
    env = Environment(bundle)
    result = env.step("get_requirements")
    assert result["ok"] is True
    assert len(result["content"]) <= 4000
    assert result["content"].endswith("...[truncated]")
