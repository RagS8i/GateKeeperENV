from __future__ import annotations

from common.models import Policy
from verifier.core import TaskService


def test_verifier_lists_all_tasks(configs):
    service = TaskService()
    listing = service.list_tasks()
    assert listing["count"] == 20
    assert {task["task_id"] for task in listing["tasks"]} == {
        config.task_id for config in configs
    }
    assert all("hidden" not in str(task).lower() for task in listing["tasks"])


def test_reset_is_sanitized_and_cached(configs):
    service = TaskService()
    first = service.reset("easy_01", 0)
    second = service.reset("easy_01", 0)
    assert first == second
    assert set(first) == {
        "task_id",
        "tier",
        "seed",
        "world",
        "broken_policy",
        "requirements",
        "visible_examples",
        "state_hash",
    }
    assert len(service._cache) == 1


def test_score_oracle_broken_and_trivial(configs):
    service = TaskService()
    bundle = service.reset("easy_01", 0)
    oracle = service.oracle("easy_01", 0)
    correct = Policy.model_validate(
        {"statements": [action["args"]["statement"] for action in oracle["actions"] if action["tool"] == "put_policy"]}
    )
    perfect = service.score("easy_01", correct.model_dump(mode="json"), 0)
    broken = service.score("easy_01", bundle["broken_policy"], 0)
    empty = service.score("easy_01", {"statements": []}, 0)
    assert perfect["score"] == 1.0
    assert broken["score"] == 0.0
    assert empty["score"] == 0.0
    assert perfect["total_requests"] == empty["total_requests"]


def test_oracle_has_delete_and_put_actions(configs):
    service = TaskService()
    bundle = service.reset("easy_02", 0)
    oracle = service.oracle("easy_02", 0)["actions"]
    assert any(action["tool"] == "delete_policy" for action in oracle)
    assert any(action["tool"] == "put_policy" for action in oracle)
    assert oracle[-1]["tool"] == "submit"
    assert len(oracle) == (
        len(bundle["broken_policy"]["statements"])
        + len([action for action in oracle if action["tool"] == "put_policy"])
        + 1
    )


def test_shadow_score_is_cheap_and_meaningful(configs):
    service = TaskService()
    bundle = service.reset("easy_02", 0)
    oracle = service.oracle("easy_02", 0)
    correct = Policy.model_validate(
        {"statements": [action["args"]["statement"] for action in oracle["actions"] if action["tool"] == "put_policy"]}
    )
    perfect = service.shadow("easy_02", correct.model_dump(mode="json"), 0)
    broken = service.shadow("easy_02", bundle["broken_policy"], 0)
    assert perfect["score"] >= 0.95
    assert broken["score"] == 0.0
    assert perfect["sample_size"] > 0
