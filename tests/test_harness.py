from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from gatekeeper.harness.append import ConfigError, load_append_text, with_append
from gatekeeper.harness.clients import MockClient, OpenAIClient
from gatekeeper.harness.limits import Limits
from gatekeeper.harness.logging import JsonlWriter
from gatekeeper.harness.parser import (
    parse_arguments,
    parse_assistant_message,
    parse_text_tool_calls,
)
from gatekeeper.harness.runner import run_episode, run_gate
from gatekeeper.harness.service_clients import EnvClient, VerifierClient


class SleepingClient:
    def start(self, _task_context):
        return None

    def finish(self):
        return None

    def complete(self, _messages, _tools, *, timeout):
        del timeout
        time.sleep(0.08)
        return parse_assistant_message(
            {"role": "assistant", "content": "still thinking"}
        )


def _clients(service_urls):
    return (
        EnvClient(service_urls.env),
        VerifierClient(service_urls.verifier),
    )


def _run(
    service_urls,
    client,
    model,
    *,
    task_id="easy_01",
    tier="easy",
    limits=None,
    oracle=False,
):
    env, verifier = _clients(service_urls)
    bundle = verifier.reset(task_id, 0)
    oracle_actions = verifier.oracle(task_id, 0)["actions"] if oracle else None
    with tempfile.TemporaryDirectory() as directory:
        with JsonlWriter(Path(directory) / "episode.jsonl") as log:
            return run_episode(
                env,
                verifier,
                task_id=task_id,
                tier=tier,
                seed=0,
                bundle=bundle,
                client=client,
                model_name=model,
                append_text="",
                limits=limits or Limits(),
                log=log,
                oracle_actions=oracle_actions,
            )


def _run_logged(service_urls, client, model, tmp_path, **kwargs):
    env, verifier = _clients(service_urls)
    task_id = kwargs.pop("task_id", "easy_01")
    tier = kwargs.pop("tier", "easy")
    bundle = verifier.reset(task_id, 0)
    log_path = tmp_path / "episode.jsonl"
    with JsonlWriter(log_path) as log:
        result = run_episode(
            env,
            verifier,
            task_id=task_id,
            tier=tier,
            seed=0,
            bundle=bundle,
            client=client,
            model_name=model,
            append_text="",
            limits=kwargs.pop("limits", Limits()),
            log=log,
            oracle_actions=kwargs.pop("oracle_actions", None),
        )
    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return result, events


def test_append_contract(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "append.txt").write_bytes(b"\xef\xbb\xbfPrefer small edits.\n")

    text, info = load_append_text(None, inputs_dir=inputs)
    assert text == "Prefer small edits.\n"
    assert info and info["file"] == "inputs/append.txt"

    (inputs / "named.txt").write_text("Use diff.", encoding="utf-8")
    text, info = load_append_text("inputs/named.txt", inputs_dir=inputs)
    assert text == "Use diff."
    assert info and info["chars"] == len(text)

    (inputs / "empty.txt").write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_append_text("empty.txt", inputs_dir=inputs)
    with pytest.raises(ConfigError, match="not found"):
        load_append_text("missing.txt", inputs_dir=inputs)
    with pytest.raises(ConfigError, match="inside inputs"):
        load_append_text("../secret.txt", inputs_dir=inputs)

    (inputs / "append.txt").unlink()
    assert load_append_text(None, inputs_dir=inputs) == ("", None)
    assert with_append("base", "extra") == "base\n\nextra"
    assert with_append("base", "") == "base"


def test_parser_native_legacy_and_text():
    native = parse_assistant_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "put_policy",
                        "arguments": '{"statement": {"id": "s1"}}',
                    },
                }
            ],
        }
    )
    assert native.calls[0].name == "put_policy"
    assert native.calls[0].arguments == {"statement": {"id": "s1"}}

    legacy = parse_assistant_message(
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "submit", "arguments": "{}"},
        }
    )
    assert legacy.calls[0].name == "submit"

    text = parse_text_tool_calls(
        'I will inspect.\n```json\n{"tool":"get_requirements","args":{}}\n```'
    )
    assert [call.name for call in text] == ["get_requirements"]

    many = parse_text_tool_calls(
        '[{"tool":"list_policies","args":{"page":1}},'
        '{"tool":"simulate","args":{"principal":"user:u1",'
        '"action":"read","resource":"resource:r1"}}]'
    )
    assert [call.name for call in many] == ["list_policies", "simulate"]

    bad, raw, error = parse_arguments("{oops")
    assert bad is None
    assert raw == "{oops"
    assert error is not None


def test_mock_solver_succeeds(service_urls):
    result = _run(service_urls, MockClient("solver"), "mock/solver", oracle=True)
    assert result["termination"] == "success"
    assert result["score"] == 1.0
    assert result["steps"] <= 8


def test_idle_submit_is_a_failure(service_urls):
    result = _run(service_urls, MockClient("idle"), "mock/idle")
    assert result["termination"] == "failure"
    assert result["score"] == 0.0
    assert result["submitted"] is True


@pytest.mark.parametrize("mode", ["allow_all", "deny_all"])
def test_trivial_mock_policies_fail(service_urls, mode):
    result = _run(service_urls, MockClient(mode), f"mock/{mode}")
    assert result["termination"] == "failure"
    assert result["score"] == 0.0


def test_fake_native_and_text_tool_calls(
    service_urls, fake_model_url, tmp_path
):
    for model in ("fake/native_submit", "fake/text_submit", "fake/legacy_submit"):
        client = OpenAIClient(model, api_key="test", base_url=fake_model_url)
        result, events = _run_logged(
            service_urls,
            client,
            model,
            tmp_path / f"{model.replace('/', '_')}.jsonl",
        )
        assert result["termination"] == "failure"
        assert result["submitted"] is True
        tool_results = [event for event in events if event["event"] == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool"] == "submit"


def test_malformed_arguments_are_logged(service_urls, fake_model_url, tmp_path):
    client = OpenAIClient(
        "fake/malformed", api_key="test", base_url=fake_model_url
    )
    result, events = _run_logged(
        service_urls, client, "fake/malformed", tmp_path
    )
    assert result["suspicion"]["parse_failures"] >= 1
    results = [event for event in events if event["event"] == "tool_result"]
    assert any(str(event.get("content", "")).startswith("ERROR") for event in results)


def test_tool_call_limit_is_enforced(service_urls, fake_model_url, tmp_path):
    client = OpenAIClient(
        "fake/25calls", api_key="test", base_url=fake_model_url
    )
    result, events = _run_logged(
        service_urls, client, "fake/25calls", tmp_path, task_id="easy_02"
    )
    tool_results = [event for event in events if event["event"] == "tool_result"]
    assert len(tool_results) == 26
    assert tool_results[-1]["tool"] == "submit"
    assert sum(
        "too many tool calls" in str(event.get("content", "")).lower()
        for event in tool_results
    ) == 15
    assert result["suspicion"]["tool_calls"] == 26


def test_no_tool_reply_hits_step_cap(service_urls, fake_model_url, tmp_path):
    client = OpenAIClient(
        "fake/no_tool", api_key="test", base_url=fake_model_url
    )
    result, events = _run_logged(
        service_urls,
        client,
        "fake/no_tool",
        tmp_path,
        limits=Limits(max_steps=3),
    )
    assert result["termination"] == "step_cap"
    assert result["steps"] == 3
    assert result["suspicion"]["no_tool_replies"] == 3
    assert any(event["event"] == "no_tool_call" for event in events)


def test_wall_clock_is_enforced(service_urls, tmp_path):
    result, _ = _run_logged(
        service_urls,
        SleepingClient(),
        "test/slow",
        tmp_path,
        limits=Limits(max_seconds=0.05, max_steps=10),
    )
    assert result["termination"] == "time_cap"


def test_gate_passes_for_one_task(service_urls, tmp_path, configs):
    env, verifier = _clients(service_urls)
    summary = run_gate(
        env,
        verifier,
        configs=configs,
        task_ids=["easy_01"],
        seed_override=0,
        limits=Limits(),
        gate_dir=tmp_path,
    )
    assert summary["passed"] is True
    assert (tmp_path / "gate.json").is_file()
    data = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert data["tasks"][0]["scores"]["oracle"] == 1.0
    assert data["tasks"][0]["scores"]["noop"] == 0.0
