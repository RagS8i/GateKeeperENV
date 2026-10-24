from __future__ import annotations

import json

from gatekeeper.harness import run


def _configure(monkeypatch, service_urls, fake_model_url, tmp_path, **overrides):
    values = {
        "ENV_URL": service_urls.env,
        "VERIFIER_URL": service_urls.verifier,
        "MODEL": "fake/native_submit",
        "API_KEY": "test-key",
        "BASE_URL": fake_model_url,
        "TASK_ID": "easy_01",
        "SEED": "0",
        "MODE": "full",
        "LOGS_DIR": str(tmp_path / "logs"),
        "APPEND_FILE": "",
        "MAX_STEPS": "8",
    }
    values.update(overrides)
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _latest_run(tmp_path):
    paths = sorted((tmp_path / "logs").iterdir())
    assert len(paths) == 1
    return paths[0]


def test_full_run_with_fake_model(
    monkeypatch, service_urls, fake_model_url, tmp_path
):
    _configure(monkeypatch, service_urls, fake_model_url, tmp_path)
    assert run.main() == 0

    out = _latest_run(tmp_path)
    gate = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert summary["status"] == "completed"
    assert summary["task"] == "easy_01"
    assert summary["termination"] == "failure"
    assert summary["score"] == 0.0
    assert summary["model_called"] is True
    assert (out / "run.jsonl").is_file()
    assert (out / "episode.jsonl").is_file()
    assert "test-key" not in json.dumps(summary)


def test_baselines_mode_does_not_need_a_model(
    monkeypatch, service_urls, fake_model_url, tmp_path, configs
):
    easy = tuple(config for config in configs if config.task_id == "easy_01")
    monkeypatch.setattr(run, "load_task_configs", lambda: easy)
    _configure(
        monkeypatch,
        service_urls,
        fake_model_url,
        tmp_path,
        MODE="baselines",
        MODEL="",
        API_KEY="",
    )
    assert run.main() == 0
    out = _latest_run(tmp_path)
    gate = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert not (out / "episode.jsonl").exists()


def test_bad_configuration_exits_three(
    monkeypatch, service_urls, fake_model_url, tmp_path
):
    _configure(
        monkeypatch,
        service_urls,
        fake_model_url,
        tmp_path,
        TASK_ID="missing_task",
    )
    assert run.main() == 3

    _configure(
        monkeypatch,
        service_urls,
        fake_model_url,
        tmp_path,
        MODE="not-a-mode",
    )
    assert run.main() == 3

    _configure(
        monkeypatch,
        service_urls,
        fake_model_url,
        tmp_path,
        MODEL="real-model",
        API_KEY="",
    )
    assert run.main() == 3

    _configure(
        monkeypatch,
        service_urls,
        fake_model_url,
        tmp_path,
        APPEND_FILE="missing.txt",
    )
    assert run.main() == 3
