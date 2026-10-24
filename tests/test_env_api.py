"""The public seeded-reset contract: env.reset(seed)."""

from __future__ import annotations

from harness.env import Env
from harness.service_clients import EnvClient, VerifierClient


def _env(service_urls, task_id: str = "med_01") -> Env:
    return Env(
        EnvClient(service_urls.env),
        VerifierClient(service_urls.verifier),
        task_id,
    )


def test_reset_same_seed_same_state_and_hash(service_urls):
    env = _env(service_urls)

    first = env.reset(seed=7)
    second = env.reset(seed=7)

    assert first["state_hash"] == second["state_hash"]
    assert first["observation"] == second["observation"]
    assert first["seed"] == 7


def test_reset_different_seed_different_variant(service_urls):
    env = _env(service_urls)

    assert env.reset(seed=7)["state_hash"] != env.reset(seed=8)["state_hash"]


def test_reset_accepts_positional_and_default_seed(service_urls):
    env = _env(service_urls)

    assert env.reset(7)["state_hash"] == env.reset(seed=7)["state_hash"]
    assert env.reset()["state_hash"]


def test_reset_can_switch_task(service_urls):
    env = _env(service_urls)

    result = env.reset(seed=7, task_id="easy_01")

    assert result["task_id"] == "easy_01"
    assert env.task_id == "easy_01"
    assert result["state_hash"]
