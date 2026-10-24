from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pytest

from gatekeeper.generator import load_task_configs
from gatekeeper.http_util import make_handler, start_http_server
from gatekeeper.services.env_server import EnvService
from gatekeeper.services.verifier_server import VerifierService
from gatekeeper.harness.fake_model_server import start_fake_model


@dataclass(frozen=True)
class ServiceUrls:
    env: str
    verifier: str


@pytest.fixture()
def service_urls() -> Iterator[ServiceUrls]:
    verifier_service = VerifierService()
    env_service = EnvService()
    verifier_handler = make_handler(
        {
            "/reset": verifier_service.reset,
            "/score": verifier_service.score,
            "/shadow": verifier_service.shadow,
            "/oracle": verifier_service.oracle,
            "/tasks": verifier_service.list_tasks,
        }
    )
    env_handler = make_handler(
        {
            "/reset": env_service.reset,
            "/tool": lambda body: env_service.tool(
                str(body.get("tool", "")), body.get("args")
            ),
            "/state": lambda _body: env_service.state(),
        }
    )
    verifier_server, verifier_thread, verifier_url = start_http_server(
        verifier_handler, host="127.0.0.1", port=0
    )
    env_server, env_thread, env_url = start_http_server(
        env_handler, host="127.0.0.1", port=0
    )
    try:
        yield ServiceUrls(
            env=f"http://127.0.0.1:{env_url}",
            verifier=f"http://127.0.0.1:{verifier_url}",
        )
    finally:
        env_server.shutdown()
        verifier_server.shutdown()
        env_server.server_close()
        verifier_server.server_close()
        env_thread.join(timeout=5)
        verifier_thread.join(timeout=5)


@pytest.fixture()
def fake_model_url() -> Iterator[str]:
    server, thread, url = start_fake_model()
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="session")
def configs():
    return load_task_configs()
