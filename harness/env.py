"""The environment: reset / step / score, backed by the sandbox and verifier.

This is the single public seam between the harness and the two services.  The
sandbox only ever receives a sanitized task bundle; the verifier keeps the
hidden answer key.  Binding the task at construction lets callers use the
documented ``env.reset(seed=...)`` form, while ``task_id`` can still be
overridden for ``MODE=all`` and the baseline gate.
"""

from __future__ import annotations

from typing import Any

from .service_clients import EnvClient, VerifierClient


class Env:
    def __init__(
        self,
        sandbox: EnvClient,
        verifier: VerifierClient,
        task_id: str,
        *,
        tool_timeout: float = 60.0,
        score_timeout: float = 120.0,
    ) -> None:
        self.sandbox = sandbox
        self.verifier = verifier
        self.task_id = task_id
        self.tool_timeout = tool_timeout
        self.score_timeout = score_timeout
        self.bundle: dict[str, Any] | None = None
        self.seed: int | None = None

    def reset(
        self,
        seed: int | None = None,
        *,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Start a clean episode. The same seed gives the same start and hash."""
        effective_task = task_id or self.task_id
        bundle = self.verifier.reset(effective_task, seed)
        response = self.sandbox.reset(bundle)
        state = response.get("state") or {}
        self.task_id = effective_task
        self.bundle = bundle
        self.seed = bundle.get("seed")
        return {
            "observation": response["observation"],
            "state": state,
            "state_hash": state.get("state_hash"),
            "bundle": bundle,
            "task_id": effective_task,
            "seed": self.seed,
        }

    def step(
        self,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.sandbox.tool(
            tool,
            args,
            timeout=self.tool_timeout if timeout is None else timeout,
        )

    def state(self, *, timeout: float | None = None) -> dict[str, Any]:
        return self.sandbox.state(
            timeout=self.tool_timeout if timeout is None else timeout
        )

    def state_hash(self) -> str | None:
        return (self.state() or {}).get("state_hash")

    def _current_policy(self) -> dict[str, Any]:
        policy = (self.state() or {}).get("policy")
        return policy if isinstance(policy, dict) else {}

    def score(
        self,
        policy: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.verifier.score(
            self.task_id,
            policy if policy is not None else self._current_policy(),
            self.seed,
            timeout=self.score_timeout if timeout is None else timeout,
        )

    def shadow(
        self,
        policy: dict[str, Any] | None = None,
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        return self.verifier.shadow(
            self.task_id,
            policy if policy is not None else self._current_policy(),
            self.seed,
            timeout=timeout,
        )

    def oracle(self, *, timeout: float = 60.0) -> list[dict[str, Any]]:
        return self.verifier.oracle(
            self.task_id,
            self.seed,
            timeout=timeout,
        )["actions"]
