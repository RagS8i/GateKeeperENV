"""Reference-solution baseline for any task.

Run against local services:

    SANDBOX_URL=http://127.0.0.1:8000 \
    VERIFIER_URL=http://127.0.0.1:8001 \
    TASK_ID=med_01 SEED=0 \
    .venv/bin/python scripts/reference_solution.py

The automatic gate uses the same `oracle_actions` path in `harness/agents.py`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.tasks import load_task_configs
from harness.agents import oracle_actions
from harness.env import Env
from harness.limits import Limits
from harness.runner import run_scripted
from harness.service_clients import EnvClient, VerifierClient


def main() -> int:
    task_id = os.environ.get("TASK_ID", "med_01")
    seed = int(os.environ["SEED"]) if os.environ.get("SEED") else None
    configs = {config.task_id: config for config in load_task_configs()}
    if task_id not in configs:
        print(f"unknown task: {task_id}", file=sys.stderr)
        return 3
    sandbox_url = os.environ.get("SANDBOX_URL") or os.environ.get("ENV_URL")
    verifier_url = os.environ.get("VERIFIER_URL")
    if not sandbox_url or not verifier_url:
        print("set SANDBOX_URL and VERIFIER_URL", file=sys.stderr)
        return 3
    env = Env(EnvClient(sandbox_url), VerifierClient(verifier_url), task_id)
    env.reset(seed, task_id=task_id)
    actions = oracle_actions(env)
    result = run_scripted(
        env,
        task_id=task_id,
        tier=configs[task_id].tier,
        seed=seed,
        actions=actions,
        source="oracle",
        limits=Limits(),
        log=None,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if float(result["score"]) >= 0.95 else 2


if __name__ == "__main__":
    raise SystemExit(main())
