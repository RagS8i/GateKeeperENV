"""Run the four-baseline gate for every task using local in-process services."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifier.generator import load_task_configs
from harness.limits import Limits
from harness.env import Env
from harness.runner import run_gate
from harness.service_clients import EnvClient, VerifierClient
from common.http_util import make_handler, start_http_server
from sandbox.server import EnvService
from verifier.server import VerifierService


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("baseline-results"))
    parser.add_argument("--tasks", default="")
    args = parser.parse_args()

    configs = load_task_configs()
    by_id = {config.task_id: config for config in configs}
    task_ids = (
        [item.strip() for item in args.tasks.split(",") if item.strip()]
        if args.tasks
        else list(by_id)
    )
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        print(f"unknown tasks: {', '.join(missing)}", file=sys.stderr)
        return 3

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
    verifier_server, verifier_thread, verifier_port = start_http_server(
        verifier_handler, host="127.0.0.1", port=0
    )
    env_server, env_thread, env_port = start_http_server(
        env_handler, host="127.0.0.1", port=0
    )
    try:
        env = Env(
            EnvClient(f"http://127.0.0.1:{env_port}"),
            VerifierClient(f"http://127.0.0.1:{verifier_port}"),
            task_ids[0],
        )
        args.out.mkdir(parents=True, exist_ok=True)
        summary = run_gate(
            env,
            configs=configs,
            task_ids=task_ids,
            seed_override=args.seed,
            limits=Limits(),
            gate_dir=args.out,
        )
    finally:
        env_server.shutdown()
        verifier_server.shutdown()
        env_server.server_close()
        verifier_server.server_close()
        env_thread.join(timeout=5)
        verifier_thread.join(timeout=5)

    print(json.dumps(
        {
            "passed": summary["passed"],
            "task_count": summary["task_count"],
            "seed": args.seed,
            "gate_json": str(args.out / "gate.json"),
        },
        indent=2,
    ))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
