"""Entry point for the one-command Gatekeeper run."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..generator import DEMO_TASK, load_task_configs
from ..http_util import wait_for_health
from .append import ConfigError, load_append_text
from .clients import MockClient, OpenAIClient
from .limits import Limits
from .logging import JsonlWriter, print_line, run_id, utc_now
from .runner import run_episode, run_gate
from .service_clients import EnvClient, VerifierClient

MODES = ("full", "all", "baselines")
GATE_BUDGET_SECONDS = 120


def setting(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def make_client(model: str, api_key: str, base_url: str | None, max_tokens: int | None):
    if model.startswith("mock/"):
        mode = model.split("/", 1)[1]
        return MockClient(mode)
    if not api_key:
        raise ValueError(f"API_KEY is required for model {model!r}")
    return OpenAIClient(
        model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> int:
    configs = load_task_configs()
    by_id = {config.task_id: config for config in configs}
    task_id = setting("TASK_ID", DEMO_TASK)
    mode = setting("MODE", "full")
    model = setting("MODEL", "mock/solver")
    api_key = setting("API_KEY")
    base_url = setting("BASE_URL") or None
    try:
        seed = int(setting("SEED", "0"))
        max_tokens = int(setting("MAX_TOKENS", "0") or 0) or None
        max_steps = int(setting("MAX_STEPS", "40"))
        max_seconds = float(setting("MAX_SECONDS", "600"))
    except ValueError:
        print_line("configuration error: SEED, MAX_TOKENS, MAX_STEPS, and MAX_SECONDS must be numeric")
        return 3

    if mode not in MODES:
        print_line(f"unknown MODE {mode!r}. Available: {', '.join(MODES)}")
        return 3
    if mode == "full" and task_id not in by_id:
        print_line(
            f"unknown TASK_ID {task_id!r}. Available: {', '.join(by_id)}"
        )
        return 3

    try:
        client = (
            None
            if mode == "baselines"
            else make_client(model, api_key, base_url, max_tokens)
        )
    except ValueError as exc:
        print_line(f"configuration error: {exc}")
        return 3

    try:
        append_text, append_info = (
            ("", None)
            if mode == "baselines"
            else load_append_text(setting("APPEND_FILE"), inputs_dir=Path("inputs"))
        )
    except ConfigError as exc:
        print_line(f"configuration error: {exc}")
        return 3

    run = run_id()
    out = Path(setting("LOGS_DIR", "logs")) / run
    out.mkdir(parents=True, exist_ok=True)
    limits = Limits(max_steps=max_steps, max_seconds=max_seconds)
    run_log = JsonlWriter(out / "run.jsonl")
    run_log.write(
        {
            "event": "run_start",
            "run_id": run,
            "mode": mode,
            "task_id": task_id,
            "seed": seed,
            "model": model,
            "base_url": base_url,
            "append": append_info,
            "limits": dataclasses.asdict(limits),
        }
    )
    if append_info:
        (out / "append.txt").write_text(append_text, encoding="utf-8")
        print_line(
            f"extra instructions: {append_info['file']}  "
            f"{append_info['chars']} chars  sha256 {append_info['sha256'][:12]}"
        )
    elif mode != "baselines":
        print_line("extra instructions: none")

    env_url = setting("ENV_URL", "http://env:8000")
    verifier_url = setting("VERIFIER_URL", "http://verifier:8000")
    try:
        wait_for_health(env_url, timeout=120.0)
        wait_for_health(verifier_url, timeout=120.0)
    except Exception as exc:
        print_line(f"error: services did not become healthy: {exc}")
        run_log.write({"event": "health_error", "error": str(exc)})
        run_log.close()
        return 1

    env = EnvClient(env_url)
    verifier = VerifierClient(verifier_url)
    gate_ids = (
        [task_id] if mode == "full" else [config.task_id for config in configs]
    )
    gate_started = time.monotonic()
    try:
        gate_summary = run_gate(
            env,
            verifier,
            configs=configs,
            task_ids=gate_ids,
            seed_override=seed,
            limits=limits,
            gate_dir=out,
        )
    except Exception as exc:
        print_line(f"error: baseline gate crashed: {type(exc).__name__}: {exc}")
        run_log.write(
            {
                "event": "gate_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        run_log.close()
        return 1
    gate_seconds = round(time.monotonic() - gate_started, 2)
    gate_summary["seed"] = seed
    gate_summary["gate_seconds"] = gate_seconds
    _write_json(out / "gate.json", gate_summary)
    print_line(f"checks took {gate_seconds}s")
    if mode == "full" and gate_seconds > GATE_BUDGET_SECONDS:
        print_line(
            f"WARNING: checks exceeded the {GATE_BUDGET_SECONDS}s budget "
            "for the live run"
        )
    if mode == "baselines":
        run_log.write(
            {
                "event": "run_end",
                "status": "baselines",
                "passed": gate_summary["passed"],
                "gate_seconds": gate_seconds,
            }
        )
        run_log.close()
        print_line(
            "baseline results: "
            + ("all checks passed" if gate_summary["passed"] else "SOME CHECKS FAILED")
            + f"  ->  {out}/gate.json"
        )
        return 0 if gate_summary["passed"] else 2
    if not gate_summary["passed"]:
        _write_json(
            out / "summary.json",
            {
                "run_id": run,
                "status": "gate_failed",
                "mode": mode,
                "tasks": gate_ids,
                "seed": seed,
                "model_called": False,
            },
        )
        run_log.write({"event": "run_end", "status": "gate_failed"})
        run_log.close()
        print_line("A baseline check failed, so the model was not called.")
        return 2

    results: dict[str, dict[str, Any]] = {}
    model_task_ids = gate_ids if mode == "all" else [task_id]
    for tid in model_task_ids:
        config = by_id[tid]
        print_line(
            f"\n== Model run: {model}  task={tid}  seed={seed} =="
        )
        try:
            bundle = verifier.reset(tid, seed)
            oracle_actions = None
            if model == "mock/solver":
                oracle_actions = verifier.oracle(tid, seed)["actions"]
            log_path = (
                out / "episode.jsonl"
                if mode == "full"
                else out / f"episode-{tid}.jsonl"
            )
            with JsonlWriter(log_path) as episode_log:
                result = run_episode(
                    env,
                    verifier,
                    task_id=tid,
                    tier=config.tier,
                    seed=seed,
                    bundle=bundle,
                    client=client,
                    model_name=model,
                    append_text=append_text,
                    limits=limits,
                    log=episode_log,
                    oracle_actions=oracle_actions,
                )
            results[tid] = result
        except Exception as exc:
            print_line(f"task {tid} crashed: {type(exc).__name__}: {exc}")
            results[tid] = {
                "event": "episode_end",
                "task_id": tid,
                "tier": config.tier,
                "seed": seed,
                "model": model,
                "termination": "crash",
                "submitted": False,
                "score": None,
                "steps": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if mode == "full":
        result = results[task_id]
        summary = {
            "run_id": run,
            "status": "completed",
            "task": task_id,
            "seed": seed,
            "model": model,
            "model_called": True,
            "append": append_info,
            "finished_at": utc_now(),
            **result,
        }
        _write_json(out / "summary.json", summary)
        run_log.write({"event": "run_end", "status": "completed", **result})
        run_log.close()
        print_line(
            f"termination: {result['termination']}   score: {result['score']}   "
            f"steps: {result['steps']}   logs: {out}"
        )
        return 1 if result["termination"] == "crash" else 0

    solved = sum(
        result.get("termination") == "success" for result in results.values()
    )
    scores = [
        result.get("score")
        for result in results.values()
        if isinstance(result.get("score"), (int, float))
    ]
    mean_score = round(sum(scores) / len(scores), 3) if scores else None
    summary = {
        "run_id": run,
        "status": "completed",
        "mode": "all",
        "seed": seed,
        "model": model,
        "model_called": True,
        "append": append_info,
        "finished_at": utc_now(),
        "solved": solved,
        "mean_score": mean_score,
        "tasks": results,
    }
    _write_json(out / "summary.json", summary)
    run_log.write({"event": "run_end", "status": "completed", "solved": solved})
    run_log.close()
    print_line("\n== All tasks ==")
    for tid, result in results.items():
        print_line(
            f"{tid:<18} {result.get('termination', '-'):<9} "
            f"score {result.get('score')}  steps {result.get('steps', 0)}"
        )
    print_line(
        f"solved {solved}/{len(results)}   mean score {mean_score}   logs: {out}"
    )
    return (
        1
        if any(
            result.get("termination") == "crash"
            for result in results.values()
        )
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())
