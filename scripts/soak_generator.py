"""Generate many seeds and check the invariant that matters for the gate."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatekeeper.generator import generate_task, load_task_configs
from gatekeeper.models import Policy, Statement


def trivial(effect: str) -> Policy:
    return Policy(
        statements=[
            Statement(
                id=f"{effect}-all",
                effect=effect,
                principals=["*"],
                actions=["*"],
                resources=["*"],
            )
        ]
    )


def check(task) -> None:
    assert task.score(task.correct_policy)["score"] >= 0.95
    assert task.score(task.broken_policy)["score"] <= 0.05
    assert task.score(Policy(statements=[]))["score"] <= 0.05
    assert task.score(trivial("allow"))["score"] <= 0.05
    assert task.score(trivial("deny"))["score"] <= 0.05
    assert task.e_start >= 20
    assert len(task.mutations) == len(task.config.mutation_types)
    assert all(record.sample_impact > 0 for record in task.mutations)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy_01")
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--determinism-every", type=int, default=100)
    args = parser.parse_args()

    config = next(
        (item for item in load_task_configs() if item.task_id == args.task),
        None,
    )
    if config is None:
        print(f"unknown task {args.task!r}", file=sys.stderr)
        return 3

    times = []
    failures = []
    duplicates = 0
    for offset in range(args.seeds):
        seed = args.start + offset
        started = time.monotonic()
        try:
            task = generate_task(config, seed_override=seed)
            check(task)
            if args.determinism_every and offset % args.determinism_every == 0:
                again = generate_task(config, seed_override=seed)
                if again.state_hash != task.state_hash:
                    duplicates += 1
                    failures.append((seed, "non-deterministic state hash"))
        except Exception as exc:
            failures.append((seed, f"{type(exc).__name__}: {exc}"))
        times.append(time.monotonic() - started)
        if (offset + 1) % 50 == 0:
            print(
                f"{args.task}: {offset + 1}/{args.seeds} seeds, "
                f"{len(failures)} failures",
                flush=True,
            )

    print(
        f"\n{args.task}: {args.seeds} seeds, {len(failures)} failures, "
        f"{duplicates} determinism failures"
    )
    if times:
        ordered = sorted(times)
        print(
            f"generation seconds: mean={statistics.mean(times):.3f} "
            f"p50={statistics.median(times):.3f} "
            f"p95={ordered[int(len(ordered) * 0.95) - 1]:.3f} "
            f"max={max(times):.3f}"
        )
    for seed, reason in failures[:20]:
        print(f"FAIL seed={seed}: {reason}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
