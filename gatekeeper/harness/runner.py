"""Episode runner, limits, model loop and baseline gate."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from ..generator import TaskConfig
from .limits import DEFAULT_LIMITS, Limits
from .logging import JsonlWriter, clip, print_line, print_step, utc_now
from .parser import ToolCall
from .service_clients import EnvClient, VerifierClient
from .tools import TOOL_NAMES, TOOL_SPECS

SYSTEM_PROMPT = """You are Gatekeeper, an autonomous access-policy repair agent inside a simulated company.

You are given a broken JSON access policy, a directory of users/groups/resources, and natural-language access requirements. Your job is to edit the policy so it matches the requirements exactly, then call submit.

Evaluation semantics:
- Default deny: a request is allowed only when an allow statement applies.
- An explicit deny always overrides an allow.
- Within a statement, the listed principals, actions and resources are alternatives. Every condition must match the request context.
- Exceptions exempt those principals from that statement.
- Statements are evaluated deterministically by id; the deciding deny is the first matching deny, otherwise the first matching allow.
- The hidden check covers every user, action, resource and supported context variant, including names and context keys not mentioned in the requirements.

Selector syntax:
- Principal: "user:<id>", "group:<id>", or "*".
- Resource: "resource:<id>", "tag:<tag>", or "*".
- Action: "read", "write", "delete", "share", or "*".
- Context: string keys such as "network" and "mfa"; a key may be absent.

Rules:
- Do not execute shell commands; you have no shell. Use only the provided tools.
- Do not guess: inspect groups, users, resources and visible examples with the tools.
- You may call up to 10 tools in one reply and take up to 40 turns.
- Work efficiently. When the policy matches the requirements, call submit.
"""


def _message_size(message: dict[str, Any]) -> int:
    size = len(str(message.get("content") or ""))
    for call in message.get("tool_calls") or []:
        size += len(json.dumps(call, default=str))
    return size


def _conversation_units(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    units: list[list[dict[str, Any]]] = []
    index = 2
    while index < len(messages):
        message = messages[index]
        unit = [message]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            call_ids = {
                call.get("id") for call in message.get("tool_calls") or []
            }
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                unit.append(messages[cursor])
                call_ids.discard(messages[cursor].get("tool_call_id"))
                cursor += 1
            index = cursor
        else:
            index += 1
        units.append(unit)
    return units


def trim_messages(
    messages: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    if len(messages) <= 2:
        return list(messages)
    system = dict(messages[0])
    first_user = dict(messages[1])
    head_size = _message_size(system) + _message_size(first_user)
    if head_size > max_chars:
        available = max(1000, max_chars - _message_size(system) - 800)
        content = str(first_user.get("content") or "")
        first_user["content"] = content[:available] + "\n...[context-truncated]"
        head_size = _message_size(system) + _message_size(first_user)

    units = _conversation_units(messages)
    budget = max_chars - head_size
    kept: list[list[dict[str, Any]]] = []
    for unit in reversed(units):
        unit_size = sum(_message_size(message) for message in unit)
        if unit_size > budget and kept:
            break
        if unit_size > budget:
            break
        kept.append(unit)
        budget -= unit_size
    kept.reverse()
    return [system, first_user] + [
        message for unit in kept for message in unit
    ]


def _conversation_chars(messages: list[dict[str, Any]]) -> int:
    return sum(_message_size(message) for message in messages)


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


@dataclasses.dataclass
class Suspicion:
    tool_calls: int = 0
    unknown_tools: int = 0
    parse_failures: int = 0
    path_like_args: int = 0
    allow_all_edits: int = 0
    repeated_calls: int = 0
    simulate_calls: int = 0
    submit_calls: int = 0
    no_tool_replies: int = 0
    max_context_ratio: float = 0.0
    _seen: set[str] = dataclasses.field(default_factory=set)

    def observe_call(self, call: ToolCall) -> None:
        self.tool_calls += 1
        if call.name == "simulate":
            self.simulate_calls += 1
        if call.name == "submit":
            self.submit_calls += 1
        if call.parse_error:
            self.parse_failures += 1
        if call.name and call.name not in TOOL_NAMES:
            self.unknown_tools += 1
        for text in _walk_strings(call.arguments or {}):
            lowered = text.lower()
            if any(
                marker in lowered
                for marker in (
                    "../",
                    "..\\",
                    "/etc/",
                    "c:\\",
                    "__import__",
                    "subprocess",
                    "os.system",
                    "http://",
                    "https://",
                )
            ):
                self.path_like_args += 1
                break
        if call.name == "put_policy" and isinstance(call.arguments, dict):
            statement = call.arguments.get("statement")
            if isinstance(statement, dict) and statement.get("effect") == "allow":
                if all(
                    "*" in (statement.get(field) or [])
                    for field in ("principals", "actions", "resources")
                ):
                    self.allow_all_edits += 1
        key = call.name + ":" + json.dumps(
            call.arguments or {}, sort_keys=True, default=str
        )
        if key in self._seen:
            self.repeated_calls += 1
        else:
            self._seen.add(key)

    def observe_context(self, ratio: float) -> None:
        self.max_context_ratio = max(self.max_context_ratio, ratio)

    def summary(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "unknown_tools": self.unknown_tools,
            "parse_failures": self.parse_failures,
            "path_like_args": self.path_like_args,
            "allow_all_edits": self.allow_all_edits,
            "repeated_calls": self.repeated_calls,
            "simulate_calls": self.simulate_calls,
            "submit_calls": self.submit_calls,
            "no_tool_replies": self.no_tool_replies,
            "max_context_ratio": round(self.max_context_ratio, 4),
        }


def _client_task_context(
    *,
    task_id: str,
    tier: str,
    bundle: dict[str, Any],
    oracle_actions: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "task_id": task_id,
        "tier": tier,
        "bundle": bundle,
    }
    if oracle_actions is not None:
        context["oracle_actions"] = oracle_actions
    return context


def _safe_policy(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if not state:
        return None
    policy = state.get("policy")
    return policy if isinstance(policy, dict) else None


def run_episode(
    env: EnvClient,
    verifier: VerifierClient,
    *,
    task_id: str,
    tier: str,
    seed: int | None,
    bundle: dict[str, Any],
    client: Any,
    model_name: str,
    append_text: str,
    limits: Limits,
    log: JsonlWriter,
    oracle_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    suspicion = Suspicion()
    system_prompt = SYSTEM_PROMPT
    if append_text:
        system_prompt = system_prompt + "\n\n" + append_text

    reset_response = env.reset(bundle)
    observation = str(reset_response["observation"])
    state = reset_response.get("state") or {}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation},
    ]
    log.write(
        {
            "event": "episode_start",
            "task_id": task_id,
            "tier": tier,
            "seed": seed,
            "model": model_name,
            "system_prompt": system_prompt,
            "initial_observation": observation,
            "initial_state_hash": state.get("state_hash"),
            "append_text": append_text,
        }
    )
    print_line(
        f"[{task_id}] episode start model={model_name} "
        f"state={state.get('state_hash')} seed={seed}"
    )

    submitted = False
    termination: str | None = None
    steps = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    last_shadow: dict[str, Any] | None = None

    try:
        client.start(
            _client_task_context(
                task_id=task_id,
                tier=tier,
                bundle=bundle,
                oracle_actions=oracle_actions,
            )
        )
        for step in range(1, limits.max_steps + 1):
            elapsed = time.monotonic() - started
            remaining = limits.max_seconds - elapsed
            if remaining <= 0:
                termination = "time_cap"
                break
            steps = step
            trimmed = trim_messages(messages, limits.context_char_budget)
            context_chars = _conversation_chars(trimmed)
            suspicion.observe_context(context_chars / limits.context_char_budget)
            model_timeout = max(1.0, min(limits.model_timeout_seconds, remaining))
            model_started = time.monotonic()

            try:
                reply = client.complete(
                    trimmed, list(TOOL_SPECS), timeout=model_timeout
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                if elapsed >= limits.max_seconds:
                    termination = "time_cap"
                else:
                    termination = "crash"
                log.write(
                    {
                        "event": "model_error",
                        "step": step,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed": round(elapsed, 3),
                    }
                )
                print_line(f"[{task_id}] model error: {type(exc).__name__}: {exc}")
                break

            latency = time.monotonic() - model_started
            usage = reply.usage or {}
            total_prompt_tokens += int(
                usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
            )
            total_completion_tokens += int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )

            max_matching_calls = 50
            calls = reply.calls[:max_matching_calls]
            assistant_message = reply.assistant_message or {
                "role": "assistant",
                "content": reply.text,
            }
            if len(reply.calls) > max_matching_calls:
                assistant_message = dict(assistant_message)
                assistant_message["tool_calls"] = (
                    assistant_message.get("tool_calls") or []
                )[:max_matching_calls]
            if len(str(assistant_message.get("content") or "")) > limits.max_assistant_chars:
                assistant_message = dict(assistant_message)
                assistant_message["content"] = str(
                    assistant_message.get("content") or ""
                )[: limits.max_assistant_chars] + "\n...[truncated]"
            messages.append(assistant_message)

            log.write(
                {
                    "event": "model_reply",
                    "step": step,
                    "text": clip(reply.text, 20_000),
                    "calls": [
                        {
                            "id": call.call_id,
                            "name": call.name,
                            "arguments": call.arguments,
                            "raw_arguments": clip(call.raw_arguments, 8_000),
                            "parse_error": call.parse_error,
                        }
                        for call in calls
                    ],
                    "usage": usage,
                    "latency": round(latency, 3),
                    "context_chars": context_chars,
                    "context_ratio": round(
                        context_chars / limits.context_char_budget, 4
                    ),
                }
            )

            if not calls:
                suspicion.no_tool_replies += 1
                hint = (
                    "No valid tool call was found. Use the provided tools. "
                    "Return at least one tool call, or call submit when done."
                )
                messages.append({"role": "user", "content": hint})
                log.write(
                    {"event": "no_tool_call", "step": step, "hint": hint}
                )
                continue

            elapsed = time.monotonic() - started
            remaining = limits.max_seconds - elapsed
            tool_time_hit = remaining <= 0
            for position, call in enumerate(calls):
                suspicion.observe_call(call)
                if position >= limits.max_tool_calls_per_reply:
                    content = (
                        "ERROR: too many tool calls in one reply "
                        f"(limit {limits.max_tool_calls_per_reply}); this call "
                        "was not run."
                    )
                elif submitted:
                    content = "Not run: the episode was already submitted."
                elif tool_time_hit:
                    content = "Not run: the wall-clock limit was reached."
                elif call.parse_error:
                    content = f"ERROR: {call.parse_error}"
                elif call.name not in TOOL_NAMES:
                    content = (
                        f"ERROR: unknown tool {call.name!r}. Available tools: "
                        + ", ".join(sorted(TOOL_NAMES))
                    )
                else:
                    remaining = limits.max_seconds - (
                        time.monotonic() - started
                    )
                    if remaining <= 0:
                        tool_time_hit = True
                        content = "Not run: the wall-clock limit was reached."
                    else:
                        try:
                            result = env.tool(
                                call.name,
                                call.arguments or {},
                                timeout=max(
                                    1.0,
                                    min(limits.tool_timeout_seconds, remaining),
                                ),
                            )
                            if result.get("ok"):
                                content = str(result.get("content") or "")
                            else:
                                content = str(
                                    result.get("content") or "ERROR: tool failed"
                                )
                            if call.name == "submit" and result.get("ok"):
                                submitted = True
                        except Exception as exc:
                            content = (
                                f"ERROR: tool call failed: {type(exc).__name__}: "
                                f"{exc}"
                            )
                content = clip(content, limits.max_output_chars)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": content,
                    }
                )
                log.write(
                    {
                        "event": "tool_result",
                        "step": step,
                        "call_id": call.call_id,
                        "tool": call.name,
                        "arguments": call.arguments,
                        "content": content,
                    }
                )

            try:
                state = env.state()
            except Exception as exc:
                log.write(
                    {
                        "event": "state_error",
                        "step": step,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                termination = "crash"
                break

            policy = _safe_policy(state) or {}
            try:
                last_shadow = verifier.shadow(
                    task_id,
                    policy,
                    seed,
                    timeout=30.0,
                )
            except Exception as exc:
                last_shadow = None
                log.write(
                    {
                        "event": "shadow_error",
                        "step": step,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

            tool_names = [call.name for call in calls[: limits.max_tool_calls_per_reply]]
            elapsed = time.monotonic() - started
            log.write(
                {
                    "event": "step",
                    "step": step,
                    "tools": tool_names,
                    "state_hash": state.get("state_hash"),
                    "statement_count": state.get("statement_count"),
                    "submitted": submitted,
                    "shadow": last_shadow,
                    "elapsed": round(elapsed, 3),
                }
            )
            print_step(
                task_id=task_id,
                step=step,
                tools=tool_names,
                state_hash=state.get("state_hash"),
                shadow_score=(
                    last_shadow.get("score") if last_shadow else None
                ),
                elapsed=elapsed,
            )
            if submitted:
                break
        else:
            termination = "step_cap"

        if termination is None and submitted:
            termination = "submitted"
        if termination is None:
            termination = "step_cap"

    except Exception as exc:
        termination = "crash"
        log.write(
            {
                "event": "runner_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print_line(f"[{task_id}] runner error: {type(exc).__name__}: {exc}")
    finally:
        try:
            client.finish()
        except Exception:
            pass

    try:
        state = env.state()
    except Exception as exc:
        log.write(
            {
                "event": "final_state_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        state = None

    score_result: dict[str, Any] | None = None
    policy = _safe_policy(state)
    if policy is not None:
        try:
            score_result = verifier.score(
                task_id,
                policy,
                seed,
                timeout=limits.verifier_timeout_seconds,
            )
        except Exception as exc:
            log.write(
                {
                    "event": "score_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    final_score = (
        float(score_result["score"]) if score_result is not None else None
    )
    if score_result is None:
        final_termination = "crash"
    elif termination in {"crash", "time_cap", "step_cap"}:
        final_termination = termination
    elif submitted and final_score is not None and final_score >= 0.95:
        final_termination = "success"
    else:
        final_termination = "failure"

    elapsed = time.monotonic() - started
    result = {
        "event": "episode_end",
        "task_id": task_id,
        "tier": tier,
        "seed": seed,
        "model": model_name,
        "termination": final_termination,
        "submitted": submitted,
        "score": final_score,
        "score_detail": score_result,
        "steps": steps,
        "elapsed": round(elapsed, 3),
        "state_hash": (state or {}).get("state_hash"),
        "suspicion": suspicion.summary(),
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "config": dataclasses.asdict(limits),
    }
    log.write(
        {
            **result,
            "final_policy": policy,
        }
    )
    print_line(
        f"[{task_id}] episode end termination={final_termination} "
        f"score={final_score} steps={steps} elapsed={elapsed:.1f}s"
    )
    return result


def baseline_actions(bundle: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    if kind == "noop":
        return [{"tool": "submit", "args": {}}]
    statements = bundle.get("broken_policy", {}).get("statements", [])
    actions: list[dict[str, Any]] = [
        {"tool": "delete_policy", "args": {"id": statement["id"]}}
        for statement in statements
    ]
    effect = "allow" if kind == "allow_all" else "deny"
    actions.append(
        {
            "tool": "put_policy",
            "args": {
                "statement": {
                    "id": "s999",
                    "effect": effect,
                    "principals": ["*"],
                    "actions": ["*"],
                    "resources": ["*"],
                    "condition": None,
                    "exceptions": [],
                }
            },
        }
    )
    actions.append({"tool": "submit", "args": {}})
    return actions


def run_scripted(
    env: EnvClient,
    verifier: VerifierClient,
    *,
    task_id: str,
    tier: str,
    seed: int | None,
    bundle: dict[str, Any],
    actions: list[dict[str, Any]],
    source: str,
    limits: Limits,
    log: JsonlWriter | None,
    log_payloads: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    env.reset(bundle)
    if log is not None:
        log.write(
            {
                "event": "baseline_start",
                "task_id": task_id,
                "tier": tier,
                "seed": seed,
                "source": source,
                "action_count": len(actions),
            }
        )
    submitted = False
    state: dict[str, Any] | None = None
    executed = 0
    for action in actions:
        if submitted:
            break
        tool = str(action["tool"])
        args = action.get("args") or {}
        try:
            result = env.tool(tool, args, timeout=limits.tool_timeout_seconds)
            content = str(result.get("content") or "")
        except Exception as exc:
            content = f"ERROR: {type(exc).__name__}: {exc}"
            result = {"ok": False}
        executed += 1
        if tool == "submit" and result.get("ok"):
            submitted = True
        if log is not None:
            log.write(
                {
                    "event": "baseline_action",
                    "source": source,
                    "index": executed,
                    "tool": tool,
                    "arguments": args if log_payloads else None,
                    "content": content if log_payloads else None,
                }
            )
        state = env.state()

    if state is None:
        state = env.state()
    policy = _safe_policy(state) or {}
    score_result = verifier.score(
        task_id,
        policy,
        seed,
        timeout=limits.verifier_timeout_seconds,
    )
    elapsed = time.monotonic() - started
    result = {
        "event": "episode_end",
        "task_id": task_id,
        "tier": tier,
        "seed": seed,
        "source": source,
        "termination": "success" if score_result["score"] >= 0.95 else "failure",
        "submitted": submitted,
        "score": score_result["score"],
        "score_detail": score_result,
        "steps": executed,
        "elapsed": round(elapsed, 3),
        "state_hash": (state or {}).get("state_hash"),
    }
    if log is not None:
        log.write(result)
    return result


def run_gate(
    env: EnvClient,
    verifier: VerifierClient,
    *,
    configs: tuple[TaskConfig, ...],
    task_ids: list[str],
    seed_override: int | None,
    limits: Limits,
    gate_dir: Path,
) -> dict[str, Any]:
    selected = [config for config in configs if config.task_id in set(task_ids)]
    gate_path = gate_dir / "gate.jsonl"
    if gate_path.exists():
        gate_path.unlink()
    gate_log = JsonlWriter(gate_path)
    records: list[dict[str, Any]] = []
    thresholds = {
        "oracle_min": 0.95,
        "noop_max": 0.05,
        "allow_all_max": 0.001,
        "deny_all_max": 0.001,
    }
    try:
        for config in selected:
            seed = config.seed if seed_override is None else seed_override
            bundle = verifier.reset(config.task_id, seed)
            oracle = verifier.oracle(config.task_id, seed)["actions"]
            sources = {
                "oracle": oracle,
                "noop": baseline_actions(bundle, "noop"),
                "allow_all": baseline_actions(bundle, "allow_all"),
                "deny_all": baseline_actions(bundle, "deny_all"),
            }
            scores: dict[str, float] = {}
            elapsed = 0.0
            for source, actions in sources.items():
                result = run_scripted(
                    env,
                    verifier,
                    task_id=config.task_id,
                    tier=config.tier,
                    seed=seed,
                    bundle=bundle,
                    actions=actions,
                    source=source,
                    limits=limits,
                    log=gate_log,
                    log_payloads=False,
                )
                scores[source] = float(result["score"])
                elapsed += float(result["elapsed"])
            passed = (
                scores["oracle"] >= thresholds["oracle_min"]
                and scores["noop"] <= thresholds["noop_max"]
                and scores["allow_all"] <= thresholds["allow_all_max"]
                and scores["deny_all"] <= thresholds["deny_all_max"]
            )
            record = {
                "task_id": config.task_id,
                "tier": config.tier,
                "seed": seed,
                "scores": scores,
                "passed": passed,
                "elapsed": round(elapsed, 3),
            }
            records.append(record)
            gate_log.write({"event": "gate_task", **record})
            print_line(
                f"[gate] {config.task_id:8s} oracle={scores['oracle']:.3f} "
                f"noop={scores['noop']:.3f} allow_all={scores['allow_all']:.3f} "
                f"deny_all={scores['deny_all']:.3f} "
                f"{'PASS' if passed else 'FAIL'}"
            )
    finally:
        gate_log.close()

    summary = {
        "run_id": gate_dir.name,
        "timestamp": utc_now(),
        "thresholds": thresholds,
        "tasks": records,
        "passed": bool(records) and all(record["passed"] for record in records),
        "task_count": len(records),
    }
    (gate_dir / "gate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print_line(
        f"[gate] {'PASS' if summary['passed'] else 'FAIL'} "
        f"({len(records)} tasks)"
    )
    return summary
