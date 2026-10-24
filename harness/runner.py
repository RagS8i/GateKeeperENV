"""Episode runner, limits, model loop and baseline gate."""

from __future__ import annotations

import copy
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from common.tasks import TaskConfig
from .agents import baseline_actions, oracle_actions
from .env import Env
from .limits import DEFAULT_LIMITS, Limits
from .logging import JsonlWriter, clip, print_line, print_step, utc_now
from .parser import ToolCall
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
- Success means the submitted policy scores at least 0.95 on the hidden check; the final score is reported as the reward.
"""


def _message_size(message: dict[str, Any]) -> int:
    size = len(str(message.get("content") or ""))
    for call in message.get("tool_calls") or []:
        size += len(json.dumps(call, default=str))
    return size


def _reward_from_shadow(shadow: dict[str, Any] | None) -> dict[str, Any]:
    if not shadow:
        return {"total": None}
    reward = dict(shadow)
    reward["total"] = shadow.get("score")
    reward["sampled"] = True
    return reward


def _step_record(
    *,
    agent: str,
    task_id: str,
    step: int,
    seed: int | None,
    model_input: Any,
    model_output_raw: Any,
    parsed_actions: list[dict[str, Any]],
    tool_result: list[str],
    errors: list[dict[str, Any]],
    state: dict[str, Any] | None,
    reward: dict[str, Any] | None,
    tokens: int,
    latency_ms: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One canonical JSONL line per model reply or scripted action."""
    record: dict[str, Any] = {
        "event": "step",
        "timestamp": utc_now(),
        "agent": agent,
        "task": task_id,
        "step": step,
        "seed": seed,
        "model_input": model_input,
        "model_output_raw": model_output_raw,
        "parsed_actions": parsed_actions,
        "tool_result": tool_result,
        "errors": errors,
        "state": (
            {
                "hash": state.get("state_hash", state.get("hash")),
                "statement_count": state.get("statement_count"),
                "submitted": state.get("submitted"),
            }
            if state
            else {"hash": None}
        ),
        "reward": reward if reward is not None else {"total": None},
        "tokens": tokens,
        "latency_ms": latency_ms,
    }
    if extra:
        record.update(extra)
    return record


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
    env: Env,
    *,
    task_id: str,
    tier: str,
    seed: int | None,
    client: Any,
    model_name: str,
    append_text: str,
    limits: Limits,
    log: JsonlWriter,
    oracle_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    suspicion = Suspicion()
    episode_errors: list[dict[str, Any]] = []
    system_prompt = SYSTEM_PROMPT
    if append_text:
        system_prompt = system_prompt + "\n\n" + append_text

    reset_response = env.reset(seed, task_id=task_id)
    bundle = reset_response.get("bundle") or {}
    if oracle_actions is None and model_name == "mock/solver":
        oracle_actions = env.oracle()
    observation = str(reset_response["observation"])
    state = reset_response.get("state") or {}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation},
    ]
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
        # The wall-clock budget starts at the first model request, not at reset.
        started = time.monotonic()
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
            sent = copy.deepcopy(trimmed)
            step_errors: list[dict[str, Any]] = []
            parsed_actions: list[dict[str, Any]] = []
            tool_results: list[str] = []

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
                detail = f"{type(exc).__name__}: {exc}"
                step_errors.append({"kind": "model_call_failed", "detail": detail})
                log.write(
                    _step_record(
                        agent="model",
                        task_id=task_id,
                        step=step,
                        seed=seed,
                        model_input=sent,
                        model_output_raw=None,
                        parsed_actions=parsed_actions,
                        tool_result=tool_results,
                        errors=step_errors,
                        state=state,
                        reward={"total": None},
                        tokens=0,
                        latency_ms=int((time.monotonic() - model_started) * 1000),
                        extra={"termination": termination},
                    )
                )
                print_line(f"[{task_id}] model error: {detail}")
                break

            latency = time.monotonic() - model_started
            usage = reply.usage or {}
            total_prompt_tokens += int(
                usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
            )
            total_completion_tokens += int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )
            reply_tokens = int(usage.get("total_tokens", 0) or 0)
            if reply_tokens <= 0:
                reply_tokens = int(
                    usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                ) + int(
                    usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                )

            max_matching_calls = 50
            calls = reply.calls[:max_matching_calls]
            parsed_actions = [
                {
                    "tool": call.name,
                    "args": call.arguments,
                    "parse_error": call.parse_error,
                }
                for call in calls
            ]
            model_output_raw = reply.raw if reply.raw is not None else reply.text
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

            if not calls:
                suspicion.no_tool_replies += 1
                hint = (
                    "No valid tool call was found. Use the provided tools. "
                    "Return at least one tool call, or call submit when done."
                )
                messages.append({"role": "user", "content": hint})
                step_errors.append({"kind": "no_tool_call", "detail": hint})
                log.write(
                    _step_record(
                        agent="model",
                        task_id=task_id,
                        step=step,
                        seed=seed,
                        model_input=sent,
                        model_output_raw=model_output_raw,
                        parsed_actions=parsed_actions,
                        tool_result=tool_results,
                        errors=step_errors,
                        state=state,
                        reward={"total": None},
                        tokens=reply_tokens,
                        latency_ms=int(latency * 1000),
                    )
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
                    step_errors.append(
                        {"kind": "too_many_tool_calls", "detail": call.name}
                    )
                elif submitted:
                    content = "Not run: the episode was already submitted."
                    step_errors.append(
                        {"kind": "not_run", "detail": "episode already submitted"}
                    )
                elif tool_time_hit:
                    content = "Not run: the wall-clock limit was reached."
                    step_errors.append(
                        {"kind": "not_run", "detail": "wall-clock limit"}
                    )
                elif call.parse_error:
                    content = f"ERROR: {call.parse_error}"
                    step_errors.append(
                        {"kind": "unparsable_arguments", "detail": call.name}
                    )
                elif call.name not in TOOL_NAMES:
                    content = (
                        f"ERROR: unknown tool {call.name!r}. Available tools: "
                        + ", ".join(sorted(TOOL_NAMES))
                    )
                    step_errors.append(
                        {"kind": "unknown_tool", "detail": call.name}
                    )
                else:
                    remaining = limits.max_seconds - (
                        time.monotonic() - started
                    )
                    if remaining <= 0:
                        tool_time_hit = True
                        content = "Not run: the wall-clock limit was reached."
                        step_errors.append(
                            {"kind": "not_run", "detail": "wall-clock limit"}
                        )
                    else:
                        try:
                            result = env.step(
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
                                step_errors.append(
                                    {"kind": "tool_error", "detail": content}
                                )
                            if call.name == "submit" and result.get("ok"):
                                submitted = True
                        except Exception as exc:
                            content = (
                                f"ERROR: tool call failed: {type(exc).__name__}: "
                                f"{exc}"
                            )
                            step_errors.append(
                                {"kind": "tool_error", "detail": content}
                            )
                content = clip(content, limits.max_output_chars)
                tool_results.append(content)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": content,
                    }
                )
            try:
                state = env.state()
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                step_errors.append({"kind": "state_error", "detail": detail})
                log.write(
                    _step_record(
                        agent="model",
                        task_id=task_id,
                        step=step,
                        seed=seed,
                        model_input=sent,
                        model_output_raw=model_output_raw,
                        parsed_actions=parsed_actions,
                        tool_result=tool_results,
                        errors=step_errors,
                        state=None,
                        reward={"total": None},
                        tokens=reply_tokens,
                        latency_ms=int(latency * 1000),
                        extra={"termination": "crash"},
                    )
                )
                termination = "crash"
                break

            policy = _safe_policy(state) or {}
            try:
                last_shadow = env.shadow(policy, timeout=30.0)
            except Exception as exc:
                last_shadow = None
                detail = f"{type(exc).__name__}: {exc}"
                step_errors.append({"kind": "shadow_error", "detail": detail})

            tool_names = [call.name for call in calls[: limits.max_tool_calls_per_reply]]
            elapsed = time.monotonic() - started
            log.write(
                _step_record(
                    agent="model",
                    task_id=task_id,
                    step=step,
                    seed=seed,
                    model_input=sent,
                    model_output_raw=model_output_raw,
                    parsed_actions=parsed_actions,
                    tool_result=tool_results,
                    errors=step_errors,
                    state=state,
                    reward=_reward_from_shadow(last_shadow),
                    tokens=reply_tokens,
                    latency_ms=int(latency * 1000),
                    extra={
                        "tools": tool_names,
                        "shadow": last_shadow,
                        "submitted": submitted,
                        "elapsed": round(elapsed, 3),
                    },
                )
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
        detail = f"{type(exc).__name__}: {exc}"
        episode_errors.append({"kind": "runner_error", "detail": detail})
        print_line(f"[{task_id}] runner error: {detail}")
    finally:
        try:
            client.finish()
        except Exception:
            pass

    try:
        state = env.state()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        episode_errors.append({"kind": "final_state_error", "detail": detail})
        state = None

    score_result: dict[str, Any] | None = None
    policy = _safe_policy(state)
    if policy is not None:
        try:
            score_result = env.score(
                policy,
                timeout=limits.verifier_timeout_seconds,
            )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            episode_errors.append({"kind": "score_error", "detail": detail})

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
    reward = dict(score_result or {})
    reward["total"] = final_score
    result = {
        "event": "episode_end",
        "agent": "model",
        "task": task_id,
        "task_id": task_id,
        "tier": tier,
        "seed": seed,
        "model": model_name,
        "termination": final_termination,
        "submitted": submitted,
        "score": final_score,
        "reward": reward,
        "score_detail": score_result,
        "errors": episode_errors,
        "steps": steps,
        "elapsed": round(elapsed, 3),
        "elapsed_s": round(elapsed, 3),
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


def run_scripted(
    env: Env,
    *,
    task_id: str,
    tier: str,
    seed: int | None,
    actions: list[dict[str, Any]],
    source: str,
    limits: Limits,
    log: JsonlWriter | None,
) -> dict[str, Any]:
    started = time.monotonic()
    env.reset(seed, task_id=task_id)
    submitted = False
    state: dict[str, Any] | None = None
    executed = 0
    episode_errors: list[dict[str, Any]] = []
    for action in actions:
        if submitted:
            break
        tool = str(action["tool"])
        args = action.get("args") or {}
        errors: list[dict[str, Any]] = []
        try:
            result = env.step(tool, args, timeout=limits.tool_timeout_seconds)
            content = str(result.get("content") or "")
            if not result.get("ok"):
                errors.append({"kind": "tool_error", "detail": content})
        except Exception as exc:
            content = f"ERROR: {type(exc).__name__}: {exc}"
            result = {"ok": False}
            errors.append({"kind": "tool_error", "detail": content})
        executed += 1
        if tool == "submit" and result.get("ok"):
            submitted = True
        state = env.state()
        try:
            shadow = env.shadow(_safe_policy(state) or {}, timeout=30.0)
        except Exception as exc:
            shadow = None
            errors.append(
                {
                    "kind": "shadow_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
        episode_errors.extend(errors)
        if log is not None:
            log.write(
                _step_record(
                    agent=source,
                    task_id=task_id,
                    step=executed,
                    seed=seed,
                    model_input=None,
                    model_output_raw=None,
                    parsed_actions=[{"tool": tool, "args": args}],
                    tool_result=[content],
                    errors=errors,
                    state=state,
                    reward=_reward_from_shadow(shadow),
                    tokens=0,
                    latency_ms=0,
                )
            )

    if state is None:
        state = env.state()
    policy = _safe_policy(state) or {}
    score_result = env.score(
        policy,
        timeout=limits.verifier_timeout_seconds,
    )
    elapsed = time.monotonic() - started
    reward = dict(score_result)
    reward["total"] = score_result["score"]
    result = {
        "event": "episode_end",
        "agent": source,
        "task": task_id,
        "task_id": task_id,
        "tier": tier,
        "seed": seed,
        "source": source,
        "termination": "success" if score_result["score"] >= 0.95 else "failure",
        "submitted": submitted,
        "score": score_result["score"],
        "reward": reward,
        "score_detail": score_result,
        "errors": episode_errors,
        "steps": executed,
        "elapsed": round(elapsed, 3),
        "elapsed_s": round(elapsed, 3),
        "state_hash": (state or {}).get("state_hash"),
    }
    if log is not None:
        log.write(result)
    return result


def run_gate(
    env: Env,
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
    gate_started = time.monotonic()
    thresholds = {
        "oracle_min": 0.95,
        "noop_max": 0.05,
        "allow_all_max": 0.001,
        "deny_all_max": 0.001,
    }
    try:
        for config in selected:
            seed = config.seed if seed_override is None else seed_override
            reset = env.reset(seed, task_id=config.task_id)
            bundle = reset["bundle"]
            oracle = oracle_actions(env)
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
                    task_id=config.task_id,
                    tier=config.tier,
                    seed=seed,
                    actions=actions,
                    source=source,
                    limits=limits,
                    log=gate_log,
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
                "task": config.task_id,
                "task_id": config.task_id,
                "difficulty": config.tier,
                "tier": config.tier,
                "seed": seed,
                "oracle": scores["oracle"],
                "noop": scores["noop"],
                "allow_all": scores["allow_all"],
                "deny_all": scores["deny_all"],
                "oracle_ok": scores["oracle"] >= thresholds["oracle_min"],
                "noop_ok": scores["noop"] <= thresholds["noop_max"],
                "scores": scores,
                "passed": passed,
                "elapsed": round(elapsed, 3),
            }
            records.append(record)
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
        "seed": seed_override,
        "gate_seconds": round(time.monotonic() - gate_started, 2),
    }
    (gate_dir / "gate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print_line(
        f"[gate] {'PASS' if summary['passed'] else 'FAIL'} "
        f"({len(records)} tasks)"
    )
    return summary
