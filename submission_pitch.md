# Gatekeeper: A Hidden-Answer RL Benchmark for IAM Access-Policy Repair

**Technical hackathon submission**

Gatekeeper is a deterministic, model-agnostic reinforcement-learning environment for access-policy repair. The agent receives a broken IAM-style policy, a synthetic enterprise directory, natural-language requirements, and a constrained tool API. It must repair the policy and submit it. A hidden verifier evaluates the submitted policy over the complete Cartesian product of users, actions, resources, and context variants.

The benchmark is intentionally about *behavioral equivalence*, not text similarity. A plausible-looking policy earns nothing unless its induced decisions match the hidden reference policy across the entire evaluation matrix.

---

## Executive Summary

### The enterprise problem

Access-policy repair is a recurring IAM and security-engineering task. Real policies accumulate wildcards, stale exceptions, condition drift, nested group memberships, and grants that were correct for an earlier organization but are over-permissive today. The difficult part is not writing a policy that looks reasonable. The difficult part is proving that a changed symbolic rule preserves the intended decision function across every user, action, resource, and context combination.

That is a constraint-satisfaction problem. A single widened wildcard can affect thousands of requests; a removed `mfa` condition can silently change behavior in one context slice; a group edit can propagate through transitive membership. Manual review and ordinary retrieval-augmented LLM workflows are poorly matched to this problem because they reason over text and examples, not over the induced decision surface.

Gatekeeper turns that enterprise task into a reproducible RL benchmark:

- **Observation:** a sanitized task bundle containing the world model, the broken policy, requirements, visible examples, and a state hash.
- **Action space:** a fixed set of policy-inspection and policy-edit tools, plus `submit`.
- **State:** the current access policy and world.
- **Reward:** a hidden, normalized score computed over the full decision matrix.
- **Episode:** at most 40 model turns, 10 executed tool calls per turn, and 600 seconds.

The benchmark is model-agnostic. Any OpenAI-compatible chat-completions model can drive the harness. The same image also ships with deterministic mock clients and a fake OpenAI-compatible server for offline validation.

### What the submission demonstrates

- A one-command Docker Compose contract with separate environment, verifier, and harness services.
- Twelve deterministic tasks across easy, medium, and hard tiers.
- A hidden verifier that keeps the correct policy, mutation records, reference decisions, and full matrix out of the environment.
- A strict tool surface with schema validation, bounded outputs, bounded turns, and no shell access.
- A pre-flight baseline gate that rejects trivial or degenerate environments before the model is called.
- A complete offline execution trace in which a control agent repairs `med_01` in three turns for a score of `1.0`, plus a negative control that submits the broken policy for `0.0`.

The shipped suite evaluates **1,906,400 request/context decisions** across twelve tasks. The demo task, `med_01`, alone contains **43,200 hidden decisions**. The largest hard task contains **470,400**. This is a 15,000-node-scale benchmark surface and substantially larger in the upper tiers, not a lookup table of a few examples.

---

## Why It's a Hard AI Benchmark

### 1. The target is a policy function, not a document

Let:

- `W` be the world: users, nested groups, resources, tags, and actions.
- `P*` be the hidden correct policy.
- `P0` be the broken policy produced by semantic mutations of `P*`.
- `V = U x A x R x C` be the hidden evaluation matrix.
- `f(P)` be the decision vector produced by evaluating policy `P` on every request in `V`.

The agent must produce a policy `P_hat` such that `f(P_hat)` matches `f(P*)`, or minimizes the weighted disagreement if it cannot match exactly. The requirements text is a lossless natural-language rendering of the correct rules and group membership, but the text is not the objective, and the agent cannot access the hidden `P*` object or matrix. The objective is the induced function over `V`. Two syntactically different policies can be equivalent; one locally plausible edit can be globally wrong.

This distinction is why Gatekeeper is not simple data retrieval. The agent can read the broken policy and the requirements, but it cannot retrieve a row that says "the hidden decision at user `u017`, action `delete`, resource `r036`, context `{mfa: false}` is deny." That label exists only inside the verifier's matrix.

### 2. Policy semantics create non-local interactions

`gatekeeper/evaluator.py` implements the semantics deliberately:

- **Default deny.** A request is allowed only if an allow statement applies.
- **Deny overrides allow.** The evaluator scans every matching deny first, then every matching allow.
- **Conditions are conjunctive.** Every key/value pair in a statement condition must match the request context.
- **Exceptions are principal-level exemptions.** An exception removes a principal from that statement's effect.
- **Groups are transitive.** `_compute_user_groups` walks nested group membership to a fixed point, so `user -> group -> parent group` is one effective membership set.
- **Wildcards and tags are set-valued.** A wildcard principal, action, or resource can cover thousands of requests; a tag selector applies to every resource carrying that tag.
- **Statement IDs are deterministic.** Decisions are independent of insertion order and stable across equivalent policies.

These rules turn local edits into global effects. Removing one condition changes an entire context slice. Flipping one effect changes every request that reaches that statement. Widening one resource selector can silently grant access to resources the agent never inspected. The model must reason about the interaction of all rules, not patch text in isolation.

Nested groups amplify this. In the hard tier, `hard_02` has ten of its sixteen groups containing nested members. A user's effective groups are the transitive closure of direct memberships and parent groups. An agent that inspects only direct membership can produce a policy that looks correct in the directory view but misclassifies many users in the matrix.

### 3. The hidden matrix is large and only sparsely observable

The matrix is the Cartesian product of users, four actions, resources, and tier-specific context variants:

- Easy: 5 context variants.
- Medium: 6 context variants.
- Hard: 7 context variants, including `network` values `corp`, `vpn`, and `guest`, and `mfa` values `true` and `false`.

Exact shipped dimensions:

| Tier | Tasks | Users | Resources | Contexts | Mutation classes | Hidden decision nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Easy | 4 | 10-16 | 30-45 | 5 | 1 | 6,000-14,400 |
| Medium | 4 | 30-45 | 60-90 | 6 | 2-3 | 43,200-97,200 |
| Hard | 4 | 90-105 | 130-160 | 7 | 4-5 | 327,600-470,400 |

The demo `med_01` has 30 users, 7 groups, 60 resources, 4 actions, and 6 contexts, for **43,200 decisions**. `hard_04` has 105 users, 18 groups, 160 resources, 4 actions, and 7 contexts, for **470,400 decisions**. Across all twelve tasks the verifier can compare **1,906,400** decisions.

The agent does not get the matrix. It gets 5-9 visible examples, a `simulate` tool that evaluates one request at a time against the current candidate policy, and a `check_examples` tool that only checks the visible examples. Even exhausting the entire 40-turn, 10-call budget yields at most 400 simulations, which is less than 1% of the `med_01` matrix. Six visible examples cover roughly 0.014% of `med_01`; nine cover roughly 0.002% of `hard_04`.

That sparse observation channel is the point. The task is underdetermined at the example level but fully determined at the semantic level. The agent must infer the correct symbolic policy and rely on the rules, not enumerate the matrix.

### 4. This is constraint satisfaction, not prompt-shaped lookup

The policy grammar is discrete and combinatorial:

- 1-200 statements per policy.
- Each statement has an effect, 1-64 principal selectors, 1-8 action selectors, 1-64 resource selectors, optional conjunctive conditions, and optional exceptions.
- Principal selectors can be users, groups, or wildcard.
- Resource selectors can be resource IDs, tags, or wildcard.
- The target is the equality of two decision vectors over tens to hundreds of thousands of nodes.

A useful formulation is:

```text
Find P_hat in the policy grammar
minimizing   3 * false_allows(P_hat, P*) + false_denies(P_hat, P*)
over         every (user, action, resource, context) in V
```

The agent receives partial labels, not a gradient, and cannot inspect `P*`. A policy that matches every visible example can still fail the hidden matrix; the test suite includes an overfit policy built solely from visible examples and verifies that it scores below `0.95` (`tests/test_generator.py::test_visible_example_overfit_policy_fails`).

This is why the benchmark stresses model reasoning rather than pattern matching. The model must:

1. Parse multi-rule natural-language specifications.
2. Inspect group, resource, and tag structure.
3. Decide which broken statements are semantically wrong.
4. Construct a compact policy whose behavior matches the intended decisions.
5. Avoid over-permission and under-permission across unmentioned names and context slices.

The six mutation classes are designed to require that reasoning: `drop_statement`, `widen_wildcard`, `remove_condition`, `remove_deny`, `change_group`, and `flip_effect`. Each mutation is applied only where it has measurable impact on a sampled portion of the hidden matrix.

---

## Architecture & Defenses

### 1. Container isolation and trust boundaries

`docker-compose.yml` defines one image and three services:

```text
                         model API (optional internet)
                                  ^
                                  |
                            +-----------+
                            |  harness  |  trusted loop, gate, scoring
                            +-----------+
                              |       |
                     tools    |       |    hidden score/oracle
                              v       v
                          +------+  +----------+
                          | env  |  | verifier |
                          +------+  +----------+
                         no internet, no verifier route
```

- `env` is attached only to the internal `tools` network. It receives the sanitized bundle and exposes the agent-facing tools.
- `verifier` is attached only to the internal `trusted` network. It holds the generated task, correct policy, mutation records, reference decisions, matrix, and oracle.
- `harness` is attached to `tools`, `trusted`, and `internet`. It is the only service that can reach the model API, and it is the only service that calls the verifier's score/oracle endpoints.

The `env` and `verifier` containers are `read_only`, with `cap_drop: ALL`, `no-new-privileges`, bounded PIDs and memory, and writable `/tmp` via tmpfs. Healthchecks gate startup; `depends_on` ensures the harness waits for both services before the baseline gate.

The model never gets a shell, a filesystem mount, or a direct route to the verifier. It gets structured tool calls. The harness is trusted; the design protects against model actions and accidental answer leakage, not against a malicious operator with host access.

### 2. The hidden answer key

`gatekeeper/verifier_core.py` implements `TaskService`:

- `get_task(task_id, seed)` generates or retrieves a cached `GeneratedTask` keyed by `(task_id, seed)`.
- `reset(...)` returns only `GeneratedTask.sanitized_bundle()`.
- `score(...)` and `shadow(...)` accept a candidate policy and return aggregate score details.
- `oracle(...)` returns the action list used by the baseline gate and the offline `mock/solver`.

The sanitized bundle contains exactly:

```text
task_id, tier, seed, world, broken_policy, requirements,
visible_examples, state_hash
```

It does not contain `correct_policy`, `reference_decisions`, `reference_sample`, `mutations`, `matrix`, `e_start`, `sample_indices`, `shadow_indices`, or `oracle_actions`. A recursive test asserts both the exact key set and the absence of those forbidden strings (`tests/test_generator.py::test_sanitized_bundle_contains_no_answers`).

The exposed environment API therefore provides no route to the hidden matrix. The verifier is the only service with the answer key, and it is not on the environment network. The harness calls `/oracle` only for the pre-flight gate and the offline control solver; it is not exposed as a model tool.

### 3. Strict tool parsing and bounded execution

The model interface is intentionally narrow. `gatekeeper/harness/parser.py` accepts the common wire formats used by OpenAI-compatible providers:

- native `tool_calls`;
- legacy `function_call`;
- JSON objects or arrays embedded in text, including fenced text;
- `tool`/`args` and `name`/`arguments` shapes.

That normalization improves model compatibility, but execution remains strict:

- `parse_arguments` requires a JSON object and reports malformed arguments as a recoverable parse error.
- `run_episode` caps executed calls at `max_tool_calls_per_reply = 10`; extra calls receive an explicit error and are not run.
- Unknown tools, malformed calls, and tool exceptions become tool-result errors, not host execution.
- Tool output is clipped by `Environment` to 4,000 characters; the harness also clips model-visible content.
- The context budget is 144,000 characters; old conversation units are trimmed before each model call.
- `submit` freezes the environment; later edits return an error.
- The model sees only the ten declared tools; there is no shell, HTTP, filesystem, or code-execution tool.

`gatekeeper/env_core.py` validates edits before they enter the policy:

- Pydantic models use `extra="forbid"` and explicit selector validators.
- `put_policy` checks that referenced users, groups, resources, tags, and actions exist in the world.
- Statement payloads are size-bounded; policies are capped at 200 statements.
- The environment checks the bundle's `state_hash` at reset, and exposes state hashes after each edit.
- `simulate` evaluates the *current candidate* policy; it does not reveal hidden reference decisions.
- `check_examples` checks only the visible examples.

The harness also records suspicion telemetry: unknown tools, parse failures, path-like arguments, repeated calls, allow-all edits, submit attempts, no-tool replies, and context pressure. This telemetry is diagnostic; it is not the reward. The reward comes from the verifier.

### 4. Deterministic seeding and reproducibility

`tasks.json` defines twelve tasks with fixed dimensions and seeds. `gatekeeper/generator.py` uses `random.Random(seed)` for world construction, group membership, resource tags, statement generation, mutation selection, visible examples, and sample selection. The harness passes `SEED` as the override, so the same `(task_id, seed)` produces the same task and matrix on every run.

Determinism is reinforced by:

- Sorted user, group, resource, action, and statement iteration.
- Canonical statement and world serialization.
- An 8-character SHA-256 `state_hash` over the canonical world and policy.
- A verifier cache keyed by `(task_id, seed)`.
- Fixed sample seeds for mutation-impact and shadow samples.
- A microsecond-resolution `run_id` for log directories.

The agent-visible state hash is a provenance aid, not a secret. The hidden matrix and answer key remain in the verifier.

### 5. Mathematical reward integrity

For a candidate policy `P_hat`:

```text
false_allows = number of requests where P_hat allows and P* denies
false_denies = number of requests where P_hat denies and P* allows
weighted_errors = 3 * false_allows + false_denies
score = max(0, 1 - weighted_errors / E_start)
```

`E_start` is the weighted error count of the broken policy:

```text
E_start = 3 * broken_false_allows + broken_false_denies
```

The score is implemented in `HiddenMatrix.score_from_errors` and used by `GeneratedTask.score`. When `E_start > 0`, a score of `1.0` requires exactly zero false allows and zero false denies over the full matrix. There is no partial credit for plausible-looking text and no reward for self-reported completion.

Several properties close common reward-hacking channels:

1. **The reference is hidden.** The candidate cannot read `P*`, the matrix, or the reference decisions. Tools expose only candidate-policy evaluation and a few visible labels.
2. **The score is behavior-based.** A policy is not judged by its JSON shape, statement count, or similarity to the broken policy; it is judged by the decision vector it induces.
3. **False allows are weighted three times false denies.** Over-permission is treated as the more dangerous IAM failure, so a policy cannot trade large over-grants for small under-grants without paying for them.
4. **The score is clipped at zero.** Making the policy worse than the broken state never increases reward.
5. **The generator rejects trivial tasks.** It requires both allow and deny decisions and ensures the broken policy has enough weighted error. It also checks that allow-all and deny-all would score at or below zero, and the gate verifies this empirically.
6. **The gate runs before the model.** `run_gate` checks oracle `>= 0.95`, no-op `<= 0.05`, allow-all `<= 0.001`, and deny-all `<= 0.001` on every selected task. A failure exits `2` and the model is not called.
7. **Shadow scoring is not success.** The harness requests a 4,096-node sample after each turn for progress telemetry, normalized by a shadow-specific broken-policy error count. Success is decided only by the full-matrix score after submission.
8. **The breakable surface is not in the model's action space.** The model cannot write logs, alter the verifier, change `E_start`, or submit a score. It can only mutate the policy through the environment.

The generator also reserves a protected allow-all baseline and a protected restricted-scope deny. The correct policy therefore keeps a realistic deny-list shape while still forcing the agent to reconstruct the semantic exceptions. The protected rules guarantee that allow-all and deny-all are poor strategies by construction, not by accidental test data.

---

## Execution Flow

### 1. Startup and pre-flight gate

`gatekeeper/harness/run.py` is the entrypoint. On startup it:

1. Loads `tasks.json`.
2. Parses `MODEL`, `TASK_ID`, `SEED`, `MODE`, `MAX_STEPS`, `MAX_SECONDS`, and `APPEND_FILE`.
3. Validates numeric configuration and exits `3` on bad values.
4. Creates `logs/<run-id>/` with `run.jsonl`.
5. Waits for `/health` on both `ENV_URL` and `VERIFIER_URL`.
6. Runs the baseline gate before any model call.

The gate in `runner.py` resets each selected task in the verifier, obtains the oracle actions, constructs no-op, allow-all, and deny-all actions, executes all four through the same env/tool path, and scores them. For `med_01` in the saved success run:

| Baseline | Score | Requirement |
| --- | ---: | --- |
| Oracle | 1.000 | `>= 0.95` |
| No-op | 0.000 | `<= 0.05` |
| Allow-all | 0.000 | `<= 0.001` |
| Deny-all | 0.000 | `<= 0.001` |

The task passed in 1.707 seconds. The full twelve-task gate passed in about 41.7 seconds. If the gate fails, the harness writes a `gate_failed` summary and exits `2` without calling the model.

### 2. Episode loop

After the gate, the harness performs the following loop (`run_episode`):

```text
bundle = verifier.reset(task_id, seed)
reset  = env.reset(bundle)
messages = [system_prompt, reset.observation]

for step in 1..MAX_STEPS:
    trimmed = trim_messages(messages, 144_000)
    reply = model.complete(trimmed, tool_specs)
    calls = parse_tool_calls(reply)
    for call in calls[:10]:
        result = env.tool(call.name, call.arguments)
        messages += tool_result
    state = env.state()
    shadow = verifier.shadow(task_id, state.policy, seed)
    if submitted: break

final = verifier.score(task_id, state.policy, seed)
termination = success if submitted and final.score >= 0.95 else failure
```

The loop enforces wall-clock limits, step limits, tool timeouts, verifier timeouts, output clipping, and context trimming. A model crash is a harness failure; a model that finishes with a low score is a completed evaluation with termination `failure`.

### 3. Real trace: `med_01`, seed 0, `mock/solver`

Source: `sample-trajectories/success-med_01-mock-solver/episode.jsonl`.

Run metadata:

- `run_id`: `20261023T223327242365Z`
- Task: `med_01` (medium)
- Seed: `0`
- Model: `mock/solver`
- Initial state hash: `073c02ef`
- Initial observation length: 5,523 characters
- Final termination: `success`
- Final score: `1.0`
- Steps: `3`
- Episode elapsed: `0.529s`

The initial observation contained the requirements, all seven groups, the broken ten-statement policy, and the tool list. The hidden task had been constructed by two semantic mutations:

| Mutation | Statement | Before | After | Full-matrix effect |
| --- | --- | --- | --- | --- |
| `remove_condition` | `s003` | deny group `g03` delete/share `r036`/`r056` only when `mfa=false` | same rule with no condition | 72 false denies |
| `flip_effect` | `s004` | deny group `g02` delete/read `r020`/`r022` | allow the same request set | 191 false allows |

These two local changes produce `E_start = 3 * 191 + 72 = 645`, which exactly matches the final score detail. This is the central difficulty: a two-rule edit expands to 645 weighted errors across 43,200 hidden decisions.

The observed agent edits were:

| Step | Tool calls | State change | Statement count | State hash | Shadow |
| ---: | --- | --- | ---: | --- | ---: |
| 1 | 8 x `delete_policy` on `s001`-`s008` | Removes most of the broken policy | 9 -> 2 | `98ba0c1a` | 0.000 |
| 2 | delete `s009`/`s010`; add `s001`-`s006` | Empties the broken policy, writes the first six correct statements | 6 | `1d4185a7` | 0.000 |
| 3 | add `s007`-`s010`; `submit` | Reaches the ten-statement target and freezes the state | 10 | `62d18b9d` | 1.000 |

The tool results returned edit receipts such as:

```json
{"deleted": true, "statement_id": "s001", "statement_count": 9}
```

and:

```json
{"updated": true, "action": "added", "statement_id": "s010", "statement_count": 10}
```

The initial state is the broken policy. Step 1 is the agent's first query into that state: it removes eight statements and observes the shrinking statement count. Step 2 deletes the remaining broken statements and begins writing the reference policy. Step 3 completes the policy, calls `submit`, and the environment freezes.

The final verifier response was:

```json
{
  "task_id": "med_01",
  "score": 1.0,
  "false_allows": 0,
  "false_denies": 0,
  "weighted_errors": 0,
  "e_start": 645,
  "total_requests": 43200
}
```

The episode used 21 tool calls: 0 unknown tools, 0 parse failures, 0 path-like arguments, 0 repeated calls, 1 `submit`, and 0 no-tool replies. The one `allow_all_edits` counter is the protected allow-all statement `s001`; it is telemetry, not a penalty.

The `mock/solver` is deliberately oracle-backed: `MockClient.start` receives `oracle_actions` from the verifier when the harness selects `mock/solver`. This trace is therefore a validation of the environment, parser, tool state machine, logging, and scoring path, not evidence of language-model reasoning. A real model receives no oracle action list and must use `get_requirements`, `lookup`, `list_policies`, `simulate`, `diff`, and `check_examples` to construct its edits.

### 4. Negative control: submit the broken policy

Source: `sample-trajectories/failure-med_01-fake-native-submit/episode.jsonl`.

A fake OpenAI-compatible model issued a native `submit` call immediately:

- Initial and final state hash: `073c02ef` (unchanged).
- Steps: 1.
- Shadow: 0.000.
- Final score: 0.0.
- False allows: 191.
- False denies: 72.
- Weighted errors: 645.
- `E_start`: 645.
- Total requests: 43,200.

The failure is exact: the broken policy's weighted errors equal `E_start`, so the normalized score is zero. Calling `submit` does not create reward; the verifier scores the policy that exists at submission time.

### 5. What the trace proves

- The environment exposes a concrete broken state and returns deterministic edit receipts.
- The parser maps a tool reply into executed policy mutations.
- `submit` freezes the state and triggers the full-matrix verifier.
- The score is derived from hidden decisions, not from model text or tool usage.
- A perfect policy scores `1.0`; an unchanged broken policy scores `0.0`.
- The same execution loop supports native tool calls, legacy calls, and text-embedded calls through the parser.

---

## Validation Snapshot

| Check | Result |
| --- | --- |
| Unit/integration suite | `72 passed` in 32.75s |
| Twelve-task baseline gate | `12/12` passed, about 41.7s |
| Demo success trace | `med_01`, 3 steps, `1.0`, 0 false allows/denies |
| Demo failure trace | `med_01`, 1 step, `0.0`, 191 false allows, 72 false denies |
| Generation soak | `easy_01` 1,000 seeds, 0 failures, 0 determinism failures |
| Additional soaks | `med_01` 20 seeds; `hard_03` 3 seeds; zero failures |
| Shadow sanity | correct policy `1.0`, broken policy `0.0` on fixed sample |

Artifacts:

- `baseline-results/gate.json` - all-task gate evidence.
- `sample-trajectories/` - success and failure traces.
- `test-run-logs/` - gate, pytest, validate-all, and soak logs.

Current limitations are explicit:

- Docker Compose was configuration-reviewed but not executed on the authoring machine because Docker is unavailable there; the judging environment must run the one-command contract.
- No hackathon API key or GPT-Luna endpoint was available, so real-model calibration is pending. The harness and fake-model path are fully exercised; the included `mock/solver` is an oracle-backed control, not a model baseline.

---

## Source Map

| File | Responsibility |
| --- | --- |
| `gatekeeper/verifier_core.py` | Hidden task service, cache, reset/score/shadow/oracle |
| `gatekeeper/generator.py` | Deterministic worlds, P*, mutations, hidden matrix, sanitized bundle |
| `gatekeeper/evaluator.py` | Deny-first semantics, nested-group closure, wildcard/tag matching |
| `gatekeeper/matrix.py` | Full Cartesian matrix, sampling, error counts, score formula |
| `gatekeeper/requirements.py` | Lossless natural-language rendering of P* |
| `gatekeeper/env_core.py` | Sanitized tool environment, validation, history, `submit` freeze |
| `gatekeeper/services/env_server.py` | Environment HTTP service |
| `gatekeeper/services/verifier_server.py` | Trusted verifier HTTP service |
| `gatekeeper/harness/runner.py` | Episode loop, gate, shadow scoring, budget enforcement |
| `gatekeeper/harness/run.py` | Configuration, modes, health checks, logging, exit codes |
| `gatekeeper/harness/parser.py` | Native, legacy, and text-embedded tool-call parsing |
| `gatekeeper/harness/clients.py` | OpenAI-compatible client and deterministic mock clients |
| `docker-compose.yml` | Three-service isolation, healthchecks, networks, resource limits |
| `tasks.json` | Twelve task configurations |
| `sample-trajectories/*/episode.jsonl` | Real execution transcripts used above |
