# Gatekeeper

Gatekeeper is a deterministic, model-agnostic reinforcement-learning environment
for **access-policy repair**. An agent receives a broken access policy, a world
model and natural-language requirements. It must use a small set of tools to
find and repair the policy, then submit it. The hidden verifier scores the
submitted policy against the correct policy over every user, action, resource
and supported context variant.

The project is designed for the "Build an RL Environment That Can't Be Broken"
hackathon: one-command startup, hidden-answer isolation, exhaustive scoring,
strong baseline gates, deterministic tasks and a harness that works with any
OpenAI-compatible chat-completions model.

## Quickstart

```bash
# Optional first time: copy and edit configuration.
cp .env.example .env

# Offline demo (no API key required). The default mock/solver is deterministic.
docker compose up --build --abort-on-container-exit --exit-code-from harness
```

That command:

1. builds three small images and starts `sandbox` and `verifier`;
2. waits for both healthchecks;
3. gates the requested task with oracle, no-op, allow-all and deny-all baselines;
4. runs the model (or the offline solver) and writes logs;
5. exits with the harness exit code.

For a real model, set `MODEL`, `API_KEY` and `BASE_URL` in `.env`. For example:

```dotenv
MODEL=<model-name-from-the-email>
API_KEY=<api-key>
BASE_URL=https://provider.example/v1
TASK_ID=med_01
SEED=0
MODE=full
```

No source rebuild is needed for configuration changes. Extra instructions can
be added without rebuilding by putting a UTF-8 file in `inputs/` and setting
`APPEND_FILE=inputs/extra.txt`.

### Modes

| `MODE` | Behaviour |
| --- | --- |
| `full` | Gate `TASK_ID`, then run the model on `TASK_ID`. Default. |
| `all` | Gate every task, then run the model on every task in order. |
| `baselines` | Gate every task only; no model is called. |

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Completed. A model failure or step/time cap is still a completed run. |
| `1` | Crash, unhealthy service, or an episode that could not finish. |
| `2` | A baseline gate failed; the model was not called. |
| `3` | Invalid configuration, task ID, mode or `APPEND_FILE`. |

## Why this environment is hard to game

- **Hidden answer key.** The correct policy, mutation list, decision matrix and
  starting error count live only in the verifier. The sandbox receives a
  sanitized bundle: world, broken policy, requirements, visible examples, task
  metadata and the initial state hash.
- **Exhaustive scoring.** A submission is evaluated on the full Cartesian
  product of users × actions × resources × context variants, not on the handful
  of visible examples.
- **Semantic mutations.** Tasks use six mutation classes: dropped statement,
  widened wildcard, removed condition, removed deny, changed group membership
  and flipped effect. The broken policy is not a random edit.
- **Deny-list baseline.** Each correct policy retains an allow-all baseline and
  a protected restricted-scope deny. This guarantees both allow-all and
  deny-all score approximately zero while still forcing real reasoning.
- **Strong gate.** Oracle must score at least 0.95; no-op at most 0.05;
  allow-all and deny-all at most 0.001. Gate failures stop the model run. The
  reference and no-op baselines are the `oracle_actions` and `noop_actions`
  functions in `harness/agents.py`; `run_gate` executes both through the same
  sandbox and scoring path as the model. Thin CLI wrappers are in
  `scripts/reference_solution.py` and `scripts/noop_agent.py`.
- **Protected answers.** The sandbox has no route to the verifier in Docker; the
  verifier is on a separate internal network. The sandbox has no internet route.
- **Deterministic replay.** World, policy, mutations, requirements, examples and
  all matrix decisions are derived from `(task_id, seed)`. `Env.reset(seed)`
  returns the same starting state and the same state hash for the same seed.

## Architecture

```
                 model API (optional internet)
                          ^
                          |
                    +-----------+
                    |  harness  |  trusted model loop, gate, logs
                    +-----------+
                      |       |
             tools    |       |    hidden score/oracle
                      v       v
                  +-------+  +----------+
                  |sandbox|  | verifier |
                  +-------+  +----------+
                  no internet, no verifier access
```

There are three images and three services:

- `sandbox` — stateful, sanitized tool server. Only `get_requirements`,
  `list_policies`, `lookup`, `simulate`, `put_policy`, `delete_policy`, `undo`,
  `diff`, `check_examples` and `submit` are exposed. Outputs are capped at 4,000
  characters. It has no network route to the verifier or the internet, runs as
  a non-root user, and is read-only.
- `verifier` — trusted generator, hidden matrix, oracle actions, full scoring
  and sampled shadow scoring. It never returns `P*`, mutations or hidden
  decisions to the sandbox. It runs as a non-root user and is read-only.
- `harness` — OpenAI-compatible model loop, baseline gate, budget enforcement,
  suspicion telemetry and JSONL logs. It is the only service with internet
  access.

The harness talks to the two services through `harness/env.py`, which exposes
the public `reset(seed)` / `step(tool, args)` / `score()` interface.

## Task set

`tasks.json` contains twelve tasks:

| Tier | Tasks | Users | Resources | Mutation classes |
| --- | --- | --- | --- | --- |
| easy | `easy_01`-`easy_04` | 10-16 | 30-45 | 1 |
| medium | `med_01`-`med_04` | 30-45 | 60-90 | 2-3 |
| hard | `hard_01`-`hard_04` | 90-105 | 130-160 | 4-5 |

Hard tasks use more statements, nested groups, multiple context keys and
larger matrices. The demo task is `med_01`.

## Scoring

For a submitted policy, the verifier builds a decision byte array over the full
hidden matrix and compares it with the correct decisions:

```
weighted_errors = 3 * false_allows + 1 * false_denies
score           = max(0, 1 - weighted_errors / E_start)
```

`E_start` is the weighted error count of the initial broken policy. A false
allow is weighted three times a false deny because over-permission is the more
dangerous failure for access control. The score is clipped at zero, so a
candidate can never receive credit for making the policy worse.

During the episode, the harness also requests a **sampled shadow score** after
each model turn. It is a cheap progress signal for logs and debugging; only the
final full score decides success.

## Tools

| Tool | Purpose |
| --- | --- |
| `get_requirements` | Read the natural-language rules. |
| `list_policies` | Page through the current statements. |
| `lookup` | Inspect users, groups, resources, tags or actions. |
| `simulate` | Evaluate one request with an explicit context. |
| `put_policy` | Add or replace one statement. |
| `delete_policy` | Delete one statement by ID. |
| `undo` | Revert the most recent edit. |
| `diff` | Compare the current policy with the starting policy. |
| `check_examples` | Check the visible examples only. |
| `submit` | End the episode and trigger full scoring. |

Policy semantics:

- default deny;
- explicit deny overrides allow;
- all conditions on a statement must match;
- exceptions exempt principals from that statement;
- group membership is nested and transitive;
- the deciding statement is deterministic by ascending statement ID.

## Harness budgets

| Limit | Default |
| --- | --- |
| Model steps | 40 |
| Wall clock per episode | 600 s |
| Tool calls per model reply | 10 |
| Tool output | 4,000 chars |
| Tool call timeout | 60 s |
| Verifier score timeout | 120 s |
| Context budget | 144,000 chars (48K tokens at 3 chars/token) |

The harness trims old assistant/tool units to stay inside the context budget,
truncates tool output, rejects unknown tools, and marks unrun calls with an
explicit error. A submission that does not reach 0.95 is reported as `failure`,
not `success`.

## Outputs

Each run writes to `logs/<run-id>/`:

- `run.jsonl` — start/end/error events;
- `gate.json` — pass/fail, thresholds, per-task oracle/no-op/allow-all/deny-all
  scores and gate duration;
- `gate.jsonl` — one canonical step record per baseline action;
- `episode.jsonl` (or `episode-<task>.jsonl`) — one canonical step record per
  model reply, followed by the final `episode_end` record;
- `summary.json` — final status, reward, score detail and step count;
- `append.txt` — exact appended instructions, when used.

A canonical step record contains `timestamp`, `agent`, `task`, `step`, `seed`,
`model_input`, `model_output_raw`, `parsed_actions`, `tool_result`, `errors`,
`state`, `reward`, `tokens` and `latency_ms`. `state` is a hash plus statement
count, not a full policy snapshot. `reward` includes the sampled shadow score
per step and the full verifier result at episode end.

Gate logs include the scripted baseline actions for audit. They are written by
the trusted harness and are never exposed to the sandbox. Episode logs contain
model-visible messages, tool results and candidate state hashes; the hidden
matrix and answer key are never written.

## Local development

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Current suite: 77 tests covering the evaluator, generator, verifier, sandbox,
seeded `Env.reset`, canonical telemetry, HTTP services, parser, append contract,
model loop, limits, fake model, gate and one-command harness flow.

Local services for debugging (run each in its own terminal):

```bash
# terminal 1
PORT=8000 .venv/bin/python -m sandbox.server

# terminal 2
PORT=8001 .venv/bin/python -m verifier.server

# terminal 3
SANDBOX_URL=http://127.0.0.1:8000 \
VERIFIER_URL=http://127.0.0.1:8001 \
MODEL=mock/solver \
.venv/bin/python -m harness.run

# optional: run the two baselines directly against the same services
TASK_ID=med_01 SEED=0 SANDBOX_URL=http://127.0.0.1:8000 \
VERIFIER_URL=http://127.0.0.1:8001 .venv/bin/python scripts/reference_solution.py
TASK_ID=med_01 SEED=0 SANDBOX_URL=http://127.0.0.1:8000 \
VERIFIER_URL=http://127.0.0.1:8001 .venv/bin/python scripts/noop_agent.py
```

`ENV_URL` is still accepted as a legacy alias for `SANDBOX_URL`.

An OpenAI-compatible fake model is included for offline integration tests:
`harness.fake_model_server`. It supports native tool calls, legacy
`function_call`, JSON-in-text calls, malformed arguments, too many tool calls
and no-tool replies.

## Repository layout

```
common/               shared models, evaluator, request, HTTP helpers, task config
sandbox/              agent-facing environment state machine + HTTP server
verifier/             generator, hidden matrix, requirements, scoring + HTTP server
harness/              model loop, gate, Env facade, clients, parser, logging
tasks.json            12 task configurations
check_model.py        one-message model connectivity check
inputs/               optional extra system-prompt text (mounted read-only)
tests/                unit, integration and end-to-end tests
scripts/              baselines, validation and soak utilities
docs/                 design document and deck
baseline-results/     all-task baseline gate evidence
sample-trajectories/  mock and real success/failure trajectories
test-run-logs/        local run evidence
Dockerfile.sandbox    sandbox image
Dockerfile.verifier   verifier image
Dockerfile.harness    harness image
```

## Evidence and artifacts

- `baseline-results/gate.json` — oracle/no-op/allow-all/deny-all gate for all
  twelve tasks at seed 0.
- `sample-trajectories/` — offline solver success/failure traces and real
  `gpt-6-luna` success/failure traces.
- `test-run-logs/` — gate, pytest, generation-soak and three real model-run logs
  from local verification.
- `docs/design-doc.md` and `docs/design-doc.pdf` — full design and threat model.
- `docs/deck.md` and `docs/deck.pdf` — short presentation deck.
- Clean `--no-cache` three-image build completed in 21.8s locally; the Docker
  all-task gate completed in 15.7s.

## Real-model calibration

The one-command contract was exercised with `gpt-6-luna` on 2026-10-25:

| Run | Task | Termination | Score | Steps | Elapsed |
| --- | --- | --- | ---: | ---: | ---: |
| Real success | `med_01` | `success` | 1.0 | 5 | 14.2s |
| Real success | `easy_01` | `success` | 1.0 | 5 | 18.1s |
| Controlled no-op append | `med_01` | `failure` | 0.0 | 1 | 2.1s |

The controlled failure used a temporary `inputs/force_submit.txt` with
`APPEND_FILE` to instruct the model to submit the broken policy immediately. It
verifies the append path and the failure/scoring path; it is not a
model-capability measurement. The exact appended text is preserved in the run
log. Full logs are in `test-run-logs/`.

## Known limitations

- The real-model evidence is a smoke calibration, not a statistically
  meaningful sweep across models, temperatures and seeds.
- Hard tasks are slower to generate because the generator scores candidate
  mutations against the sampled matrix. Standalone hard-task generation took
  roughly 8-20 seconds locally; the all-task gate took about 15.7 seconds.
- The offline `mock/solver` consumes oracle actions from the harness. It is a
  deterministic gate/demo aid, not a model baseline.
- `GATE_BUDGET_SECONDS = 120` currently emits a warning only. `MODE=all` uses
  the per-episode `MAX_SECONDS` budget rather than one global 10-minute cap.

## Security boundaries

- The model only sees messages built by the harness and tool output from the
  sandbox. It never sees the verifier's answer key.
- The verifier is not on the sandbox network.
- The sandbox has no internet route and drops all Linux capabilities.
- `APPEND_FILE` is confined to `inputs/`, must be UTF-8, and is copied into the
  run log for auditability.
- `API_KEY` is read only by the harness; it is not written to summaries or logs.

## Reproducing the gate

```bash
# All twelve tasks, no model:
MODE=baselines SEED=0 docker compose up --build --abort-on-container-exit --exit-code-from harness
```

Or locally:

```bash
MODE=baselines SEED=0 .venv/bin/python -m harness.run
```
