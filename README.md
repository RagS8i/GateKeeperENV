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

1. builds one image and starts `env` and `verifier`;
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
  starting error count live only in the verifier. The env receives a sanitized
  bundle: world, broken policy, requirements, visible examples, task metadata
  and the initial state hash.
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
  allow-all and deny-all at most 0.001. Gate failures stop the model run.
- **Protected answers.** The env has no route to the verifier in Docker; the
  verifier is on a separate internal network. Env has no internet route.
- **Deterministic replay.** World, policy, mutations, requirements, examples and
  all matrix decisions are derived from `(task_id, seed)`. The same seed gives
  the same task and the same score.

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
                  +------+  +----------+
                  | env  |  | verifier |
                  +------+  +----------+
                  no internet, no verifier access
```

There is one image and three services:

- `env` — stateful, sanitized tool server. Only `get_requirements`,
  `list_policies`, `lookup`, `simulate`, `put_policy`, `delete_policy`, `undo`,
  `diff`, `check_examples` and `submit` are exposed. Outputs are capped at 4,000
  characters. The env has no network route to the verifier or the internet.
- `verifier` — trusted generator, hidden matrix, oracle actions, full scoring
  and sampled shadow scoring. It never returns `P*`, mutations or hidden
  decisions to the env.
- `harness` — OpenAI-compatible model loop, baseline gate, budget enforcement,
  suspicion telemetry and JSONL logs. It is the only service with internet
  access.

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
- `gate.json` and `gate.jsonl` — baseline checks;
- `episode.jsonl` (or `episode-<task>.jsonl`) — full trajectory;
- `summary.json` — final status, score and step count;
- `append.txt` — exact appended instructions, when used.

Gate logs deliberately omit hidden payloads. Episode logs contain agent-visible
messages only.

## Local development

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Current suite: 72 tests covering the evaluator, generator, verifier, env, HTTP
services, parser, append contract, model loop, limits, fake model, gate and
one-command harness flow.

Local services for debugging (run each in its own terminal):

```powershell
# terminal 1
& ".\.venv\Scripts\python.exe" -m gatekeeper.services.env_server

# terminal 2
$env:PORT="8001"
& ".\.venv\Scripts\python.exe" -m gatekeeper.services.verifier_server

# terminal 3
$env:ENV_URL="http://127.0.0.1:8000"
$env:VERIFIER_URL="http://127.0.0.1:8001"
$env:MODEL="mock/solver"
& ".\.venv\Scripts\python.exe" -m gatekeeper.harness.run
```

An OpenAI-compatible fake model is included for offline integration tests:
`gatekeeper.harness.fake_model_server`. It supports native tool calls, legacy
`function_call`, JSON-in-text calls, malformed arguments, too many tool calls
and no-tool replies.

## Repository layout

```
gatekeeper/
  evaluator.py       deterministic policy evaluator
  models.py          validated Pydantic models + canonical state hash
  generator.py       task/world/policy/mutation generator
  matrix.py          hidden decision matrix and scoring primitives
  requirements.py    lossless natural-language renderer
  verifier_core.py   trusted task service and cache
  env_core.py        sanitized tool environment
  http_util.py       stdlib JSON HTTP helpers
  services/          env and verifier HTTP services
  harness/           model loop, gate, clients, parser, logging, fake model
tasks.json           12 task configurations
tests/               unit, integration and end-to-end tests
docs/                design document and deck
baseline-results/    all-task baseline gate evidence
sample-trajectories/ offline success/failure trajectories
test-run-logs/       local run evidence
```

## Evidence and artifacts

- `baseline-results/gate.json` — oracle/no-op/allow-all/deny-all gate for all
  twelve tasks.
- `sample-trajectories/` — offline solver success and no-op failure traces.
- `test-run-logs/` — gate, pytest and generation-soak logs from local verification.
- `docs/design-doc.md` — full design and threat model.
- `docs/deck.md` — short presentation deck.

## Known limitations

- Real-model calibration has not been run locally because no hackathon API key
  or GPT-Luna endpoint was available. The harness supports any OpenAI-compatible
  chat-completions endpoint.
- Docker is not installed on the local development machine, so the Compose
  build and container isolation are validated by configuration review and
  in-process tests rather than `docker compose` locally.
- Hard tasks are slower to generate because the generator scores candidate
  mutations against the sampled matrix. Standalone hard-task generation took
  roughly 8-20 seconds locally; the all-task gate took about 43 seconds.
- The offline `mock/solver` consumes oracle actions from the harness. It is a
  deterministic gate/demo aid, not a model baseline.

## Security boundaries

- The model only sees messages built by the harness and tool output from the
  env. It never sees the verifier's answer key.
- The verifier is not on the env network.
- The env has no internet route and drops all Linux capabilities.
- `APPEND_FILE` is confined to `inputs/`, must be UTF-8, and is copied into the
  run log for auditability.
- `API_KEY` is read only by the harness; it is not written to summaries or logs.

## Reproducing the gate

```bash
# All twelve tasks, no model:
MODE=baselines SEED=0 docker compose up --build --abort-on-container-exit --exit-code-from harness
```

Or locally:

```powershell
$env:MODE="baselines"
$env:SEED="0"
& ".\.venv\Scripts\python.exe" -m gatekeeper.harness.run
```
