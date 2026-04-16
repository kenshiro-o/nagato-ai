# T6 — Verification: unit tests, end-to-end scenarios, smoke

## Goal of the task

Confirm the epic's success criteria from `prd.md`:

1. `GoalRunner` converges on a trivial testable goal within a
   small iteration budget.
2. `GoalRunner` terminates on a hazy goal with an LLM-judge gate.
3. `GoalRunner` terminates cleanly (not by exception) on an
   unsatisfiable goal.
4. No regression in existing tests.

## Unit tests

### `tests/nagatoai_core/goal/test_gate.py`

- `ProgrammaticGate` — deterministic pass and fail cases.
- `LLMJudgeGate` — mock the `Agent` to return canned XML; assert
  the gate parses it correctly. Include one malformed-XML case
  and assert a sensible failure surface.
- `CompositeGate`
  - `mode="all"`: all children pass → achieved; any fails → not.
  - `mode="any"`: any child passes → achieved; all fail → not.
  - `reason` is a concatenation of child reasons, labelled.

### `tests/nagatoai_core/graph/test_delta.py`

- Round-trip each op in isolation: build a trivial graph, apply
  a single-op delta, assert the result matches the expected
  shape.
- Cycle-inducing delta: `apply_delta` raises `ValueError` from
  `Graph.compile`.
- Input graph is **not** mutated after `apply_delta` (pure
  function property).
- `parse_delta` round-trip: XML → `GraphDelta` → `apply_delta` →
  `Graph` matches an expected state.
- `XMLPlanParser.serialize` → `XMLPlanParser.parse_graph` round
  trip on a handcrafted graph of each flow type.

### `tests/nagatoai_core/planner/test_planner.py`

- Happy path: mock `planner_agent` to return a canned
  `<delta>` XML; assert `propose_delta` returns the expected
  `GraphDelta`.
- Parse retry: first response is malformed, second is valid.
  Assert retry happens and the valid response is returned.
- Retry exhaustion: both responses malformed; assert
  `PlannerError` is raised.
- Fallback mode: `delta_mode="replace"`, mock agent returns a
  `<graph>`; assert a synthesized `GraphDelta` that replaces the
  current graph.

### `tests/nagatoai_core/runner/test_goal_runner.py`

- **Convergence.** Build a stub graph and a mock planner whose
  deltas provably move the output toward a target. Gate is a
  `ProgrammaticGate` checking the target. Assert
  `terminated_reason == ACHIEVED` within a small number of
  iterations.
- **Budget exhaustion.** Planner always proposes a non-trivial,
  non-repeating delta; gate never accepts. Assert
  `BUDGET_EXHAUSTED`.
- **Stuck: empty delta.** Planner returns `GraphDelta(ops=[])`.
  Assert `STUCK_EMPTY_DELTA`.
- **Stuck: repeat verdict.** Gate returns identical
  `verdict.reason` twice. Assert `STUCK_REPEAT_VERDICT`.
- **Stuck: repeat delta.** Planner returns identical delta
  twice. Assert `STUCK_REPEAT_DELTA`.
- **Planner failure.** Planner raises `PlannerError`. Assert
  `PLANNER_FAILED` and no exception escapes `run`.
- **Event history.** Assert `history` has one entry per
  iteration in order.

## End-to-end scenarios

Gate behind an env var (e.g. `NAGATO_RUN_E2E=1`) like other
examples in the repo. These hit real models and should not run in
CI by default.

### E2E-1: Testable goal — "produce 42"

- `Goal(description="produce the number 42")`.
- Initial graph: empty (or single `AgentNode` stub).
- `ProgrammaticGate` with predicate: "final output parses as int
  and equals 42".
- Planner: real Opus 4.6 agent.
- Budget: 5 iterations.
- Assertion: `terminated_reason == ACHIEVED` and the final
  output equals 42.
- **This is the canonical "does the loop actually close?" test.**
  If this does not pass, nothing else matters.

### E2E-2: Hazy goal — "summarize article in 3 bullets"

- Fixed input: a short article as a `NodeResult`.
- `Goal(description="summarize this article in three clear
  bullet points")`.
- `LLMJudgeGate` backed by an Opus 4.6 agent.
- Planner: real Opus 4.6 agent, distinct instance from the judge.
- Budget: 5 iterations.
- Assertion: `terminated_reason == ACHIEVED`; final output
  contains exactly three bullet points.

### E2E-3: Unsatisfiable goal

- `Goal(description="produce a proof that P=NP")` or similar
  deliberately-impossible target.
- `LLMJudgeGate` that will refuse.
- Budget: 3 iterations.
- Assertion: `terminated_reason` is one of
  `{STUCK_EMPTY_DELTA, STUCK_REPEAT_VERDICT, STUCK_REPEAT_DELTA,
  BUDGET_EXHAUSTED, PLANNER_FAILED}`. Critically: **no exception
  escapes `GoalRunner.run`.**

## Manual smoke

- Take the existing
  `docs/examples/youtube_video_highlights_clip_extraction_chain_example.py`
  and rewrite it as a `GoalRunner` invocation with a stub graph
  and a plain-English goal. Watch the planner extend the graph
  step by step until the output satisfies the gate.
- Paste the rendered `history` (one line per iteration) into the
  PR description so reviewers can see the loop in action.

## Commands

```bash
uv run pytest tests/nagatoai_core/goal \
              tests/nagatoai_core/graph/test_delta.py \
              tests/nagatoai_core/planner \
              tests/nagatoai_core/runner/test_goal_runner.py

# Run the full existing suite to confirm no regression
make test

# Formatting and lint
make fmt
make lint

# E2E (gated)
NAGATO_RUN_E2E=1 uv run pytest -k "e2e" -s
```

## Done when

- All unit tests pass locally.
- `make test` is green on the full existing suite (no
  regression).
- E2E-1 passes with a real Opus 4.6 agent.
- E2E-3 demonstrates clean termination on an unsatisfiable goal.
- Manual smoke output is captured in the PR description.
