# T4 — Planner agent + prompts

> **This is the load-bearing task of the epic.** If the planner
> cannot reliably emit a well-formed, useful graph delta, the entire
> outer loop is dead. Validate it in isolation before wiring it
> into the runner.

## Goal of the task

Given the current graph, the last iteration's output, and a gate
verdict that says "not yet, because X", produce a minimal graph
delta that moves the plan closer to satisfying the goal.

## Files to create

- `nagatoai_core/planner/__init__.py`
- `nagatoai_core/planner/planner.py`
- `nagatoai_core/planner/prompts.py`

## Design

### `Planner` (in `planner.py`)

```
class Planner(BaseModel):
    planner_agent: Agent
    delta_mode: Literal["incremental", "replace"] = "incremental"
    max_parse_retries: int = 1

    class Config:
        arbitrary_types_allowed = True

    def propose_delta(
        self,
        goal: Goal,
        graph: Graph,
        execution_trace: List[NodeResult],
        gate_verdict: GateVerdict,
    ) -> GraphDelta: ...
```

### `propose_delta` — step by step

1. **Serialize current graph.** Call
   `XMLPlanParser.serialize(graph)` to produce the XML view the
   planner will reason about.
2. **Build the prompt.** Render `PLANNER_PROMPT` with:
   - `goal.description`
   - `current_plan_xml`
   - a compact dump of `execution_trace` (last-iteration output)
   - `gate_verdict.reason`
   - a single few-shot example of a well-formed `<delta>`
3. **Call the planner agent.** Use the same
   `common.send_agent_request` helper pattern used elsewhere in
   the runner layer, with conservative temperature (≤0.4 — we
   want consistency, not creativity).
4. **Parse.** Extract the `<delta>` block with BeautifulSoup, pass
   the XML to `XMLPlanParser.parse_delta(xml, graph)`.
5. **Retry on malformed XML.** If parsing fails, re-prompt the
   planner once with the parser error appended. After
   `max_parse_retries` failures, raise `PlannerError` — the runner
   will convert that into a stuck-iteration event.
6. **Fallback mode.** If `delta_mode == "replace"`, expect a full
   `<graph>` from the planner instead of `<delta>`. Parse it via
   the existing `XMLPlanParser.parse_graph`, diff it against the
   current graph, and synthesize a `GraphDelta` that transforms
   current → new (or just drop-and-replace semantics — whichever
   is simpler at implementation time).

### `PLANNER_PROMPT` (in `prompts.py`)

Structure the prompt like the existing `CRITIC_PROMPT` in
`nagatoai_core/prompt/templates.py` — explicit role, clearly
labelled input sections, explicit output format block.

Contents the prompt **must** contain, in order:

1. **Role.** "You are the planner layer of Nagato-AI. Your job is
   to emit minimal modifications to an execution graph so that it
   makes progress toward the user's goal."
2. **Minimality rule.** "Emit the smallest delta that could
   plausibly satisfy the verdict. Do not rewrite the graph if you
   can add a single node. Large deltas are a smell."
3. **Goal.** Plain English.
4. **Current plan (XML).** The full serialized graph.
5. **Last iteration's output.** A trimmed dump — cap to N characters
   to avoid context bloat.
6. **Verdict.** The gate's reason string verbatim.
7. **Output format.** Must emit a `<delta>...</delta>` block using
   the schema documented in `task-graph-delta.md`. Nothing else.
8. **Few-shot example.** One example showing a minimal delta that
   adds a single node.

## Validation before wiring

Before integrating the `Planner` with `GoalRunner`, verify it in
isolation:

1. Handcraft three `(goal, current_graph, verdict)` triples for
   which the ideal delta is obvious.
2. Run the planner against each one using Opus 4.6.
3. Inspect the emitted XML manually. Grade on:
   - Parses into a valid `GraphDelta`.
   - Applies cleanly to the current graph.
   - Is minimal (does not rewrite the world for a small gap).
   - Actually addresses the verdict's reason.
4. If < 2 of 3 pass, **do not proceed** to the runner. Either
   refine the prompt or flip `delta_mode` default to `"replace"`
   and retry.

Document the three scenarios and their outcomes inline in the PR
description so the behaviour is reviewable.

## Errors

- `PlannerError(Exception)` — raised when the planner cannot
  produce a usable delta after retries. The runner catches this
  and emits a stuck-iteration event; it never propagates out of
  `GoalRunner.run`.

## Risks

- **Token bloat from large graph serialization.** Cap the current
  plan at N KB before embedding in the prompt; warn when the cap
  is hit. The same strategy for the execution trace.
- **Planner loops.** The planner may propose the same delta twice.
  That's detected in `GoalRunner`'s stuck-detection (see
  `task-goal-runner.md`), not here — keep the planner itself
  stateless.
- **Model drift.** The prompt is tuned to Opus 4.6 today. If you
  switch to a cheaper model, re-validate with the three scenarios
  above.

## Done when

- `Planner.propose_delta` returns a parseable, applicable
  `GraphDelta` for the three validation scenarios.
- Unit test in `tests/nagatoai_core/planner/test_planner.py`
  covers the happy path with a mocked agent, the parse-retry path,
  and the fallback-to-replace path.
- `PlannerError` exists and is raised on retry exhaustion.
- `make fmt && make lint` pass.
