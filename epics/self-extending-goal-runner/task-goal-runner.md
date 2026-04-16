# T5 — GoalRunner + RunBudget + stuck detection

## Goal of the task

Compose `Goal`, `AbstractGate`, `Planner`, and `Graph` into an
outer loop that iteratively executes → checks → plans → applies
delta → re-executes, until the goal is achieved, the budget is
exhausted, or the loop is detected as stuck. Every iteration must
be observable (structured event model + structlog + optional
Langfuse).

## Files to create

- `nagatoai_core/runner/goal_runner.py`

## Design

### `RunBudget` (in `goal_runner.py`)

```
class RunBudget(BaseModel):
    max_iterations: int = 8
    max_total_tokens: Optional[int] = 200_000
    max_wall_time_seconds: Optional[float] = None
```

Budget is **mandatory at the API level** — defaults are
conservative. Callers raise them explicitly when they want longer
runs. No silent unbounded runs.

### `IterationEvent` (in `goal_runner.py`)

```
class IterationEvent(BaseModel):
    index: int
    graph_before_xml: str          # snapshot via XMLPlanParser.serialize
    graph_output: List[NodeResult]
    verdict: GateVerdict
    proposed_delta: Optional[GraphDelta]  # None on final iteration
    applied: bool                  # False if compile failed
    error: Optional[str] = None    # PlannerError, compile error, etc.
```

### `GoalRunResult` (in `goal_runner.py`)

```
class TerminatedReason(str, Enum):
    ACHIEVED = "achieved"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STUCK_EMPTY_DELTA = "stuck_empty_delta"
    STUCK_REPEAT_VERDICT = "stuck_repeat_verdict"
    STUCK_REPEAT_DELTA = "stuck_repeat_delta"
    PLANNER_FAILED = "planner_failed"

class GoalRunResult(BaseModel):
    terminated_reason: TerminatedReason
    final_output: List[NodeResult]
    final_graph: Graph
    history: List[IterationEvent]
```

### `GoalRunner` (in `goal_runner.py`)

```
class GoalRunner(BaseModel):
    goal: Goal
    initial_graph: Graph
    gate: AbstractGate
    planner: Planner
    budget: RunBudget = Field(default_factory=RunBudget)
    tracing_enabled: bool = False

    class Config:
        arbitrary_types_allowed = True

    def run(self, inputs: List[NodeResult]) -> GoalRunResult: ...
```

### Outer loop

```
def run(self, inputs):
    logger = structlog.get_logger(...)
    history: List[IterationEvent] = []
    graph = self.initial_graph
    previous_verdict_reason = None
    previous_delta_signature = None
    start_time = time.time()
    tokens_used = 0

    for i in range(self.budget.max_iterations):
        # 1. execute
        output = graph.run(inputs)

        # 2. check gate
        verdict = self.gate.evaluate(self.goal, output)
        logger.info("goal_iteration", index=i, achieved=verdict.achieved, reason=verdict.reason)

        if verdict.achieved:
            history.append(IterationEvent(
                index=i,
                graph_before_xml=XMLPlanParser().serialize(graph),
                graph_output=output,
                verdict=verdict,
                proposed_delta=None,
                applied=False,
            ))
            return GoalRunResult(
                terminated_reason=TerminatedReason.ACHIEVED,
                final_output=output,
                final_graph=graph,
                history=history,
            )

        # 3. budget checks (wall time + tokens, iteration count is the for-loop)
        if self._budget_exhausted(start_time, tokens_used):
            # record last event before returning
            ...
            return GoalRunResult(terminated_reason=BUDGET_EXHAUSTED, ...)

        # 4. plan delta
        try:
            delta = self.planner.propose_delta(self.goal, graph, output, verdict)
        except PlannerError as e:
            history.append(IterationEvent(..., error=str(e)))
            return GoalRunResult(terminated_reason=PLANNER_FAILED, ...)

        # 5. stuck detection
        delta_sig = _signature_of(delta)
        if not delta.ops:
            return GoalRunResult(terminated_reason=STUCK_EMPTY_DELTA, ...)
        if verdict.reason == previous_verdict_reason:
            return GoalRunResult(terminated_reason=STUCK_REPEAT_VERDICT, ...)
        if delta_sig == previous_delta_signature:
            return GoalRunResult(terminated_reason=STUCK_REPEAT_DELTA, ...)

        # 6. apply delta
        try:
            new_graph = apply_delta(graph, delta)
        except ValueError as e:
            # compile failed — surface to planner next iteration as part of verdict.reason
            history.append(IterationEvent(..., applied=False, error=str(e)))
            # continue with SAME graph but pass error into next verdict reason
            # (alternative: treat as stuck after N consecutive compile failures)
            previous_verdict_reason = f"compile_failed: {e}"
            previous_delta_signature = delta_sig
            continue

        history.append(IterationEvent(
            index=i,
            graph_before_xml=XMLPlanParser().serialize(graph),
            graph_output=output,
            verdict=verdict,
            proposed_delta=delta,
            applied=True,
        ))

        graph = new_graph
        previous_verdict_reason = verdict.reason
        previous_delta_signature = delta_sig

    # fell out of loop
    return GoalRunResult(terminated_reason=BUDGET_EXHAUSTED, ...)
```

### Stuck detection — three triggers

| Trigger                          | Meaning                                          | Action                       |
| -------------------------------- | ------------------------------------------------ | ---------------------------- |
| Empty delta                      | Planner has nothing to propose.                  | `STUCK_EMPTY_DELTA`          |
| Verdict reason unchanged 2×      | Progress indicator is flat.                      | `STUCK_REPEAT_VERDICT`       |
| Identical delta proposed 2×      | Planner is cycling without learning.             | `STUCK_REPEAT_DELTA`         |

Any trigger returns a valid `GoalRunResult` — never an exception.

### Delta signature

A hash of the ordered ops. Simple approach: serialize `delta.ops`
to JSON (Pydantic gives us this free) and hash with SHA-1. Sort is
intentionally **not** applied — ordering matters for correctness
and should matter for equality.

### Logging + tracing

- **structlog** — already project convention (commit `1cafac1`).
  Log one event per iteration with `index`, `achieved`,
  `verdict.reason`, `delta_op_count`.
- **Langfuse** — optional. If `tracing_enabled=True`, create a
  trace at the start of `run` and a `span` per iteration, mirroring
  the pattern in `single_critic_evaluator.py:84-107`.

## Non-goals for this task

- Goal decomposition. The `GoalRunner` handles a single atomic
  goal; sub-goals are future work.
- Concurrent multiple-planner consultation.
- Cost accounting per-iteration (beyond token counting against
  the budget).
- Mid-run human intervention hooks.

## Risks

- **Compile-failure loops.** If every delta the planner produces
  fails `Graph.compile`, the runner currently retries with the
  same graph and a new verdict reason. This could loop. Consider
  a counter: after N consecutive compile failures, return
  `STUCK_REPEAT_DELTA`.
- **Slow serialization.** `XMLPlanParser.serialize` is called
  twice per iteration (once for the event, once for the planner).
  Call it once and reuse.
- **Token accounting.** `max_total_tokens` needs access to token
  counts from the planner agent's responses. Agents already track
  `token_stats_and_params` (see critic evaluator lines 102-107);
  accumulate those per iteration.

## Done when

- `GoalRunner.run` returns a `GoalRunResult` for every code path,
  never raises.
- All three stuck triggers are implemented and tested.
- Budget exhaustion returns cleanly.
- Each iteration emits a structlog event and populates one entry
  in `history`.
- `make fmt && make lint` pass.
- Unit tests in `tests/nagatoai_core/runner/test_goal_runner.py`
  cover: convergence, budget exhaustion, all three stuck triggers.
  See `task-verification.md`.
