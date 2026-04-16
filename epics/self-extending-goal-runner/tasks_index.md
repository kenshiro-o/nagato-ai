# Task Index — Self-Extending Goal Runner

Status legend: `[ ]` pending · `[~]` in progress · `[x]` done ·
`[!]` blocked.

| Status | ID  | Title                                    | Hook                                                                                          | Detail                        |
| ------ | --- | ---------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------- |
| `[x]`  | T1  | Goal + Gate module                       | `Goal` model, `AbstractGate`, Programmatic/LLMJudge/Composite gates                           | [task-goal-and-gates.md](./task-goal-and-gates.md) |
| `[x]`  | T2  | `GraphDelta` + `apply_delta`             | Typed delta ops, pure `apply` returning a new `Graph`, compile-failure path                   | [task-graph-delta.md](./task-graph-delta.md) |
| `[x]`  | T3  | XML delta parser + graph serializer      | `parse_delta()` and `serialize(graph)`; reuses existing `parse_nodes`                         | [task-graph-delta.md](./task-graph-delta.md) |
| `[x]`  | T4  | Planner agent + prompts                  | Plan serialization, delta emission, retry, full-plan fallback — load-bearing, most risk       | [task-planner.md](./task-planner.md) |
| `[x]`  | T5  | `GoalRunner` + `RunBudget`               | Outer loop, event model, stuck detection (3 triggers), structlog + optional Langfuse          | [task-goal-runner.md](./task-goal-runner.md) |
| `[x]`  | T6  | Unit + end-to-end verification           | Gate/delta/planner/runner unit tests + three E2E scenarios + manual smoke                     | [task-verification.md](./task-verification.md) |

## Dependencies

- T1 and (T2, T3) are independent and may proceed in parallel.
- T4 depends on T3 (needs the graph serializer and delta parser).
- T5 depends on T1 + T2 + T4.
- T6 runs last, but unit-level tests for T1–T5 should be written
  alongside each task; the E2E scenarios in T6 require the whole stack.

## Suggested order

1. T1 (Goal + Gate) in parallel with T2 + T3 (delta module + XML
   extension). Self-contained, low risk.
2. T4 (Planner) — validate in isolation before wiring in. If delta
   emission is unreliable, flip `delta_mode` default to `"replace"`
   and re-evaluate.
3. T5 (GoalRunner) — compose the pieces.
4. T6 (verification + E2E) — close the loop.
