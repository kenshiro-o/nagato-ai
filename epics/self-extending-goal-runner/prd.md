# PRD: Self-Extending, Goal-Verifying Execution Loop

## Why this exists

Nagato-AI's original vision was to let a user state a plain-English goal and
have a program iteratively build and verify itself until that goal is met.
In a design discussion we clarified that "Turing-completeness" was a proxy
for what actually matters:

> **Self-extension + self-verification + convergence on the user's goal
> under a bounded budget.**

The aesthetic framing of "an agent harness as a new kind of computer" is
preserved, but the technical north star is **goal achievement**, not raw
computational expressiveness. A Turing-complete system that never
terminates is worse than a finite system that reliably converges.

## Current state (what is already in the repo)

- `nagatoai_core/graph/graph.py` is a static DAG.
  - `Graph.compile()` forbids cycles (line 73).
  - `Graph.run()` performs a single topological pass (line 169).
  - No re-entry, no mutable state across nodes, no runtime mutation.
- Flow primitives (`SequentialFlow`, `ParallelFlow`, `ConditionalFlow`,
  `TransformerFlow`, `UnfoldFlow`) are pure-functional compositions.
  `UnfoldFlow` is flatten+broadcast, **not** iteration.
- `runner/single_critic_evaluator.py` invokes a critic that emits an
  `(outcome, evaluation)` pair. The verdict is **terminal** — nothing
  feeds it back into plan mutation.
- `graph/plan/xml_parser.py` instantiates the whole graph up front; no
  delta / partial-plan / incremental-expansion schema exists.

The "keeps building" and "keeps verifying itself" parts of the original
vision are not in the repo at all. This epic adds them.

## Architectural split

- **Dumb executor** — the existing `Graph` stays acyclic, single-pass,
  analyzable. No loops, no mutable state, no runtime mutation inside the
  executor. All properties that make it debuggable are preserved.
- **Smart planner layer (new)** — sits above `Graph`, owns the iteration,
  the goal check, the graph mutation, and the budget. All new
  intelligence lives here. The planner is itself an agent, so Nagato
  dogfoods its own abstractions.

## Target behaviour

```
GoalRunner(goal, initial_graph, gate, planner, budget).run(inputs)
  └─ loop until: goal achieved | budget exhausted | stuck
       1. graph.run(inputs)                    # dumb executor, unchanged
       2. gate.evaluate(goal, output)          # programmatic or LLM-judge
       3. if achieved → return output
       4. planner.propose_delta(               # smart layer
              goal, graph, execution_trace, gate_verdict)
       5. graph = apply_delta(graph, delta)    # returns a new Graph
       6. if delta is no-op OR verdict stable for N → mark stuck, return
```

The inner `Graph.run` never changes shape. All "iteration" is in
`GoalRunner`.

## Audience

1. **The user as a research substrate.** Primary audience today. The
   epic should favour conceptual clarity over ergonomics when the two
   conflict.
2. **Other Nagato-AI developers as a production library.** Secondary.
   Design choices must not foreclose hardening — no permanent
   research-y shortcuts, no APIs that would need to be renamed on the
   way to production.

## Success criteria

- A new `GoalRunner` can take a plain-English goal and a stub graph,
  drive an outer loop that alternates execution → gate check → planner
  delta, and converge on a simple testable goal ("produce 42") within
  a small number of iterations and a bounded token budget.
- The same runner can handle a hazy goal ("summarize this article in
  3 bullets") using an LLM-judge gate, and terminate.
- The runner terminates cleanly — never by exception — on an
  unsatisfiable goal, reporting the reason.
- The existing `Graph`, flows, nodes, `SingleAgentTaskRunner`,
  `SingleCriticEvaluator`, and XML plan schema continue to work
  unchanged. No regression in existing tests.

## What does NOT change

- `Graph`, `Graph.run`, `Graph.compile`, all `Flow` / `Node` classes.
- Cycle detection in `Graph.compile`.
- `SingleAgentTaskRunner`, `SingleCriticEvaluator`, `Mission`, `Task`.
  `GoalRunner` is a **parallel** runner, not a replacement. Users who
  don't need self-extension keep using the existing single-shot path.
- Existing XML plan schema. We only **add** a new `<delta>` element;
  existing `<graph>` plans continue to parse identically.

## Out of scope (future work)

- **Restricted Python DSL ↔ XML.** A future `nagatoai_core/graph/dsl/`
  module with `sequential(...)`, `parallel(...)`, `conditional(...)`,
  `agent(...)`, `tool(...)`, `unfold(...)` builder functions plus an
  AST visitor that lifts a DSL script into XML for audit. The planner
  would emit Python DSL (cleaner for the model) while XML remains the
  typed, diffable intermediate representation. We validate the
  self-extending loop with XML deltas first, then layer the DSL on top.
- Goal decomposition into sub-goals.
- Multi-planner councils / delta voting.
- Cost-aware delta ranking.
- Semantic (not just structural) validation of deltas.
