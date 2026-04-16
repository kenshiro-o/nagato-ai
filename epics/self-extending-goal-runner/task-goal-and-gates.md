# T1 — Goal + Gate module

## Goal of the task

Introduce a minimal, composable abstraction for describing what the
user wants and for checking whether a graph's output satisfies it.
The goal representation must stay flexible enough to cover both
strict, programmatic acceptance (numbers, strings, typed predicates)
and hazy, LLM-judged acceptance (prose, summaries, qualitative
requirements).

## Files to create

- `nagatoai_core/goal/__init__.py`
- `nagatoai_core/goal/goal.py`
- `nagatoai_core/goal/gate.py`

## Design

### `Goal` (in `goal.py`)

```
class Goal(BaseModel):
    description: str                       # plain-English statement
    structured_spec: Optional[Dict] = None # optional typed refinement
```

- `description` is the canonical statement; it's what gets fed to
  the planner and to any `LLMJudgeGate`.
- `structured_spec` is a free-form dict the caller can use when the
  goal has a typed shape ("return a JSON object matching schema X").
  Left intentionally loose — we don't yet know what structure the
  planner will want to consume.

### `GateVerdict` (in `gate.py`)

```
class GateVerdict(BaseModel):
    achieved: bool
    reason: str                # human-readable, fed back to planner
    evidence: Any = None       # optional: the inspected output, a diff, etc.
```

`reason` is load-bearing: the planner uses it to decide what to
change in the next delta. A gate that fails silently is useless.

### `AbstractGate` (in `gate.py`)

```
class AbstractGate(BaseModel, ABC):
    @abstractmethod
    def evaluate(self, goal: Goal, graph_output: List[NodeResult]) -> GateVerdict: ...
```

### Concrete gates (in `gate.py`)

1. **`ProgrammaticGate`** — wraps a user-supplied callable.

   ```
   class ProgrammaticGate(AbstractGate):
       predicate: Callable[[Goal, List[NodeResult]], GateVerdict]

       def evaluate(self, goal, graph_output):
           return self.predicate(goal, graph_output)
   ```

   Use when the goal is a concrete number, string, file-on-disk,
   test-that-passes, etc. Deterministic. Fast. No LLM cost.

2. **`LLMJudgeGate`** — wraps an `Agent` that judges against the
   plain-English goal.

   ```
   class LLMJudgeGate(AbstractGate):
       judge_agent: Agent
       prompt_template: str = LLM_JUDGE_PROMPT  # defined alongside

       def evaluate(self, goal, graph_output):
           # serialize graph_output → string
           # render prompt → (goal.description, serialized_output)
           # call judge_agent, parse XML verdict with BeautifulSoup
           # (same pattern as SingleCriticEvaluator lines 77–80)
           # return GateVerdict
   ```

   Parse the judge's response using the same BeautifulSoup pattern
   already in `single_critic_evaluator.py:77-80`, for stylistic
   consistency with the rest of the codebase. Expected XML from the
   judge:

   ```xml
   <verdict>
     <achieved>true|false</achieved>
     <reason>...</reason>
   </verdict>
   ```

3. **`CompositeGate`** — AND / OR composition of child gates.

   ```
   class CompositeGate(AbstractGate):
       gates: List[AbstractGate]
       mode: Literal["all", "any"] = "all"

       def evaluate(self, goal, graph_output):
           # run each child; combine per mode
           # reason = concatenation of child reasons, labelled by index
           # achieved = all(...) or any(...)
   ```

   Why both modes: the user may want a strict programmatic check
   **plus** an LLM sanity judge (`"all"`), or may want to accept
   either a programmatic hit **or** a human-equivalent judgment
   (`"any"`).

## Prompt constants

Define `LLM_JUDGE_PROMPT` in `gate.py` or in a small
`nagatoai_core/goal/prompts.py` — author's preference. Keep it in
the epic module rather than `nagatoai_core/prompt/templates.py` so
goal-related prompts stay co-located with the code that uses them.
Model after the existing `CRITIC_PROMPT` style: clear role
description, the goal statement, the output to evaluate, and an
explicit output-format block asking for `<verdict>...</verdict>`.

## Notes

- Do **not** import `Graph` here — gates consume `List[NodeResult]`
  only, so they're decoupled from the executor. This lets gates be
  unit-tested without building a graph.
- `structured_spec` is intentionally `Dict` not `BaseModel` for
  flexibility; callers who want type-safety can bolt on their own
  Pydantic models and serialize.
- Keep every class a `BaseModel` subclass to match the project-wide
  convention (see `CLAUDE.md` — "All non-test classes must inherit
  from Pydantic `BaseModel`").

## Done when

- Module imports cleanly.
- `ProgrammaticGate`, `LLMJudgeGate`, `CompositeGate` all implement
  `AbstractGate.evaluate`.
- `make fmt && make lint` pass.
- Unit tests in `tests/nagatoai_core/goal/test_gate.py` cover all
  three gates (see `task-verification.md`).
