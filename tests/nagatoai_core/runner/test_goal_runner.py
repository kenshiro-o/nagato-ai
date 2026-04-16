# Standard Library
from typing import List
from unittest.mock import MagicMock

# Nagato AI
from nagatoai_core.goal.gate import GateVerdict, ProgrammaticGate
from nagatoai_core.goal.goal import Goal
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.delta import AddEdge, GraphDelta, RemoveEdge
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.types import NodeResult
from nagatoai_core.planner.planner import Planner, PlannerError
from nagatoai_core.runner.goal_runner import GoalRunner, RunBudget, TerminatedReason


class CounterNode(AbstractNode):
    """Node that returns its input incremented by a fixed value."""

    increment: int = 1

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        results = []
        for inp in inputs:
            val = inp.result if isinstance(inp.result, (int, float)) else 0
            results.append(NodeResult(node_id=self.id, result=val + self.increment))
        return results


def _build_two_node_graph():
    """a(+1) -> b(+2): input 0 produces 3."""
    a = CounterNode(id="a", name="node_a", increment=1)
    b = CounterNode(id="b", name="node_b", increment=2)
    g = Graph()
    g.add_edge(a, b)
    g.compile()
    return g


def _make_converging_gate(target: int, converge_on_iteration: int):
    """Gate that achieves on a specific iteration count."""
    call_count = {"n": 0}

    def predicate(goal: Goal, output: List[NodeResult]) -> GateVerdict:
        call_count["n"] += 1
        if call_count["n"] >= converge_on_iteration:
            return GateVerdict(achieved=True, reason="goal met")
        return GateVerdict(achieved=False, reason=f"iteration {call_count['n']}, not yet")

    return ProgrammaticGate(predicate=predicate)


def _make_never_gate():
    """Gate that never achieves."""

    def predicate(goal: Goal, output: List[NodeResult]) -> GateVerdict:
        return GateVerdict(achieved=False, reason="never achieved")

    return ProgrammaticGate(predicate=predicate)


def _make_mock_planner(deltas: List[GraphDelta], *, cycle: bool = False):
    """Create a mock Planner that returns deltas from a list in order.

    If cycle=True, wraps around; otherwise clamps to the last delta.
    """
    planner = MagicMock(spec=Planner)
    call_count = {"n": 0}

    def side_effect(goal, graph, trace, verdict):
        if cycle:
            idx = call_count["n"] % len(deltas)
        else:
            idx = min(call_count["n"], len(deltas) - 1)
        call_count["n"] += 1
        return deltas[idx], None

    planner.propose_delta.side_effect = side_effect
    return planner


def _noop_delta_for_graph(graph):
    """A delta that removes and re-adds an edge, producing a structurally equivalent graph but with unique ops."""
    for source_id, targets in graph.adjacency_list.items():
        for target_id in targets:
            return GraphDelta(
                ops=[
                    RemoveEdge(source_id=source_id, target_id=target_id),
                    AddEdge(source_id=source_id, target_id=target_id),
                ]
            )
    return GraphDelta(ops=[])


def test_convergence():
    """GoalRunner returns ACHIEVED when gate passes on iteration 2."""
    graph = _build_two_node_graph()
    goal = Goal(description="produce something")
    gate = _make_converging_gate(target=42, converge_on_iteration=2)

    # Use unique deltas to avoid repeat-delta stuck detection
    deltas = [
        GraphDelta(ops=[RemoveEdge(source_id="a", target_id="b"), AddEdge(source_id="a", target_id="b")]),
        GraphDelta(
            ops=[
                AddEdge(source_id="a", target_id="b"),
                RemoveEdge(source_id="a", target_id="b"),
                AddEdge(source_id="a", target_id="b"),
            ]
        ),
    ]
    planner = _make_mock_planner(deltas)

    runner = GoalRunner(
        goal=goal,
        initial_graph=graph,
        gate=gate,
        planner=planner,
        budget=RunBudget(max_iterations=5, max_total_tokens=None),
    )
    result = runner.run([NodeResult(node_id="input", result=0)])

    assert result.terminated_reason == TerminatedReason.ACHIEVED
    assert len(result.history) == 2


def test_budget_exhausted():
    """GoalRunner returns BUDGET_EXHAUSTED when max_iterations reached."""
    graph = _build_two_node_graph()
    goal = Goal(description="impossible goal")

    # Gate with varying reasons to avoid repeat-verdict stuck
    vary_count = {"n": 0}

    def varying_predicate(goal, output):
        vary_count["n"] += 1
        return GateVerdict(achieved=False, reason=f"not met attempt {vary_count['n']}")

    gate = ProgrammaticGate(predicate=varying_predicate)

    deltas = [
        GraphDelta(ops=[RemoveEdge(source_id="a", target_id="b"), AddEdge(source_id="a", target_id="b")]),
        GraphDelta(
            ops=[
                AddEdge(source_id="a", target_id="b"),
                RemoveEdge(source_id="a", target_id="b"),
                AddEdge(source_id="a", target_id="b"),
            ]
        ),
        GraphDelta(
            ops=[
                RemoveEdge(source_id="a", target_id="b"),
                AddEdge(source_id="a", target_id="b"),
                RemoveEdge(source_id="a", target_id="b"),
                AddEdge(source_id="a", target_id="b"),
            ]
        ),
    ]
    planner = _make_mock_planner(deltas)

    runner = GoalRunner(
        goal=goal,
        initial_graph=graph,
        gate=gate,
        planner=planner,
        budget=RunBudget(max_iterations=3, max_total_tokens=None),
    )
    result = runner.run([NodeResult(node_id="input", result=0)])

    assert result.terminated_reason == TerminatedReason.BUDGET_EXHAUSTED
    assert len(result.history) == 3


def test_stuck_empty_delta():
    """GoalRunner returns STUCK_EMPTY_DELTA when planner proposes empty ops."""
    graph = _build_two_node_graph()
    goal = Goal(description="produce something")
    gate = _make_never_gate()
    planner = _make_mock_planner([GraphDelta(ops=[])])

    runner = GoalRunner(
        goal=goal,
        initial_graph=graph,
        gate=gate,
        planner=planner,
        budget=RunBudget(max_iterations=5, max_total_tokens=None),
    )
    result = runner.run([NodeResult(node_id="input", result=0)])

    assert result.terminated_reason == TerminatedReason.STUCK_EMPTY_DELTA
    assert len(result.history) == 1


def test_stuck_repeat_verdict():
    """GoalRunner returns STUCK_REPEAT_VERDICT when verdict reason is unchanged."""
    graph = _build_two_node_graph()
    goal = Goal(description="produce something")

    # Gate always returns same reason
    def same_reason_predicate(goal, output):
        return GateVerdict(achieved=False, reason="always the same")

    gate = ProgrammaticGate(predicate=same_reason_predicate)

    delta = _noop_delta_for_graph(graph)
    planner = _make_mock_planner([delta, delta])

    runner = GoalRunner(
        goal=goal,
        initial_graph=graph,
        gate=gate,
        planner=planner,
        budget=RunBudget(max_iterations=5, max_total_tokens=None),
    )
    result = runner.run([NodeResult(node_id="input", result=0)])

    assert result.terminated_reason == TerminatedReason.STUCK_REPEAT_VERDICT
    assert len(result.history) == 2


def test_stuck_repeat_delta():
    """GoalRunner returns STUCK_REPEAT_DELTA when planner proposes same delta twice."""
    graph = _build_two_node_graph()
    goal = Goal(description="produce something")

    call_count = {"n": 0}

    def varying_verdict(goal, output):
        call_count["n"] += 1
        return GateVerdict(achieved=False, reason=f"fail {call_count['n']}")

    gate = ProgrammaticGate(predicate=varying_verdict)

    # Same delta every time
    same_delta = _noop_delta_for_graph(graph)
    planner = _make_mock_planner([same_delta, same_delta])

    runner = GoalRunner(
        goal=goal,
        initial_graph=graph,
        gate=gate,
        planner=planner,
        budget=RunBudget(max_iterations=5, max_total_tokens=None),
    )
    result = runner.run([NodeResult(node_id="input", result=0)])

    assert result.terminated_reason == TerminatedReason.STUCK_REPEAT_DELTA
    assert len(result.history) == 2


def test_planner_failure():
    """GoalRunner returns PLANNER_FAILED when planner raises PlannerError."""
    graph = _build_two_node_graph()
    goal = Goal(description="produce something")
    gate = _make_never_gate()

    planner = MagicMock(spec=Planner)
    planner.propose_delta.side_effect = PlannerError("LLM failed")

    runner = GoalRunner(
        goal=goal,
        initial_graph=graph,
        gate=gate,
        planner=planner,
        budget=RunBudget(max_iterations=5, max_total_tokens=None),
    )
    result = runner.run([NodeResult(node_id="input", result=0)])

    assert result.terminated_reason == TerminatedReason.PLANNER_FAILED
    assert "LLM failed" in result.history[0].error
    assert len(result.history) == 1


def test_event_history_ordering():
    """History entries have sequential indices and one entry per iteration."""
    graph = _build_two_node_graph()
    goal = Goal(description="produce something")
    gate = _make_converging_gate(target=0, converge_on_iteration=3)

    # Unique deltas to avoid stuck detection
    deltas = [
        GraphDelta(ops=[RemoveEdge(source_id="a", target_id="b"), AddEdge(source_id="a", target_id="b")]),
        GraphDelta(
            ops=[
                AddEdge(source_id="a", target_id="b"),
                RemoveEdge(source_id="a", target_id="b"),
                AddEdge(source_id="a", target_id="b"),
            ]
        ),
        GraphDelta(
            ops=[
                RemoveEdge(source_id="a", target_id="b"),
                AddEdge(source_id="a", target_id="b"),
                RemoveEdge(source_id="a", target_id="b"),
                AddEdge(source_id="a", target_id="b"),
            ]
        ),
    ]
    planner = _make_mock_planner(deltas)

    runner = GoalRunner(
        goal=goal,
        initial_graph=graph,
        gate=gate,
        planner=planner,
        budget=RunBudget(max_iterations=5, max_total_tokens=None),
    )
    result = runner.run([NodeResult(node_id="input", result=0)])

    assert result.terminated_reason == TerminatedReason.ACHIEVED
    for idx, event in enumerate(result.history):
        assert event.index == idx


def test_compile_failure_continues():
    """GoalRunner continues with same graph when apply_delta raises ValueError."""
    graph = _build_two_node_graph()
    goal = Goal(description="produce something")

    call_count = {"n": 0}

    def varying_verdict(goal, output):
        call_count["n"] += 1
        return GateVerdict(achieved=call_count["n"] >= 3, reason=f"attempt {call_count['n']}")

    gate = ProgrammaticGate(predicate=varying_verdict)

    # First delta creates a cycle (will fail compile), second is valid
    cycle_delta = GraphDelta(ops=[AddEdge(source_id="b", target_id="a")])
    valid_delta = GraphDelta(ops=[RemoveEdge(source_id="a", target_id="b"), AddEdge(source_id="a", target_id="b")])
    planner = _make_mock_planner([cycle_delta, valid_delta])

    runner = GoalRunner(
        goal=goal,
        initial_graph=graph,
        gate=gate,
        planner=planner,
        budget=RunBudget(max_iterations=5, max_total_tokens=None),
    )
    result = runner.run([NodeResult(node_id="input", result=0)])

    assert result.terminated_reason == TerminatedReason.ACHIEVED
    # First iteration: compile failure (applied=False, error set)
    assert result.history[0].applied is False
    assert result.history[0].error is not None


def test_execution_failure():
    """GoalRunner returns EXECUTION_FAILED when graph.run raises."""

    class FailingNode(AbstractNode):
        def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
            raise RuntimeError("node exploded")

    a = FailingNode(id="a", name="fail_a")
    b = CounterNode(id="b", name="node_b", increment=1)
    graph = Graph()
    graph.add_edge(a, b)
    graph.compile()

    goal = Goal(description="produce something")
    gate = _make_never_gate()
    planner = MagicMock(spec=Planner)

    runner = GoalRunner(
        goal=goal,
        initial_graph=graph,
        gate=gate,
        planner=planner,
        budget=RunBudget(max_iterations=5, max_total_tokens=None),
    )
    result = runner.run([NodeResult(node_id="input", result=0)])

    assert result.terminated_reason == TerminatedReason.EXECUTION_FAILED
    assert "node exploded" in result.history[0].error


def test_no_exception_escapes():
    """GoalRunner.run never raises — all errors result in a GoalRunResult."""
    graph = _build_two_node_graph()
    goal = Goal(description="produce something")
    gate = _make_never_gate()

    planner = MagicMock(spec=Planner)
    planner.propose_delta.side_effect = PlannerError("catastrophic failure")

    runner = GoalRunner(
        goal=goal,
        initial_graph=graph,
        gate=gate,
        planner=planner,
        budget=RunBudget(max_iterations=5, max_total_tokens=None),
    )

    # Should not raise
    result = runner.run([NodeResult(node_id="input", result=0)])
    assert result.terminated_reason == TerminatedReason.PLANNER_FAILED
