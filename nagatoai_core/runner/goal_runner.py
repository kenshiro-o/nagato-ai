# Standard Library
import hashlib
import time
from enum import Enum
from typing import List, Optional

# Third Party
import structlog
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.goal.gate import AbstractGate, GateVerdict
from nagatoai_core.goal.goal import Goal
from nagatoai_core.graph.delta import GraphDelta, apply_delta
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.serializer import serialize_graph
from nagatoai_core.graph.types import NodeResult
from nagatoai_core.planner.planner import Planner, PlannerError

logger = structlog.get_logger(__name__)


class RunBudget(BaseModel):
    """
    Budget constraints for a GoalRunner execution.

    :param max_iterations: Maximum number of execute-check-plan iterations.
    :param max_total_tokens: Maximum total tokens across all LLM calls (gate + planner). None for unlimited.
    :param max_wall_time_seconds: Maximum wall-clock time in seconds. None for unlimited.
    """

    max_iterations: int = 8
    max_total_tokens: Optional[int] = 200_000
    max_wall_time_seconds: Optional[float] = None


class TerminatedReason(str, Enum):
    """Reason the GoalRunner loop terminated."""

    ACHIEVED = "achieved"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STUCK_EMPTY_DELTA = "stuck_empty_delta"
    STUCK_REPEAT_VERDICT = "stuck_repeat_verdict"
    STUCK_REPEAT_DELTA = "stuck_repeat_delta"
    PLANNER_FAILED = "planner_failed"
    EXECUTION_FAILED = "execution_failed"


class IterationEvent(BaseModel):
    """
    Record of a single iteration through the GoalRunner loop.

    :param index: The iteration index (0-based).
    :param graph_before_xml: Serialized XML of the graph before this iteration's delta.
    :param graph_output: Results from graph execution.
    :param verdict: The gate's verdict on the output.
    :param proposed_delta: The delta proposed by the planner, if any.
    :param applied: Whether the delta was successfully applied.
    :param error: Error message if delta application or planning failed.
    """

    index: int
    graph_before_xml: str
    graph_output: List[NodeResult]
    verdict: GateVerdict
    proposed_delta: Optional[GraphDelta] = None
    applied: bool = False
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class GoalRunResult(BaseModel):
    """
    Result of a GoalRunner execution.

    :param terminated_reason: Why the loop ended.
    :param final_output: Output from the last graph execution.
    :param final_graph: The graph at termination.
    :param history: List of iteration events.
    """

    terminated_reason: TerminatedReason
    final_output: List[NodeResult]
    final_graph: Graph
    history: List[IterationEvent]

    class Config:
        arbitrary_types_allowed = True


class GoalRunner(BaseModel):
    """
    Outer loop that iteratively executes a graph, checks a gate, and applies planner deltas
    until the goal is achieved, budget is exhausted, or the loop is stuck.

    :param goal: The goal to achieve.
    :param initial_graph: The starting graph.
    :param gate: The gate that checks goal achievement.
    :param planner: The planner that proposes graph deltas.
    :param budget: Budget constraints for the run.
    """

    goal: Goal
    initial_graph: Graph
    gate: AbstractGate
    planner: Planner
    budget: RunBudget = Field(default_factory=RunBudget)
    tracing_enabled: bool = False

    class Config:
        arbitrary_types_allowed = True

    def run(self, inputs: List[NodeResult]) -> GoalRunResult:
        """
        Execute the goal-runner loop.

        :param inputs: Initial inputs to the graph.
        :returns: A GoalRunResult describing how the run terminated.
        """
        history: List[IterationEvent] = []
        graph = self.initial_graph
        output: List[NodeResult] = []
        previous_verdict_reason: Optional[str] = None
        previous_delta_sig: Optional[str] = None
        start_time = time.time()
        tokens_used = 0

        for i in range(self.budget.max_iterations):
            graph_xml = serialize_graph(graph)

            # 1. Execute the graph
            try:
                output = graph.run(inputs)
            except Exception as e:
                logger.error("graph_execution_failed", index=i, error=str(e))
                history.append(
                    IterationEvent(
                        index=i,
                        graph_before_xml=graph_xml,
                        graph_output=[],
                        verdict=GateVerdict(achieved=False, reason=f"Graph execution failed: {e}"),
                        applied=False,
                        error=str(e),
                    )
                )
                return GoalRunResult(
                    terminated_reason=TerminatedReason.EXECUTION_FAILED,
                    final_output=output,
                    final_graph=graph,
                    history=history,
                )

            # 2. Check the gate
            verdict = self.gate.evaluate(self.goal, output)
            logger.info("goal_iteration", index=i, achieved=verdict.achieved, reason=verdict.reason)

            if verdict.achieved:
                history.append(
                    IterationEvent(
                        index=i,
                        graph_before_xml=graph_xml,
                        graph_output=output,
                        verdict=verdict,
                        proposed_delta=None,
                        applied=False,
                    )
                )
                return GoalRunResult(
                    terminated_reason=TerminatedReason.ACHIEVED,
                    final_output=output,
                    final_graph=graph,
                    history=history,
                )

            # 3. Budget checks
            if self._wall_time_exhausted(start_time):
                history.append(
                    IterationEvent(
                        index=i,
                        graph_before_xml=graph_xml,
                        graph_output=output,
                        verdict=verdict,
                        applied=False,
                    )
                )
                return GoalRunResult(
                    terminated_reason=TerminatedReason.BUDGET_EXHAUSTED,
                    final_output=output,
                    final_graph=graph,
                    history=history,
                )

            if self._tokens_exhausted(tokens_used):
                history.append(
                    IterationEvent(
                        index=i,
                        graph_before_xml=graph_xml,
                        graph_output=output,
                        verdict=verdict,
                        applied=False,
                    )
                )
                return GoalRunResult(
                    terminated_reason=TerminatedReason.BUDGET_EXHAUSTED,
                    final_output=output,
                    final_graph=graph,
                    history=history,
                )

            # 4. Ask planner for delta
            try:
                delta, exchange = self.planner.propose_delta(self.goal, graph, output, verdict)
            except PlannerError as e:
                history.append(
                    IterationEvent(
                        index=i,
                        graph_before_xml=graph_xml,
                        graph_output=output,
                        verdict=verdict,
                        applied=False,
                        error=str(e),
                    )
                )
                return GoalRunResult(
                    terminated_reason=TerminatedReason.PLANNER_FAILED,
                    final_output=output,
                    final_graph=graph,
                    history=history,
                )

            if exchange and exchange.token_stats_and_params:
                tokens_used += (
                    exchange.token_stats_and_params.input_tokens_used
                    + exchange.token_stats_and_params.output_tokens_used
                )

            # 5. Stuck detection
            if not delta.ops:
                history.append(
                    IterationEvent(
                        index=i,
                        graph_before_xml=graph_xml,
                        graph_output=output,
                        verdict=verdict,
                        proposed_delta=delta,
                        applied=False,
                    )
                )
                return GoalRunResult(
                    terminated_reason=TerminatedReason.STUCK_EMPTY_DELTA,
                    final_output=output,
                    final_graph=graph,
                    history=history,
                )

            delta_sig = _delta_signature(delta)

            if verdict.reason == previous_verdict_reason:
                history.append(
                    IterationEvent(
                        index=i,
                        graph_before_xml=graph_xml,
                        graph_output=output,
                        verdict=verdict,
                        proposed_delta=delta,
                        applied=False,
                    )
                )
                return GoalRunResult(
                    terminated_reason=TerminatedReason.STUCK_REPEAT_VERDICT,
                    final_output=output,
                    final_graph=graph,
                    history=history,
                )

            if delta_sig == previous_delta_sig:
                history.append(
                    IterationEvent(
                        index=i,
                        graph_before_xml=graph_xml,
                        graph_output=output,
                        verdict=verdict,
                        proposed_delta=delta,
                        applied=False,
                    )
                )
                return GoalRunResult(
                    terminated_reason=TerminatedReason.STUCK_REPEAT_DELTA,
                    final_output=output,
                    final_graph=graph,
                    history=history,
                )

            # 6. Apply delta
            try:
                new_graph = apply_delta(graph, delta)
            except ValueError as e:
                logger.warning("delta_compile_failed", index=i, error=str(e))
                history.append(
                    IterationEvent(
                        index=i,
                        graph_before_xml=graph_xml,
                        graph_output=output,
                        verdict=verdict,
                        proposed_delta=delta,
                        applied=False,
                        error=str(e),
                    )
                )
                previous_verdict_reason = f"compile_failed: {e}"
                previous_delta_sig = delta_sig
                continue

            history.append(
                IterationEvent(
                    index=i,
                    graph_before_xml=graph_xml,
                    graph_output=output,
                    verdict=verdict,
                    proposed_delta=delta,
                    applied=True,
                )
            )

            graph = new_graph
            previous_verdict_reason = verdict.reason
            previous_delta_sig = delta_sig

        # Fell out of loop — budget exhausted by iteration count
        return GoalRunResult(
            terminated_reason=TerminatedReason.BUDGET_EXHAUSTED,
            final_output=output,
            final_graph=graph,
            history=history,
        )

    def _wall_time_exhausted(self, start_time: float) -> bool:
        if self.budget.max_wall_time_seconds is None:
            return False
        return (time.time() - start_time) > self.budget.max_wall_time_seconds

    def _tokens_exhausted(self, tokens_used: int) -> bool:
        if self.budget.max_total_tokens is None:
            return False
        return tokens_used >= self.budget.max_total_tokens


def _delta_signature(delta: GraphDelta) -> str:
    """
    Compute a hash signature of a delta for stuck detection.

    :param delta: The delta to hash.
    :returns: A hex SHA-1 digest.
    """
    return hashlib.sha1(delta.model_dump_json().encode()).hexdigest()
