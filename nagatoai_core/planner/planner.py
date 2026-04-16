# Standard Library
import re
from typing import Dict, List, Literal, Optional, Type

# Third Party
import structlog
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.message import Exchange
from nagatoai_core.common.common import send_agent_request
from nagatoai_core.goal.gate import GateVerdict
from nagatoai_core.goal.goal import Goal
from nagatoai_core.graph.delta import AddEdge, AddNode, GraphDelta, RemoveEdge, RemoveNode
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.plan.xml_parser import XMLPlanParser
from nagatoai_core.graph.serializer import serialize_graph
from nagatoai_core.graph.types import NodeResult
from nagatoai_core.planner.prompts import PLANNER_PROMPT, PLANNER_REPLACE_PROMPT

logger = structlog.get_logger(__name__)

MAX_TRACE_CHARS = 8192


class PlannerError(Exception):
    """Raised when the planner cannot produce a usable delta after retries."""

    pass


class Planner(BaseModel):
    """
    Delta-focused planner that proposes minimal graph modifications to move closer to a goal.

    :param planner_agent: The agent used to generate delta proposals.
    :param agents: Agents available for new nodes in the delta.
    :param output_schemas: Output schemas available for new nodes.
    :param delta_mode: Whether to request incremental deltas or full graph replacements.
    :param max_parse_retries: Maximum number of retries on malformed XML.
    :param temperature: Temperature for the planner agent.
    :param max_tokens: Maximum tokens for the planner response.
    """

    planner_agent: Agent
    agents: Dict[str, Agent] = Field(default_factory=dict)
    output_schemas: Dict[str, Type[BaseModel]] = Field(default_factory=dict)
    delta_mode: Literal["incremental", "replace"] = "incremental"
    max_parse_retries: int = 1
    temperature: float = 0.4
    max_tokens: int = 6000

    class Config:
        arbitrary_types_allowed = True

    def propose_delta(
        self,
        goal: Goal,
        graph: Graph,
        execution_trace: List[NodeResult],
        gate_verdict: GateVerdict,
    ) -> tuple[GraphDelta, Optional[Exchange]]:
        """
        Propose a graph delta to move the graph closer to satisfying the goal.

        :param goal: The goal to achieve.
        :param graph: The current graph.
        :param execution_trace: The output from the last graph execution.
        :param gate_verdict: The verdict explaining why the goal is not yet achieved.
        :returns: A tuple of (GraphDelta, Exchange or None).
        :raises PlannerError: If the planner cannot produce a valid delta after retries.
        """
        if self.delta_mode == "replace":
            return self._propose_replace(goal, graph, execution_trace, gate_verdict)
        return self._propose_incremental(goal, graph, execution_trace, gate_verdict)

    def _propose_incremental(
        self,
        goal: Goal,
        graph: Graph,
        execution_trace: List[NodeResult],
        gate_verdict: GateVerdict,
    ) -> tuple[GraphDelta, Optional[Exchange]]:
        current_plan_xml = serialize_graph(graph)
        trace_str = _serialize_trace(execution_trace)

        prompt = PLANNER_PROMPT.format(
            goal_description=goal.description,
            current_plan_xml=current_plan_xml,
            execution_trace=trace_str,
            gate_verdict_reason=gate_verdict.reason,
        )

        last_error: Optional[str] = None
        for attempt in range(1 + self.max_parse_retries):
            if attempt > 0 and last_error:
                prompt = (
                    f"{prompt}\n\nYour previous response could not be parsed. Error: {last_error}\n"
                    "Please try again with valid XML."
                )

            logger.info("planner_request", attempt=attempt, delta_mode=self.delta_mode)

            exchange = send_agent_request(self.planner_agent, None, prompt, [], self.temperature, self.max_tokens)
            response_content = exchange.agent_response.content

            try:
                delta_xml = _extract_delta_xml(response_content)
                parser = XMLPlanParser()
                delta = parser.parse_delta(delta_xml, self.agents, self.output_schemas, graph)
                logger.info("planner_delta_parsed", op_count=len(delta.ops))
                return delta, exchange
            except Exception as e:
                last_error = str(e)
                logger.warning("planner_parse_failed", attempt=attempt, error=last_error)

        raise PlannerError(
            f"Planner failed to produce a valid delta after {1 + self.max_parse_retries} attempts: {last_error}"
        )

    def _propose_replace(
        self,
        goal: Goal,
        graph: Graph,
        execution_trace: List[NodeResult],
        gate_verdict: GateVerdict,
    ) -> tuple[GraphDelta, Optional[Exchange]]:
        current_plan_xml = serialize_graph(graph)
        trace_str = _serialize_trace(execution_trace)

        prompt = PLANNER_REPLACE_PROMPT.format(
            goal_description=goal.description,
            current_plan_xml=current_plan_xml,
            execution_trace=trace_str,
            gate_verdict_reason=gate_verdict.reason,
        )

        last_error: Optional[str] = None
        for attempt in range(1 + self.max_parse_retries):
            if attempt > 0 and last_error:
                prompt = (
                    f"{prompt}\n\nYour previous response could not be parsed. Error: {last_error}\n"
                    "Please try again with valid XML."
                )

            logger.info("planner_replace_request", attempt=attempt)

            exchange = send_agent_request(self.planner_agent, None, prompt, [], self.temperature, self.max_tokens)
            response_content = exchange.agent_response.content

            try:
                plan_xml = _extract_plan_xml(response_content)
                parser = XMLPlanParser()
                new_plan = parser.parse(plan_xml)
                delta = _synthesize_replace_delta(graph, new_plan.graph)
                logger.info("planner_replace_parsed", op_count=len(delta.ops))
                return delta, exchange
            except Exception as e:
                last_error = str(e)
                logger.warning("planner_replace_parse_failed", attempt=attempt, error=last_error)

        raise PlannerError(f"Planner (replace mode) failed after {1 + self.max_parse_retries} attempts: {last_error}")


def _extract_plan_xml(response: str) -> str:
    """
    Extract a full plan XML block from a planner response, stripping code fences if present.

    :param response: The raw response from the planner agent.
    :returns: The extracted plan XML string.
    :raises ValueError: If no plan XML is found.
    """
    text = response
    if "```xml" in text and "```" in text:
        start_idx = text.find("```xml") + 6
        end_idx = text.rfind("```")
        text = text[start_idx:end_idx].strip()
    elif "```" in text:
        start_idx = text.find("```") + 3
        end_idx = text.rfind("```")
        text = text[start_idx:end_idx].strip()

    match = re.search(r"<plan\b[^>]*>.*?</plan>", text, re.DOTALL)
    if match:
        return match.group(0)

    if text.strip().startswith("<plan") and text.strip().endswith("</plan>"):
        return text.strip()

    raise ValueError("No <plan> block found in planner response")


def _synthesize_replace_delta(old_graph: Graph, new_graph: Graph) -> GraphDelta:
    """
    Synthesize a GraphDelta that transforms old_graph into new_graph by removing all old
    nodes/edges and adding all new ones.

    :param old_graph: The current graph.
    :param new_graph: The replacement graph from the planner.
    :returns: A GraphDelta that replaces the old graph with the new one.
    """
    ops = []

    for source_id, targets in old_graph.adjacency_list.items():
        for target_id in targets:
            ops.append(RemoveEdge(source_id=source_id, target_id=target_id))

    for node_id in list(old_graph.node_map.keys()):
        ops.append(RemoveNode(node_id=node_id))

    for node_id, node in new_graph.node_map.items():
        ops.append(AddNode(node=node))

    for source_id, targets in new_graph.adjacency_list.items():
        for target_id in targets:
            ops.append(AddEdge(source_id=source_id, target_id=target_id))

    return GraphDelta(ops=ops)


def _extract_delta_xml(response: str) -> str:
    """
    Extract the <delta>...</delta> block from a planner response, stripping code fences if present.

    :param response: The raw response from the planner agent.
    :returns: The extracted delta XML string.
    :raises ValueError: If no <delta> block is found.
    """
    text = response
    if "```xml" in text and "```" in text:
        start_idx = text.find("```xml") + 6
        end_idx = text.rfind("```")
        text = text[start_idx:end_idx].strip()
    elif "```" in text:
        start_idx = text.find("```") + 3
        end_idx = text.rfind("```")
        text = text[start_idx:end_idx].strip()

    match = re.search(r"<delta\b[^>]*>.*?</delta>", text, re.DOTALL)
    if match:
        return match.group(0)

    if text.strip().startswith("<delta") and text.strip().endswith("</delta>"):
        return text.strip()

    raise ValueError("No <delta> block found in planner response")


def _serialize_trace(trace: List[NodeResult]) -> str:
    """
    Serialize an execution trace to a compact string, capped at MAX_TRACE_CHARS.

    :param trace: The list of node results to serialize.
    :returns: A string representation of the trace.
    """
    parts = []
    for r in trace:
        parts.append(f"[{r.node_id}] {r.result}")
    full = "\n".join(parts)
    if len(full) > MAX_TRACE_CHARS:
        return full[:MAX_TRACE_CHARS] + "\n... (truncated)"
    return full
