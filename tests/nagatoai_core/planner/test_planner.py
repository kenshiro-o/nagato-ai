# Standard Library
from typing import List
from unittest.mock import patch

# Third Party
import pytest

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.message import Exchange, Message
from nagatoai_core.goal.gate import GateVerdict
from nagatoai_core.goal.goal import Goal
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.types import NodeResult
from nagatoai_core.planner.planner import Planner, PlannerError, _extract_delta_xml


class StubAgent(Agent):
    """A minimal concrete Agent for testing."""

    def __init__(self):
        super().__init__(model="test-model", role="tester", role_description="test agent", nickname="stub")

    def chat(self, task, prompt, tools, temperature, max_tokens, target_output_schema=None):
        return Exchange(
            user_msg=Message(role="user", content=prompt),
            agent_response=Message(role="assistant", content=""),
        )

    def send_tool_run_results(self, task, tool_results, tools, temperature, max_tokens):
        return Exchange(
            user_msg=Message(role="user", content=""),
            agent_response=Message(role="assistant", content=""),
        )

    def clear_memory(self):
        pass

    @property
    def maker(self):
        return "test"

    @property
    def history(self):
        return []

    @property
    def family(self):
        return "test"


class IncrementNode(AbstractNode):
    increment_by: int = 1

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        return [NodeResult(node_id=self.id, result=inp.result + self.increment_by) for inp in inputs]


def _build_simple_graph():
    """Build a two-node graph: a -> b."""
    a = IncrementNode(id="a", name="node_a", increment_by=1)
    b = IncrementNode(id="b", name="node_b", increment_by=2)
    g = Graph()
    g.add_edge(a, b)
    g.compile()
    return g


def _make_exchange(content: str):
    """Create a mock Exchange with the given agent response content."""
    # Standard Library
    from datetime import datetime

    # Nagato AI
    from nagatoai_core.agent.message import Sender, TokenStatsAndParams

    now = datetime.now()
    return Exchange(
        chat_history=[],
        user_msg=Message(sender=Sender.USER, content="test prompt", created_at=now),
        agent_response=Message(sender=Sender.AGENT, content=content, created_at=now),
        token_stats_and_params=TokenStatsAndParams(
            input_tokens_used=100, output_tokens_used=50, max_tokens=6000, temperature=0.4
        ),
    )


VALID_DELTA_XML = """<delta>
  <remove_edge source="a" target="b"/>
  <add_edge source="a" target="b"/>
</delta>"""

EMPTY_DELTA_XML = "<delta></delta>"


def test_planner_happy_path():
    """Planner returns a valid delta on the first attempt."""
    graph = _build_simple_graph()
    goal = Goal(description="produce 42")
    verdict = GateVerdict(achieved=False, reason="output was 3, not 42")
    trace = [NodeResult(node_id="b", result=3)]

    stub_agent = StubAgent()
    planner = Planner(planner_agent=stub_agent)

    with patch("nagatoai_core.planner.planner.send_agent_request") as mock_send:
        mock_send.return_value = _make_exchange(VALID_DELTA_XML)
        delta, exchange = planner.propose_delta(goal, graph, trace, verdict)

    assert len(delta.ops) == 2
    mock_send.assert_called_once()


def test_planner_empty_delta():
    """Planner returns an empty delta when no changes needed."""
    graph = _build_simple_graph()
    goal = Goal(description="produce 42")
    verdict = GateVerdict(achieved=False, reason="no progress possible")
    trace = [NodeResult(node_id="b", result=3)]

    stub_agent = StubAgent()
    planner = Planner(planner_agent=stub_agent)

    with patch("nagatoai_core.planner.planner.send_agent_request") as mock_send:
        mock_send.return_value = _make_exchange(EMPTY_DELTA_XML)
        delta, exchange = planner.propose_delta(goal, graph, trace, verdict)

    assert len(delta.ops) == 0


def test_planner_parse_retry():
    """Planner retries when first response is malformed, second is valid."""
    graph = _build_simple_graph()
    goal = Goal(description="produce 42")
    verdict = GateVerdict(achieved=False, reason="output was 3")
    trace = [NodeResult(node_id="b", result=3)]

    stub_agent = StubAgent()
    planner = Planner(planner_agent=stub_agent, max_parse_retries=1)

    with patch("nagatoai_core.planner.planner.send_agent_request") as mock_send:
        mock_send.side_effect = [
            _make_exchange("This is not XML at all"),
            _make_exchange(EMPTY_DELTA_XML),
        ]
        delta, exchange = planner.propose_delta(goal, graph, trace, verdict)

    assert mock_send.call_count == 2
    assert len(delta.ops) == 0


def test_planner_retry_exhaustion():
    """Planner raises PlannerError after all retries are exhausted."""
    graph = _build_simple_graph()
    goal = Goal(description="produce 42")
    verdict = GateVerdict(achieved=False, reason="output was 3")
    trace = [NodeResult(node_id="b", result=3)]

    stub_agent = StubAgent()
    planner = Planner(planner_agent=stub_agent, max_parse_retries=1)

    with patch("nagatoai_core.planner.planner.send_agent_request") as mock_send:
        mock_send.return_value = _make_exchange("garbage response")
        with pytest.raises(PlannerError, match="Planner failed"):
            planner.propose_delta(goal, graph, trace, verdict)

    assert mock_send.call_count == 2


def test_planner_code_fenced_response():
    """Planner handles response wrapped in code fences."""
    graph = _build_simple_graph()
    goal = Goal(description="produce 42")
    verdict = GateVerdict(achieved=False, reason="output was 3")
    trace = [NodeResult(node_id="b", result=3)]

    stub_agent = StubAgent()
    planner = Planner(planner_agent=stub_agent)

    fenced_response = f"Here is my delta:\n```xml\n{EMPTY_DELTA_XML}\n```\n"

    with patch("nagatoai_core.planner.planner.send_agent_request") as mock_send:
        mock_send.return_value = _make_exchange(fenced_response)
        delta, exchange = planner.propose_delta(goal, graph, trace, verdict)

    assert len(delta.ops) == 0


def test_extract_delta_xml_direct():
    """_extract_delta_xml extracts delta from mixed content."""
    content = "Some preamble text\n<delta>\n  <remove_node id='x'/>\n</delta>\nSome epilogue"
    result = _extract_delta_xml(content)
    assert "<delta>" in result
    assert "<remove_node" in result


def test_extract_delta_xml_no_delta():
    """_extract_delta_xml raises ValueError when no delta block found."""
    with pytest.raises(ValueError, match="No <delta> block found"):
        _extract_delta_xml("This has no delta at all")


def test_planner_replace_mode():
    """Planner in replace mode parses a full <plan> and synthesizes a replacement delta."""
    graph = _build_simple_graph()
    goal = Goal(description="produce 42")
    verdict = GateVerdict(achieved=False, reason="output was 3")
    trace = [NodeResult(node_id="b", result=3)]

    stub_agent = StubAgent()
    planner = Planner(planner_agent=stub_agent, agents={"planner": stub_agent}, delta_mode="replace")

    # A minimal valid plan XML that the XMLPlanParser can parse.
    # The parser requires <model>, <role>, <role_description>, <nickname> as child elements of <agent>.
    # Using gpt-4o since create_agent needs a recognized model name.
    plan_xml = """<?xml version="1.0" encoding="UTF-8"?>
<plan xmlns="http://nagatoai.com/schema/plan">
  <agents>
    <agent name="planner">
      <model>gpt-4o</model>
      <role>tester</role>
      <role_description>test agent</role_description>
      <nickname>stub</nickname>
    </agent>
  </agents>
  <nodes>
    <agent_node id="new_a" name="new_node_a">
      <agent name="planner"/>
      <prompt>Produce the number 42</prompt>
      <temperature>0.3</temperature>
      <max_tokens>100</max_tokens>
    </agent_node>
    <agent_node id="new_b" name="new_node_b">
      <agent name="planner"/>
      <prompt>Format the result</prompt>
      <temperature>0.3</temperature>
      <max_tokens>100</max_tokens>
    </agent_node>
  </nodes>
  <graph>
    <edges>
      <edge from="new_a" to="new_b"/>
    </edges>
  </graph>
</plan>"""

    with patch("nagatoai_core.planner.planner.send_agent_request") as mock_send:
        mock_send.return_value = _make_exchange(plan_xml)
        delta, exchange = planner.propose_delta(goal, graph, trace, verdict)

    # Delta should remove old nodes/edges and add new ones
    assert len(delta.ops) > 0
    op_kinds = [op.kind for op in delta.ops]
    assert "remove_edge" in op_kinds
    assert "remove_node" in op_kinds
    assert "add_node" in op_kinds
    assert "add_edge" in op_kinds
