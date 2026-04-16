# Standard Library
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.message import Exchange, Message, Sender, TokenStatsAndParams
from nagatoai_core.goal.gate import CompositeGate, GateVerdict, LLMJudgeGate, ProgrammaticGate
from nagatoai_core.goal.goal import Goal
from nagatoai_core.goal.prompts import LLM_JUDGE_PROMPT
from nagatoai_core.graph.types import NodeResult


@pytest.fixture
def sample_goal():
    return Goal(description="Summarize the document into 3 bullet points.")


@pytest.fixture
def sample_graph_output():
    return [
        NodeResult(node_id="node_1", result="Bullet 1: Key finding A"),
        NodeResult(node_id="node_2", result="Bullet 2: Key finding B"),
        NodeResult(node_id="node_3", result="Bullet 3: Key finding C"),
    ]


def _make_exchange(content: str) -> Exchange:
    now = datetime.now(timezone.utc)
    return Exchange(
        chat_history=[],
        user_msg=Message(sender=Sender.USER, content="test prompt", created_at=now),
        agent_response=Message(sender=Sender.AGENT, content=content, created_at=now),
        token_stats_and_params=TokenStatsAndParams(
            input_tokens_used=10, output_tokens_used=20, max_tokens=2000, temperature=0.6
        ),
    )


def test_programmatic_gate_pass(sample_goal, sample_graph_output):
    def predicate(goal, graph_output):
        return GateVerdict(achieved=True, reason="All checks passed.")

    gate = ProgrammaticGate(predicate=predicate)
    verdict = gate.evaluate(sample_goal, sample_graph_output)
    assert verdict.achieved is True
    assert verdict.reason == "All checks passed."


def test_programmatic_gate_fail(sample_goal, sample_graph_output):
    def predicate(goal, graph_output):
        return GateVerdict(achieved=False, reason="Output was incomplete.")

    gate = ProgrammaticGate(predicate=predicate)
    verdict = gate.evaluate(sample_goal, sample_graph_output)
    assert verdict.achieved is False
    assert verdict.reason == "Output was incomplete."


@patch("nagatoai_core.goal.gate.send_agent_request")
def test_llm_judge_gate_achieved(mock_send, sample_goal, sample_graph_output):
    xml_response = "<verdict><achieved>true</achieved><reason>Goal fully met with 3 bullet points.</reason></verdict>"
    mock_send.return_value = _make_exchange(xml_response)

    judge_agent = MagicMock(spec=Agent)
    gate = LLMJudgeGate.model_construct(
        judge_agent=judge_agent, prompt_template=LLM_JUDGE_PROMPT, temperature=0.6, max_tokens=2000
    )
    verdict = gate.evaluate(sample_goal, sample_graph_output)

    assert verdict.achieved is True
    assert "3 bullet points" in verdict.reason
    assert verdict.evidence == xml_response
    mock_send.assert_called_once()


@patch("nagatoai_core.goal.gate.send_agent_request")
def test_llm_judge_gate_not_achieved(mock_send, sample_goal, sample_graph_output):
    xml_response = "<verdict><achieved>false</achieved><reason>Only 2 bullet points provided.</reason></verdict>"
    mock_send.return_value = _make_exchange(xml_response)

    judge_agent = MagicMock(spec=Agent)
    gate = LLMJudgeGate.model_construct(
        judge_agent=judge_agent, prompt_template=LLM_JUDGE_PROMPT, temperature=0.6, max_tokens=2000
    )
    verdict = gate.evaluate(sample_goal, sample_graph_output)

    assert verdict.achieved is False
    assert "2 bullet points" in verdict.reason
    mock_send.assert_called_once()


@patch("nagatoai_core.goal.gate.send_agent_request")
def test_llm_judge_gate_malformed_xml(mock_send, sample_goal, sample_graph_output):
    mock_send.return_value = _make_exchange("This is not valid XML at all, just garbage text.")

    judge_agent = MagicMock(spec=Agent)
    gate = LLMJudgeGate.model_construct(
        judge_agent=judge_agent, prompt_template=LLM_JUDGE_PROMPT, temperature=0.6, max_tokens=2000
    )
    verdict = gate.evaluate(sample_goal, sample_graph_output)

    assert verdict.achieved is False
    assert "missing" in verdict.reason.lower() or "failed" in verdict.reason.lower()


def test_composite_gate_all_pass(sample_goal, sample_graph_output):
    gate1 = ProgrammaticGate(predicate=lambda g, o: GateVerdict(achieved=True, reason="Gate 0 passed."))
    gate2 = ProgrammaticGate(predicate=lambda g, o: GateVerdict(achieved=True, reason="Gate 1 passed."))

    composite = CompositeGate(gates=[gate1, gate2], mode="all")
    verdict = composite.evaluate(sample_goal, sample_graph_output)

    assert verdict.achieved is True
    assert "[gate 0]" in verdict.reason
    assert "[gate 1]" in verdict.reason


def test_composite_gate_all_one_fails(sample_goal, sample_graph_output):
    gate1 = ProgrammaticGate(predicate=lambda g, o: GateVerdict(achieved=True, reason="Gate 0 passed."))
    gate2 = ProgrammaticGate(predicate=lambda g, o: GateVerdict(achieved=False, reason="Gate 1 failed."))

    composite = CompositeGate(gates=[gate1, gate2], mode="all")
    verdict = composite.evaluate(sample_goal, sample_graph_output)

    assert verdict.achieved is False


def test_composite_gate_any_one_passes(sample_goal, sample_graph_output):
    gate1 = ProgrammaticGate(predicate=lambda g, o: GateVerdict(achieved=False, reason="Gate 0 failed."))
    gate2 = ProgrammaticGate(predicate=lambda g, o: GateVerdict(achieved=True, reason="Gate 1 passed."))

    composite = CompositeGate(gates=[gate1, gate2], mode="any")
    verdict = composite.evaluate(sample_goal, sample_graph_output)

    assert verdict.achieved is True


def test_composite_gate_any_all_fail(sample_goal, sample_graph_output):
    gate1 = ProgrammaticGate(predicate=lambda g, o: GateVerdict(achieved=False, reason="Gate 0 failed."))
    gate2 = ProgrammaticGate(predicate=lambda g, o: GateVerdict(achieved=False, reason="Gate 1 failed."))

    composite = CompositeGate(gates=[gate1, gate2], mode="any")
    verdict = composite.evaluate(sample_goal, sample_graph_output)

    assert verdict.achieved is False
    assert "[gate 0]" in verdict.reason
    assert "[gate 1]" in verdict.reason
