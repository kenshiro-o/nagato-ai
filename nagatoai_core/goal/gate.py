# Standard Library
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Literal

# Third Party
from bs4 import BeautifulSoup
from pydantic import BaseModel

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.common.common import send_agent_request
from nagatoai_core.goal.goal import Goal
from nagatoai_core.goal.prompts import LLM_JUDGE_PROMPT
from nagatoai_core.graph.types import NodeResult


class GateVerdict(BaseModel):
    """
    Represents the outcome of a gate evaluation.

    :param achieved: Whether the goal was achieved.
    :param reason: Explanation of the verdict.
    :param evidence: Optional supporting evidence.
    """

    achieved: bool
    reason: str
    evidence: Any = None


class AbstractGate(BaseModel, ABC):
    """
    Abstract base class for all gate types.
    """

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def evaluate(self, goal: Goal, graph_output: List[NodeResult]) -> GateVerdict:
        """
        Evaluates whether the goal has been achieved given the graph output.

        :param goal: The goal to evaluate against.
        :param graph_output: The list of node results from the graph execution.
        :return: A GateVerdict indicating whether the goal was achieved.
        """
        pass


class ProgrammaticGate(AbstractGate):
    """
    A gate that uses a callable predicate to evaluate the goal.

    :param predicate: A callable that takes a Goal and list of NodeResult and returns a GateVerdict.
    """

    predicate: Callable[[Goal, List[NodeResult]], GateVerdict]

    def evaluate(self, goal: Goal, graph_output: List[NodeResult]) -> GateVerdict:
        """
        Evaluates the goal using the provided predicate function.

        :param goal: The goal to evaluate against.
        :param graph_output: The list of node results from the graph execution.
        :return: A GateVerdict from the predicate.
        """
        return self.predicate(goal, graph_output)


class LLMJudgeGate(AbstractGate):
    """
    A gate that uses an LLM agent as a judge to evaluate goal achievement.

    :param judge_agent: The agent used to judge goal achievement.
    :param prompt_template: The prompt template for the judge. Defaults to LLM_JUDGE_PROMPT.
    :param temperature: The temperature for the judge agent. Defaults to 0.6.
    :param max_tokens: The maximum number of tokens for the judge response. Defaults to 2000.
    """

    judge_agent: Agent
    prompt_template: str = LLM_JUDGE_PROMPT
    temperature: float = 0.6
    max_tokens: int = 2000

    def evaluate(self, goal: Goal, graph_output: List[NodeResult]) -> GateVerdict:
        """
        Evaluates the goal by sending the graph output to the judge agent and parsing the XML response.

        :param goal: The goal to evaluate against.
        :param graph_output: The list of node results from the graph execution.
        :return: A GateVerdict based on the judge agent's assessment.
        """
        serialized = "\n".join(f"[{r.node_id}] {r.result}" for r in graph_output)
        prompt = self.prompt_template.format(goal_description=goal.description, graph_output=serialized)

        try:
            exchange = send_agent_request(self.judge_agent, None, prompt, [], self.temperature, self.max_tokens)
            soup = BeautifulSoup(exchange.agent_response.content, "html.parser")
            achieved_tag = soup.find("achieved")
            reason_tag = soup.find("reason")
            if achieved_tag is None:
                return GateVerdict(
                    achieved=False,
                    reason="LLM judge response missing <achieved> tag",
                    evidence=exchange.agent_response.content,
                )
            achieved = achieved_tag.get_text(strip=True).lower() == "true"
            reason = reason_tag.get_text(strip=True) if reason_tag else "No reason provided"
            return GateVerdict(achieved=achieved, reason=reason, evidence=exchange.agent_response.content)
        except Exception as e:
            return GateVerdict(achieved=False, reason=f"LLM judge gate evaluation failed: {e}")


class CompositeGate(AbstractGate):
    """
    A gate that combines multiple gates using a logical mode.

    :param gates: The list of gates to evaluate.
    :param mode: The logical mode for combining verdicts. 'all' requires all gates to pass, 'any' requires at least one.
    """

    gates: List[AbstractGate]
    mode: Literal["all", "any"] = "all"

    def evaluate(self, goal: Goal, graph_output: List[NodeResult]) -> GateVerdict:
        """
        Evaluates the goal by running all child gates and combining their verdicts.

        :param goal: The goal to evaluate against.
        :param graph_output: The list of node results from the graph execution.
        :return: A combined GateVerdict based on the mode.
        """
        verdicts = [gate.evaluate(goal, graph_output) for gate in self.gates]
        if self.mode == "all":
            achieved = all(v.achieved for v in verdicts)
        else:
            achieved = any(v.achieved for v in verdicts)
        reason = "; ".join(f"[gate {i}] {v.reason}" for i, v in enumerate(verdicts))
        return GateVerdict(achieved=achieved, reason=reason)
