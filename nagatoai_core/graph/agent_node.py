from __future__ import annotations

# Standard Library
import string
from typing import List, Optional, Type, Union

# Third Party
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.types import NodeResult
from nagatoai_core.mission.task import Task
from nagatoai_core.prompt.template.prompt_template import PromptTemplate
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider


class AgentNode(AbstractNode):
    """
    AgentNode is a node that executes an LLM Agent.

    It wraps an Agent instance and delegates execution to the agent's chat method.
    The node can store default parameters (such as task, prompt, tools, etc.) which can
    be overridden by keyword arguments during execution.
    """

    agent: Agent
    task: Optional[Task] = None
    prompt_template: Optional[PromptTemplate] = None
    tools: Optional[List[AbstractToolProvider]] = None
    temperature: float = 0.7
    max_tokens: int = 150
    output_schema: Optional[Type[BaseModel]] = None

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Executes the Agent's chat method.

        Parameters can be provided via the node's attributes or overridden by kwargs.
        The agent's chat method is expected to generate and return an Exchange object.
        """

        prompt = ""
        if self.prompt_template:
            # Build the prompt data dictionary - we expect the placeholders to be of the form "result_input[i]"
            prompt_data = {"inputs": inputs}

            self.logger.debug(f"Prompt data", prompt_data=prompt_data)

            prompt = self.prompt_template.generate_prompt(prompt_data)

        self.logger.debug(f"Full prompt", prompt=prompt)

        exchange = self.agent.chat(
            task=self.task,
            prompt=prompt,
            tools=self.tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            target_output_schema=self.output_schema,
        )

        return [NodeResult(node_id=self.id, result=exchange.agent_response.content, step=inputs[0].step + 1)]
