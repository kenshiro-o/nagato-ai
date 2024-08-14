from typing import Dict, Any, Optional

from nagatoai_core.chain.chain import Link
from nagatoai_core.agent.agent import Agent
from nagatoai_core.prompt.template.prompt_template import PromptTemplate


class AgentLink(Link):
    """
    AgentLink represents a link that executes an agent.
    """

    agent: Agent
    temperature: float = 0.6
    max_tokens: int = 2000
    input_prompt: Optional[str] = None
    input_prompt_template: Optional[PromptTemplate] = None
    clear_memory_after_chat: bool = True
    # TODO - In the future add tools so that we can implement tool calls too

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def forward(self, input_data: Dict) -> Any:
        """
        Forward the data through the link
        :param input_data: The configuration object that will be used to as parameter to the agent prompt
        :return: The output data obtained after processing the input data through the link
        """
        final_prompt = f"{self.input_prompt}\n{input_data}"

        if self.input_prompt_template:
            final_prompt = self.input_prompt_template.generate_prompt(input_data)

        exchange = self.agent.chat(
            None,
            final_prompt,
            tools=[],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if self.clear_memory_after_chat:
            self.agent.clear_memory()

        return exchange.agent_response.content

    def category(self) -> str:
        """
        Get the category of the link
        :return: The category of the link
        """
        return "AGENT_LINK"
