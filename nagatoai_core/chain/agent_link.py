from typing import Dict, Any

from nagatoai_core.chain.chain import Link
from nagatoai_core.agent.agent import Agent


class AgentLink(Link):
    """
    AgentLink represents a link that executes an agent.
    """

    agent: Agent
    temperature: float = 0.6
    max_tokens: int = 2000
    input_prompt: str
    # TODO - In the future add tools so that we can implement tool calls too

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def forward(self, input_data: Dict) -> Any:
        """
        Forward the data through the link
        :param input_data: The configuration object that will be used to power the tool
        :return: The output data obtained after processing the input data through the link
        """
        final_prompt = f"{self.input_prompt}\n{input_data}"
        exchange = self.agent.chat(
            final_prompt,
            tools=[],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return exchange.agent_response.content

    def category(self) -> str:
        """
        Get the category of the link
        :return: The category of the link
        """
        return "AGENT_LINK"
