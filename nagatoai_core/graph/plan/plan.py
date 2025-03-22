from typing import Dict, Type

from pydantic import BaseModel, ConfigDict

from nagatoai_core.agent.agent import Agent
from nagatoai_core.graph.graph import Graph


class Plan(BaseModel):
    """
    Represents a plan with agents and output schemas.

    A plan defines the agents that will be used in a graph along with
    their configurations and output schemas for standardized data structures.
    """

    # Add model config to allow arbitrary fields
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agents: Dict[str, Agent]
    output_schemas: Dict[str, Type[BaseModel]]
    graph: Graph

    def get_agent(self, name: str) -> Agent:
        """
        Get an agent by name.

        Args:
            name: The name of the agent

        Returns:
            The Agent object

        Raises:
            KeyError: If the agent is not found
        """
        if name not in self.agents:
            raise KeyError(f"Agent '{name}' not found in plan")
        return self.agents[name]

    def get_output_schema(self, name: str) -> Type[BaseModel]:
        """
        Get an output schema model by name.

        Args:
            name: The name of the output schema

        Returns:
            The Pydantic model class

        Raises:
            KeyError: If the schema is not found
        """
        if name not in self.output_schemas:
            raise KeyError(f"Output schema '{name}' not found in plan")
        return self.output_schemas[name]
