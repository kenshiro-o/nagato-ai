from typing import Any, Type

from pydantic import BaseModel, Field
from rich.console import Console

from nagatoai_core.tool.abstract_tool import AbstractTool


class HumanInputConfig(BaseModel):
    """
    HumanInputConfig represents the configuration for the HumanInputTool.
    """

    message: str = Field(
        ...,
        description="The message to display to the user to prompt for input",
    )


class HumanInputTool(AbstractTool):
    """
    HumanInputTool represents a tool that prompts the user to input a value.
    """

    name: str = "human_input"
    description: str = (
        """Prompts the user to input a value. Returns the value entered by the user."""
    )
    args_schema: Type[BaseModel] = HumanInputConfig

    def _run(self, config: HumanInputConfig) -> Any:
        """
        Prompts the user to input a value from the keyboard.
        :param config: The configuration for the tool.
        :return: The string value entered by the user.
        """
        console = Console()
        human_input = console.input("[bold yellow]" + config.message + "[/bold yellow]")

        return human_input
