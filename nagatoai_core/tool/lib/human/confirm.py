from typing import Any, Type

from pydantic import BaseModel, Field
from rich.prompt import Confirm

from nagatoai_core.tool.abstract_tool import AbstractTool


class HumanConfirmInputConfig(BaseModel):
    """
    HumanConfirmInputConfig represents the configuration for the HumanConfirmInputTool.
    """

    message: str = Field(
        ...,
        description="The message to display to the user to confirm whether to proceed or not",
    )


class HumanConfirmInputTool(AbstractTool):
    """
    HumanConfirmInputTool represents a tool that prompts the user to confirm whether to proceed or not.
    """

    name: str = "human_confirm_input"
    description: str = (
        """Prompts the user to confirm whether to proceed or not. Returns a boolean value indicating the user's choice."""
    )
    args_schema: Type[BaseModel] = HumanConfirmInputConfig

    def _run(self, config: HumanConfirmInputConfig) -> Any:
        """
        Prompts the user to confirm whether to proceed or not.
        :param message: The message to display to the user to confirm whether to proceed or not.
        :return: A boolean value indicating the user's choice.
        """
        confirm = Confirm.ask("[bold yellow]" + config.message + "[/bold yellow]")

        return {
            "message": config.message,
            "confirmed": confirm,
        }
