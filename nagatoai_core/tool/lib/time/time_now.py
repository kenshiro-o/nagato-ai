# Standard Library
from datetime import UTC, datetime
from typing import Any, Type

# Third Party
from pydantic import BaseModel, Field

# Nagato AI
# Company Libraries
from nagatoai_core.tool.abstract_tool import AbstractTool


class TimeNowConfig(BaseModel):
    """
    TimeNowConfig represents the configuration for the TimeNowTool.
    """

    class Config:
        extra = "forbid"

    use_utc_timezone: bool = Field(
        True,
        description="Whether to use the UTC timezone.",
    )


class TimeNowTool(AbstractTool):
    """
    TimeNowTool represents a tool that returns the time now in UTC.
    """

    name: str = "time_now"
    description: str = """Returns the time now in UTC."""
    args_schema: Type[BaseModel] = TimeNowConfig

    def _run(self, config: TimeNowConfig) -> Any:
        """
        Returns the time now in UTC.
        :param config: The configuration for the tool.
        :return: The UTC time now in ISO
        """
        if config.use_utc_timezone:
            return datetime.now(UTC).isoformat()
        else:
            return datetime.now().isoformat()
