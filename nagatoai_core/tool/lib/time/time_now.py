from typing import Any, Type

from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from nagatoai_core.tool.abstract_tool import AbstractTool


class TimeNowConfig(BaseModel):
    """
    TimeNowConfig represents the configuration for the TimeNowTool.
    """

    pass


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
        return datetime.utcnow().isoformat()
