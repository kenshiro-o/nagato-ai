from typing import Any, Type, Optional

from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

from nagatoai_core.tool.abstract_tool import AbstractTool


class TimeOffsetConfig(BaseModel):
    """
    TimeOffsetConfig represents the configuration for the TimeOffsetTool.
    """

    base_time: Optional[str] = Field(
        default=None,
        description="The reference time to apply the offset to. If not provided, the current time will be used.",
    )

    years: int = Field(
        default=0,
        description="The number of years offset to apply",
    )
    months: int = Field(
        default=0,
        description="The number of months offset to apply",
    )
    days: int = Field(
        default=0,
        description="The number of days offset to apply",
    )
    hours: int = Field(
        default=0,
        description="The number of hours offset to apply",
    )
    minutes: int = Field(
        default=0,
        description="The number of minutes offset to apply",
    )
    seconds: int = Field(
        default=0,
        description="The number of seconds offset to apply",
    )


class TimeOffsetTool(AbstractTool):
    """
    TimeOffsetTool represents a tool that applies an offset the reference time to obtain a time either in the future or a past.
    All times are in UTC.
    """

    name: str = "time_offset"
    description: str = """Applies an offset to a given datetime."""
    args_schema: Type[BaseModel] = TimeOffsetConfig

    def _run(self, config: TimeOffsetConfig) -> Any:
        """
        Applies an offset to a given datetime.
        :param config: The configuration for the tool.
        :return: The datetime in ISO format after applying the offset.
        """
        ref_dt = (
            datetime.utcnow() if not config.base_time else parse_date(config.base_time)
        )

        offset = timedelta(
            days=config.days,
            seconds=config.seconds,
            microseconds=0,
            milliseconds=0,
            minutes=config.minutes,
            hours=config.hours,
            weeks=0,
        )

        return (ref_dt + offset).isoformat()
