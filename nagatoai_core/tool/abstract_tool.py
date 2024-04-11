from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, Field

TOOL_METADATA_EXCLUDE_FROM_SCHEMA = "exclude_from_schema"


class AbstractTool(BaseModel, ABC):
    """
    AbstractTool represents the base class for all AI Tools.
    Any concrete AI Tool must inherit from this class and implement the run method.
    """

    name: str
    """The unique name of the tool."""

    description: str
    """The description of the tool, which indicates what the tool does. Include short examples if possible."""

    args_schema: Optional[Type[BaseModel]] = None
    """The schema for the arguments that the tool accepts."""

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the tool with the given arguments.
        :param args: The arguments to run the tool with.
        :param kwargs: The arguments to run the tool with.
        :return: The result of running the tool.
        """
        pass
