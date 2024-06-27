from typing import Dict, Type, List

from pydantic import BaseModel, Field
import Levenshtein

from nagatoai_core.tool.abstract_tool import AbstractTool


class ToolNameMismatchError(Exception):
    """
    ToolNameMismatchError represents an error that occurs when the tool name does not match any registered tool.
    """

    def __init__(self, name: str, closest_matches: List[str]):
        """
        Initializes a new instance of the ToolNameMismatchError class.
        :param name: The name that did not match any registered tool.
        :param closest_matches: The closest tool names within the maximum distance.
        """
        super().__init__(
            f"Tool '{name}' not found in the registry. Did you mean one of the following: {closest_matches}?"
        )


class ToolNotFoundError(Exception):
    """
    ToolNotFoundError represents an error that occurs when the tool is not found in the registry.
    """

    def __init__(self, name: str):
        """
        Initializes a new instance of the ToolNotFoundError class.
        :param name: The name of the tool that was not found.
        """
        super().__init__(f"Tool '{name}' not found in the registry.")


class ToolRegistry(BaseModel):
    """
    ToolRegistry represents a registry of tools.
    """

    tools_map: Dict[str, Type[AbstractTool]] = Field(
        default_factory=dict,
        description="The map of tools in the registry.",
    )

    def __init__(self):
        """
        Initializes a new instance of the ToolRegistry class.
        """
        super().__init__()

    def get_closest_tool(self, name: str, max_distance=2) -> str:
        """
        Get the closest tool name to the given name.
        :param name: The name to match.
        :param max_distance: The maximum Levenshtein distance to consider.
        :return: The closest tool names within the maximum distance.
        """
        potential_matches = list(self.tools_map.keys())
        close_matches = []

        for match in potential_matches:
            distance = Levenshtein.distance(name, match)
            if distance <= max_distance:
                close_matches.append((match, distance))

        # Sort the matches by distance (closest first)
        close_matches.sort(key=lambda x: x[1])

        return close_matches

    def register_tool(self, tool: Type[AbstractTool]):
        """
        Registers a tool with the registry.
        :param tool: The tool to register.
        """
        tool_instance = tool()
        self.tools_map[tool_instance.name] = tool

    def get_tool(self, name: str) -> Type[AbstractTool]:
        """
        Gets the tool with the given name.
        :param name: The name of the tool to get.
        :return: The tool with the given name.
        """
        if name not in self.tools_map:
            closest_matches = self.get_closest_tool(name)
            if not closest_matches:
                raise ToolNotFoundError(name)

            raise ToolNameMismatchError(name, closest_matches)

        return self.tools_map[name]

    def get_all_tools(self) -> List[Type[AbstractTool]]:
        """
        Gets all the tools in the registry.
        :return: The list of tools in the registry.
        """
        return list(self.tools_map.values())
