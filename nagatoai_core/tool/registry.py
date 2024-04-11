from typing import Dict, Type, List

from pydantic import BaseModel, Field

from nagatoai_core.tool.abstract_tool import AbstractTool


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
        return self.tools_map[name]

    def get_all_tools(self) -> List[Type[AbstractTool]]:
        """
        Gets all the tools in the registry.
        :return: The list of tools in the registry.
        """
        return list(self.tools_map.values())
