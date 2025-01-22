# Standard Library
from typing import Any, Dict, List, Optional, Type, Union

# Nagato AI
# Company Libraries
from nagatoai_core.chain.chain import Link
from nagatoai_core.tool.abstract_tool import AbstractTool


class ToolLink(Link):
    """
    ToolLink represents a link that executes a tool.
    """

    partial_tool_args: Optional[Dict[str, Any]] = None
    tool: Type[AbstractTool]

    class Config:
        extra = "ignore"

    def forward(self, input_data: Dict) -> Any:
        """
        Forward the data through the link
        :param input_data: The configuration object that will be used to power the tool
        :return: The output data obtained after processing the input data through the link
        """
        tool_instance = self.tool()
        tool_args_schema = tool_instance.args_schema

        # Merge the partial tool args with the input data
        if self.partial_tool_args is not None:
            print(f"Merging partial tool args: {self.partial_tool_args} with input data: {input_data}")
            input_data = {
                **input_data,
                **self.partial_tool_args,
            }
            print(f"Merged input data: {input_data}")

        tool_params = tool_args_schema(**input_data)

        return tool_instance._run(tool_params)

    def category(self) -> str:
        """
        Get the category of the link
        :return: The category of the link
        """
        return "TOOL_LINK"
