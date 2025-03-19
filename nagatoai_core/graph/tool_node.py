from __future__ import annotations

# Standard Library
import traceback
from typing import List

# Nagato AI
# Import necessary modules
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.types import NodeResult
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider


class ToolNode(AbstractNode):
    """
    A node that encapsulates a tool within a graph-based system.

    It delegates execution to the provided tool's process method.
    The node can store default parameters (such as inputs, context, and logs)
    which can be overridden by keyword arguments during execution.
    """

    # Tool to be used by this node
    tool_provider: AbstractToolProvider

    # TODO - in the future consider
    # - partial parameters for the tool, which will not be dynamically generated

    def execute(self, inputs: List[NodeResult]) -> NodeResult:
        """
        Executes the tool's process method.

        Parameters can be provided via the node's attributes or overridden by kwargs.
        The tool's process method is expected to generate and return a result object.
        """
        try:
            # TODO - Revisit this logic so that we could instantiate a new tool everyime
            tool_instance = self.tool_provider.tool
            tool_params_schema = tool_instance.args_schema

            obj_params = {}
            for inp in inputs:
                if isinstance(inp.result, dict):
                    obj_params.update(inp.result)

            self.logger.info(f"Tool params created from inputs", tool_params=obj_params)

            # Pick the first input if there are no params and check whether it is the same type as the tool params schema
            if len(obj_params) == 0 and isinstance(inputs[0].result, tool_params_schema):
                tool_params = inputs[0].result
            else:
                tool_params = tool_params_schema(**obj_params)

            self.logger.info(f"Tool params created from inputs and schema", tool_params=tool_params)

            res = tool_instance._run(tool_params)
            return [NodeResult(node_id=self.id, result=res, step=inputs[0].step + 1)]
        except Exception as e:
            self.logger.error(f"An error occurred while trying to run the tool node: {e}")
            traceback.print_exc()
            return [NodeResult(node_id=self.id, result=None, error=e, step=inputs[0].step + 1)]
