from __future__ import annotations
from typing import Optional, List, Type
import traceback

from pydantic import BaseModel, Field

import logging

# Import necessary modules
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider
from nagatoai_core.mission.task import Task
from nagatoai_core.graph.types import NodeResult


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

            logging.info(f"*** Params to tool are {obj_params}")

            tool_params = tool_params_schema(**obj_params)

            logging.info(f"Schema generated is {tool_params}")

            res = tool_instance._run(tool_params)
            return [NodeResult(node_id=self.id, result=res, step=inputs[0].step + 1)]
        except Exception as e:
            print(f"An error occurred while trying to run the tool node: {e}")
            traceback.print_exc()
            return [NodeResult(node_id=self.id, result=None, error=str(e), step=inputs[0].step + 1)]
