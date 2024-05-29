import json
from typing import List, Dict, Optional

from pydantic import BaseModel

from nagatoai_core.agent.message import ToolRun


class ToolRunCache(BaseModel):
    """
    ToolRunCache represents a cache for tool run
    """

    cache: Dict[str, ToolRun] = {}
    runs: List[ToolRun] = []

    def get_last_tool_run(self) -> ToolRun:
        """
        Gets the last tool run from the cache
        :return: The last tool run from the cache
        """
        return self.runs[-1]

    def _get_tool_run_key(self, tool_name: str, parameters: Dict) -> str:
        """
        Gets the key for a tool run
        :param tool_name: The name of the tool
        :param parameters: The parameters of the tool
        :return: The key for the tool run
        """
        parameters_json = json.dumps(parameters, sort_keys=True, default=str)
        return f"{tool_name}_{hash(parameters_json)}"

    def get_tool_run(self, tool_name: str, parameters: Dict) -> Optional[ToolRun]:
        key = self._get_tool_run_key(tool_name, parameters)
        return self.cache.get(key)

    def add_tool_run(self, tool_run: ToolRun):
        """
        Adds a tool run to the cache
        :param tool_run: The tool run to add to the cache
        """
        self.runs.append(tool_run)

        key = f"{tool_run.call.name}_{self._get_tool_run_key(tool_run.call.name, tool_run.call.parameters)}"
        self.cache[key] = tool_run
