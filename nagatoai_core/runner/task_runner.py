from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Type

from pydantic import BaseModel


from nagatoai_core.mission.task import Task
from nagatoai_core.agent.message import Exchange
from nagatoai_core.tool.registry import ToolRegistry
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider
from nagatoai_core.agent.agent import Agent
from nagatoai_core.runner.task_evaluator import TaskEvaluator
from nagatoai_core.memory.tool_run_cache import ToolRunCache


class TaskRunner(BaseModel, ABC):
    """
    TaskRunner is used to execute a given task
    """

    class Config:
        arbitrary_types_allowed = True

    previous_task: Optional[Task] = None
    current_task: Task
    agents: Dict[str, Agent]
    tool_registry: ToolRegistry
    agent_tool_providers: Dict[str, Type[AbstractToolProvider]]
    task_evaluator: TaskEvaluator
    tool_cache: ToolRunCache = ToolRunCache()

    @abstractmethod
    def run(self) -> List[Exchange]:
        """
        Runs the task
        """
        pass
