from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Type

from pydantic import BaseModel
from rich.progress import Progress, SpinnerColumn
from rich.console import Console
from rich.panel import Panel


from nagatoai_core.mission.task import Task, TaskResult
from nagatoai_core.agent.message import Exchange
from nagatoai_core.agent.agent import Agent


class TaskEvaluator(BaseModel, ABC):
    """
    TaskEvaluator is used to evaluate a given task
    """

    def __init__(self, **data):
        super().__init__(**data)

    @abstractmethod
    def evaluate(
        self, task: Task, agents: Dict[str, Agent], exchanges: List[Exchange]
    ) -> TaskResult:
        """
        Evaluates the task
        """
        pass
