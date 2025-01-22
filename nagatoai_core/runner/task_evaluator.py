# Standard Library
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

# Third Party
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn

# Nagato AI
# Company Libraries
from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.message import Exchange
from nagatoai_core.mission.task import Task, TaskResult


class TaskEvaluator(BaseModel, ABC):
    """
    TaskEvaluator is used to evaluate a given task
    """

    tracing_enabled: bool = False

    def __init__(self, **data):
        super().__init__(**data)

    @abstractmethod
    def evaluate(self, task: Task, agents: Dict[str, Agent], exchanges: List[Exchange]) -> TaskResult:
        """
        Evaluates the task
        """
        pass
