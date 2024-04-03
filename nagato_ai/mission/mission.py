from typing import List
from enum import Enum

from pydantic import BaseModel

from .task import Task


class MissionStatus(Enum):
    """
    MissionStatus is an emum representing the status of a mission
    """

    PENDING = 0
    IN_PROGRESS = 1
    PAUSED = 2
    COMPLETED = 3
    FAILED = 4

    def __str__(self) -> str:
        return self.name


class Mission(BaseModel):
    """
    Mission represents a mission an agent or a group of agents must complete
    """

    # Your Mission class definition
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    problem_statement: str
    objective: str
    tasks: List[Task]
    status: MissionStatus = MissionStatus.PENDING

    def update_status(self, status: MissionStatus):
        """
        Updates the status of the mission
        :param status: The new status of the mission
        """
        self.status = status

    def __str__(self) -> str:
        return f"Mission: {self.objective} | Tasks={self.tasks}"
