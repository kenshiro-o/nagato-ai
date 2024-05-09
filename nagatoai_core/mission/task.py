from enum import Enum
from typing import Union, Optional
from datetime import datetime
import uuid

from pydantic import BaseModel


# Create a status enum with values PENDING, IN_PROGRESS, COMPLETED, and FAILED
class TaskStatus(Enum):
    PENDING = 0
    IN_PROGRESS = 1
    COMPLETED = 2
    FAILED = 3


class TaskOutcome(Enum):
    NOT_YET_EVALUATED = 0
    MEETS_REQUIREMENT = 1
    PARTIALLY_MEETS_REQUIREMENT = 2
    DOES_NOT_MEET_REQUIREMENT = 3
    OTHER = 4

    @classmethod
    def from_str(cls, value: str) -> "TaskOutcome":
        """
        Converts a string to a TaskOutcome enum
        """
        try:
            return cls[value.upper()]
        except KeyError:
            return cls.OTHER


class TaskResult(BaseModel):
    """
    TaskResult represents the result of a task
    """

    result: str
    evaluation: str
    outcome: Optional[TaskOutcome] = None

    def __str__(self):
        return f"Result={self.result} | Evaluation={self.evaluation} | Outcome={self.outcome}"


class Task(BaseModel):
    """
    Task represents a task that an agent must complete
    """

    id: str
    goal: str
    description: str
    result: Union[TaskResult, None] = None
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __init__(self, **data):
        # create the id if it does not exist
        if "id" not in data:
            data["id"] = str(uuid.uuid4())

        super().__init__(**data)

    def update(self, result: TaskResult):
        """
        Updates the task with the result from its execution
        :param result: The result for executing the task
        """
        if result.outcome == TaskOutcome.MEETS_REQUIREMENT:
            self.status = TaskStatus.COMPLETED
        elif result.outcome == TaskOutcome.PARTIALLY_MEETS_REQUIREMENT:
            self.status = TaskStatus.IN_PROGRESS
        elif result.outcome == TaskOutcome.DOES_NOT_MEET_REQUIREMENT:
            self.status = TaskStatus.FAILED
        else:
            self.status = TaskStatus.FAILED

        self.result = result

    def is_completed(self) -> bool:
        """
        Checks if the task is completed
        :return: True if the task is completed, False otherwise
        """
        return self.status == TaskStatus.COMPLETED

    def __str__(self):
        return (
            f"Goal={self.goal} | Description={self.description} | Status={self.status}"
        )
