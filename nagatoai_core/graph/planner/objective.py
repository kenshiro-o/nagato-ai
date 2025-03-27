# Standard Library
from typing import List

# Third Party
from pydantic import BaseModel, Field


class Objective(BaseModel):
    """
    Represents an objective derived from a user request.

    Contains the original user request, the extracted main objective,
    and a list of milestones (sub-objectives) needed to complete the main objective.
    """

    user_request: str = Field(description="The original request from the user")

    objective: str = Field(description="The main objective extracted from the user request")

    milestones: List[str] = Field(
        default_factory=list, description="A list of sub-objectives necessary to complete the overarching objective"
    )
