# Standard Library
from typing import Dict, Optional

# Third Party
from pydantic import BaseModel


class Goal(BaseModel):
    """
    Represents a plain-English goal statement with an optional structured specification.

    :param description: A plain-English statement of the goal.
    :param structured_spec: An optional typed refinement of the goal as a dictionary.
    """

    description: str
    structured_spec: Optional[Dict] = None
