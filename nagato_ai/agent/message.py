from enum import Enum

from pydantic import BaseModel


class Sender(Enum):
    """
    Sender represents the sender of a message
    """

    USER = 0
    AGENT = 1


class Message(BaseModel):
    """
    Message represents a message sent by the user or the agent
    """

    sender: Sender
    content: str


class Exchange(BaseModel):
    """
    Exchange represents a message exchange between the user and the agent
    """

    user_msg: Message
    agent_response: Message
