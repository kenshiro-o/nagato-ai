from enum import Enum
from typing import Optional, List, Dict, Any, Union

from pydantic import BaseModel


class ToolCall(BaseModel):
    """
    ToolCall represents a request to call a tool
    """

    id: str
    name: str
    parameters: Dict


class ToolResult(BaseModel):
    """
    ToolResult represents the result of calling a tool
    """

    id: str
    name: str
    result: Any
    error: Optional[Any]


class ToolRun(BaseModel):
    """
    ToolRun represents the run of calling and receiving a response from a tool
    """

    id: str
    call: ToolCall
    result: Optional[ToolResult]


class Sender(Enum):
    """
    Sender represents the sender of a message
    """

    USER = 0
    AGENT = 1
    TOOL_RESULT = 2


class Message(BaseModel):
    """
    Message represents a message sent by the user or the agent
    """

    sender: Sender
    content: str
    tool_calls: List[ToolCall] = []
    tool_results: List[ToolResult] = []


class Exchange(BaseModel):
    """
    Exchange represents a message exchange between the user and the agent
    """

    user_msg: Message
    agent_response: Message
