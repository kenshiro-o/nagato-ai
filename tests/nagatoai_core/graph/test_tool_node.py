import pytest
from datetime import UTC, datetime, timedelta
import logging

# Import necessary modules and classes
from typing import Any, Type

from pydantic import BaseModel, Field


from nagatoai_core.tool.abstract_tool import AbstractTool
from nagatoai_core.tool.provider.openai import OpenAIToolProvider
from nagatoai_core.graph.tool_node import ToolNode
from nagatoai_core.graph.types import NodeResult

from nagatoai_core.tool.lib.time.time_now import TimeNowTool, TimeNowConfig


@pytest.fixture
def tool_provider():
    """Fixture providing an instance of TimeNowTool."""

    t = TimeNowTool()
    return OpenAIToolProvider(name=t.name, description=t.description, args_schema=t.args_schema, tool=t)


def test_correct_tool_execution(tool_provider: Type[AbstractTool]):
    """Test the correct execution of a ToolNode with valid parameters."""

    node = ToolNode(id="tool_node", parents=[], descendants=[], tool_provider=tool_provider)
    inputs = [NodeResult(node_id="some_id", result={"use_utc_timezone": False}, error=None, step=1)]

    result = node.execute(inputs)

    assert isinstance(result, list)
    assert len(result) == 1
    current_time_str = datetime.now(UTC).isoformat().split("+")[0]  # Remove the timezone part

    result_item = result[0]
    assert isinstance(result_item, NodeResult)
    assert result_item.node_id is not None

    # Can this be more elegant in the future
    time_str = result_item.result.split("+")[0]  # Remove the timezone part
    tool_time_result = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")

    current_time = datetime.strptime(current_time_str, "%Y-%m-%dT%H:%M:%S.%f")

    delta: timedelta = tool_time_result - current_time
    logging.info(
        f"current time is {current_time} while tool time is {tool_time_result} and total seconds delta is {delta.total_seconds()}"
    )

    assert delta.total_seconds() <= 5
    assert result_item.step == 2


# def test_incorrect_tool_execution(tool_provider: AbstractTool):
#     """Test the execution of a ToolNode with invalid parameters, resulting in an error."""

#     node = ToolNode(id="tool_node", parents=[], descendants=[], tool_provider=tool_provider)

#     inputs = [NodeResult(node_id="some_id", result={"not_a_good_param": False}, error=None, step=1)]

#     results = node.execute(inputs)

#     assert isinstance(results, list)
#     assert len(results) == 1

#     result_item = results[0]
#     assert isinstance(result_item, NodeResult)
#     assert result_item.node_id is not None

#     assert result_item.result is None
#     assert results.step == 2
#     assert result_item.error is not None
