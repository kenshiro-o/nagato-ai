# Standard Library
from datetime import datetime
from typing import List, Optional

# Third Party
import pytest
from pydantic import BaseModel, Field

# Nagato AI
# Company Libraries
from nagatoai_core.tool.abstract_tool import TOOL_METADATA_EXCLUDE_FROM_SCHEMA, AbstractTool
from nagatoai_core.tool.provider.anthropic import AnthropicToolProvider


class InputSchemaFixture(BaseModel):
    name: str = Field(description="The name field")
    age: int = Field(description="The age field")
    tags: List[str] = Field(description="List of tags")
    birthday: datetime = Field(description="Birthday in datetime")
    nickname: Optional[str] = Field(default=None, description="Optional nickname")
    hidden: str = Field(
        description="Hidden field",
        json_schema_extra={TOOL_METADATA_EXCLUDE_FROM_SCHEMA: True},
    )


class ToolFixture(AbstractTool):
    """Test tool implementation"""

    def _run(self, args: InputSchemaFixture) -> str:
        """Mock implementation of run method"""
        return f"Test run with name: {args.name}"


@pytest.fixture
def tool_provider():
    tool = ToolFixture(name="test_tool", description="A test tool", args_schema=InputSchemaFixture)
    return AnthropicToolProvider(
        tool=tool,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def test_schema_basic_structure(tool_provider):
    """Test the basic structure of the generated schema"""
    schema = tool_provider.schema()

    assert schema["name"] == "test_tool"
    assert schema["description"] == "A test tool"
    assert "input_schema" in schema
    assert schema["input_schema"]["type"] == "object"
    assert "properties" in schema["input_schema"]
    assert "required" in schema["input_schema"]


def test_schema_field_types(tool_provider):
    """Test that field types are correctly converted"""
    schema = tool_provider.schema()
    properties = schema["input_schema"]["properties"]

    # Test string field
    assert properties["name"]["type"] == "string"
    assert properties["name"]["description"] == "The name field"

    # Test integer field
    assert properties["age"]["type"] == "integer"
    assert properties["age"]["description"] == "The age field"

    # Test array field
    assert properties["tags"]["type"] == "array"
    assert properties["tags"]["items"]["type"] == "string"

    # Test datetime field
    assert properties["birthday"]["type"] == "string"
    assert properties["birthday"]["format"] == "date-time"


def test_schema_optional_fields(tool_provider):
    """Test handling of optional fields"""
    schema = tool_provider.schema()
    properties = schema["input_schema"]["properties"]

    # Test optional field
    assert properties["nickname"]["type"] == "string"
    assert properties["nickname"]["nullable"] is True


def test_schema_required_fields(tool_provider):
    """Test that required fields are correctly marked"""
    schema = tool_provider.schema()
    required_fields = schema["input_schema"]["required"]

    assert "name" in required_fields
    assert "age" in required_fields
    assert "tags" in required_fields
    assert "birthday" in required_fields
    assert "nickname" not in required_fields


def test_schema_excluded_fields(tool_provider):
    """Test that excluded fields are not included in the schema"""
    schema = tool_provider.schema()
    properties = schema["input_schema"]["properties"]

    assert "hidden" not in properties


def test_field_schema_list_type(tool_provider):
    """Test handling of list field types"""
    list_type = List[str]
    schema = tool_provider._get_field_schema(list_type)

    assert schema["type"] == "array"
    assert schema["items"]["type"] == "string"


def test_field_schema_optional_type(tool_provider):
    """Test handling of optional field types"""
    optional_type = Optional[str]
    schema = tool_provider._get_field_schema(optional_type)

    assert schema["type"] == "string"
    assert schema["nullable"] is True


def test_field_schema_datetime_type(tool_provider):
    """Test handling of datetime field type"""
    schema = tool_provider._get_field_schema(datetime)

    assert schema["type"] == "string"
    assert schema["format"] == "date-time"
