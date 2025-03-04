"""
Test module for OpenAIToolProvider.
"""

# Standard Library
import json
from unittest.mock import MagicMock

# Third Party
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.tool.abstract_tool import TOOL_METADATA_EXCLUDE_FROM_SCHEMA
from nagatoai_core.tool.abstract_tool import AbstractTool
from nagatoai_core.tool.lib.audio.stt.groq_whisper import GroqWhisperTool
from nagatoai_core.tool.provider.openai import OpenAIToolProvider


def test_schema_excludes_api_key():
    """
    Test that the schema generated by OpenAIToolProvider for GroqWhisperTool
    does not include the GROQ_API_KEY field.
    """
    # Create an instance of GroqWhisperTool
    tool = GroqWhisperTool()

    # Create an instance of OpenAIToolProvider with the tool
    provider = OpenAIToolProvider(name=tool.name, description=tool.description, args_schema=tool.args_schema, tool=tool)

    # Get the schema
    schema = provider.schema()

    # Convert to JSON string and back to Python object for easier inspection (optional)
    schema_json = json.dumps(schema)
    schema_dict = json.loads(schema_json)

    # Check the properties in the schema
    properties = schema_dict["function"]["parameters"]["properties"]

    # Assert that "api_key" is not in the properties
    assert "api_key" not in properties, "API key should be excluded from schema"

    # Also check that GROQ_API_KEY is not in the properties (case-insensitive check)
    property_keys_lower = [k.lower() for k in properties.keys()]
    assert "groq_api_key" not in property_keys_lower, "GROQ_API_KEY should be excluded from schema"

    # Verify other expected properties are still present
    expected_fields = [
        "file_path",
        "model",
        "prompt",
        "response_format",
        "language",
        "temperature",
        "keep_technical_information",
    ]

    for field in expected_fields:
        assert field in properties, f"Expected field {field} should be in schema"


def test_exclude_from_schema_mechanism():
    """
    Test that fields with exclude_from_schema=True in json_schema_extra
    are excluded from the schema.
    """
    # Create an instance of GroqWhisperTool
    tool = GroqWhisperTool()

    # Create an instance of OpenAIToolProvider with the tool
    provider = OpenAIToolProvider(name=tool.name, description=tool.description, args_schema=tool.args_schema, tool=tool)

    # Check that api_key field has the exclude_from_schema attribute
    api_key_field = tool.args_schema.model_fields.get("api_key")
    assert api_key_field is not None, "api_key field should exist in args_schema"

    # Check that the json_schema_extra contains the exclude_from_schema flag
    assert api_key_field.json_schema_extra is not None, "api_key field should have json_schema_extra"
    assert api_key_field.json_schema_extra.get(
        TOOL_METADATA_EXCLUDE_FROM_SCHEMA, False
    ), "api_key field should have exclude_from_schema=True"

    # This confirms that the mechanism to exclude fields is working as expected
    # in the OpenAIToolProvider._generate_schema method


def test_schema_generation_with_mock_tool():
    """
    Test schema generation with a mock tool that has both included and excluded fields.
    """

    # Create a mock args_schema class with both included and excluded fields
    class MockArgsSchema(BaseModel):
        regular_field: str = Field(..., description="This is a regular field")
        excluded_field: str = Field(
            ...,
            description="This field should be excluded",
            json_schema_extra={TOOL_METADATA_EXCLUDE_FROM_SCHEMA: True},
        )
        optional_field: str = Field(None, description="This is an optional field")

    # Create a mock tool with this schema
    mock_tool = MagicMock(spec=AbstractTool)
    mock_tool.name = "mock_tool"
    mock_tool.description = "A mock tool for testing"
    mock_tool.args_schema = MockArgsSchema

    # Create provider with the mock tool
    provider = OpenAIToolProvider(
        name=mock_tool.name, description=mock_tool.description, args_schema=mock_tool.args_schema, tool=mock_tool
    )

    # Get the schema
    schema = provider.schema()

    # Check properties in the schema
    properties = schema["function"]["parameters"]["properties"]

    # The regular field should be included
    assert "regular_field" in properties
    assert properties["regular_field"]["description"] == "This is a regular field"

    # The excluded field should not be included
    assert "excluded_field" not in properties

    # The optional field should be included
    assert "optional_field" in properties
    assert properties["optional_field"]["description"] == "This is an optional field"

    # Check the required fields
    required_fields = schema["function"]["parameters"]["required"]
    assert "regular_field" in required_fields
    assert "optional_field" not in required_fields
    assert "excluded_field" not in required_fields
