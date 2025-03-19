"""Tests for GoogleToolProvider."""

# Standard Library
from typing import Dict, Type

# Third Party
import pytest
from google.genai import types
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.tool.abstract_tool import AbstractTool
from nagatoai_core.tool.lib.time.time_now import TimeNowTool
from nagatoai_core.tool.lib.video.youtube.video_download import YouTubeVideoDownloadTool
from nagatoai_core.tool.provider.google import GoogleToolProvider


@pytest.fixture
def time_now_tool():
    """Fixture for TimeNowTool."""
    return TimeNowTool()


@pytest.fixture
def youtube_video_download_tool():
    """Fixture for YouTubeVideoDownloadTool."""
    return YouTubeVideoDownloadTool()


def test_time_now_tool_schema(time_now_tool: TimeNowTool):
    """Test schema generation for TimeNowTool."""
    provider = GoogleToolProvider(
        name=time_now_tool.name,
        description=time_now_tool.description,
        args_schema=time_now_tool.args_schema,
        tool=time_now_tool,
    )
    schema = provider.schema()

    assert isinstance(schema, types.FunctionDeclaration)
    assert schema.name == "time_now"
    assert schema.description == "Returns the time now in UTC."

    # Check parameters schema
    assert schema.parameters.type == "OBJECT"
    assert isinstance(schema.parameters.properties, Dict)

    # Check use_utc_timezone property
    assert "use_utc_timezone" in schema.parameters.properties
    utc_prop = schema.parameters.properties["use_utc_timezone"]
    assert utc_prop.type == "BOOLEAN"
    assert utc_prop.description == "Whether to use the UTC timezone."

    # Check that required fields are correctly set
    assert not schema.parameters.required  # use_utc_timezone is optional with default value


def test_youtube_video_download_tool_schema(youtube_video_download_tool: YouTubeVideoDownloadTool):
    """Test schema generation for YouTubeVideoDownloadTool."""
    provider = GoogleToolProvider(
        name=youtube_video_download_tool.name,
        description=youtube_video_download_tool.description,
        args_schema=youtube_video_download_tool.args_schema,
        tool=youtube_video_download_tool,
    )
    schema = provider.schema()

    assert isinstance(schema, types.FunctionDeclaration)
    assert schema.name == "youtube_video_download"
    assert "Downloads a video from YouTube" in schema.description

    # Check parameters schema
    assert schema.parameters.type == "OBJECT"
    assert isinstance(schema.parameters.properties, Dict)

    # Check required properties
    required_fields = ["video_id", "file_name"]
    for field in required_fields:
        assert field in schema.parameters.properties
        assert field in schema.parameters.required

    # Check video_id property
    video_id_prop = schema.parameters.properties["video_id"]
    assert video_id_prop.type == "STRING"
    assert "ID of the YouTube video" in video_id_prop.description

    # Check output_path property
    output_path_prop = schema.parameters.properties["output_folder"]
    assert output_path_prop.type == "STRING"
    assert "output directory path" in output_path_prop.description.lower()

    # Check file_name property
    file_name_prop = schema.parameters.properties["file_name"]
    assert file_name_prop.type == "STRING"
    assert "name of the file" in file_name_prop.description.lower()


def test_schema_with_nullable_field():
    """Test schema generation for a tool with nullable fields."""

    class TestToolConfig(BaseModel):
        field_with_default: str = Field("default_value", description="Field default value is 'default_value'")

    class TestTool(AbstractTool):
        """Test tool with a nullable field."""

        name: str = "test_tool"
        description: str = "Test tool description"
        args_schema: Type[BaseModel] = TestToolConfig

        def _run(self, config: TestToolConfig) -> str:
            return ""

    test_tool = TestTool()
    provider = GoogleToolProvider(
        name=test_tool.name,
        description=test_tool.description,
        args_schema=test_tool.args_schema,
        tool=test_tool,
    )
    schema = provider.schema()

    assert "field_with_default" in schema.parameters.properties
    field_with_default_prop = schema.parameters.properties["field_with_default"]
    assert field_with_default_prop.type == "STRING"
    assert field_with_default_prop.description == "Field default value is 'default_value'"
