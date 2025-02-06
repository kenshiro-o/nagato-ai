# Standard Library
import json
import logging
import os

# Third Party
import pytest
from dotenv import load_dotenv

# Nagato AI
from nagatoai_core.tool.lib.audio.stt.groq_whisper import GroqWhisperConfig, GroqWhisperTool


@pytest.fixture(autouse=True)
def setup_env():
    """Load environment variables before each test."""
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set in environment")


def test_groq_whisper_transcription_success(youtube_video):
    """Test successful transcription of a YouTube video."""
    # Arrange
    tool = GroqWhisperTool()
    config = GroqWhisperConfig(
        file_path=youtube_video,
        model="whisper-large-v3",
        response_format="verbose_json",
        language="en",
    )

    # Act
    result = tool._run(config)

    # Print result for debugging
    logging.info(f"Transcription result: {json.dumps(result, indent=2)}")

    # Assert
    assert isinstance(result, list)
    assert len(result) > 0

    # Check first chunk's structure
    first_chunk = result[0]
    assert "text" in first_chunk
    assert "segments" in first_chunk
    assert "from_second" in first_chunk
    assert "to_second" in first_chunk
    assert isinstance(first_chunk["text"], str)
    assert len(first_chunk["text"]) > 0


def test_groq_whisper_file_not_found():
    """Test transcription with non-existent file."""
    # Arrange
    tool = GroqWhisperTool()
    config = GroqWhisperConfig(
        file_path="/path/to/nonexistent/file.mp4",
        model="whisper-large-v3",
        response_format="verbose_json",
        language="en",
    )

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="File not found:"):
        tool._run(config)
