# Standard Library
import json
import logging
import os
from typing import Dict, List

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


def test_groq_whisper_transcription_success_with_technical_information(youtube_video):
    """Test successful transcription of a YouTube video."""
    # Arrange
    tool = GroqWhisperTool()
    config = GroqWhisperConfig(
        file_path=youtube_video,
        model="whisper-large-v3",
        response_format="verbose_json",
        language="en",
        keep_technical_information=True,
    )

    # Act
    result = tool._run(config)

    # Print result for debugging
    logging.debug(f"Transcription result: {json.dumps(result, indent=2)}")

    # Assert
    assert isinstance(result, Dict)
    assert "full_transcription" in result
    assert "segments" in result
    assert "file_name" in result

    assert result["model_used"] == "whisper-large-v3"
    assert result["language"] == "en"
    assert result["response_format"] == "verbose_json"

    # Check the segments
    assert len(result["segments"]) > 0
    first_segment = result["segments"][0]

    logging.debug(f"First segment: {json.dumps(first_segment, indent=2)}")

    assert "start" in first_segment and isinstance(first_segment["start"], int | float)
    assert "end" in first_segment and isinstance(first_segment["end"], int | float)
    assert "text" in first_segment and isinstance(first_segment["text"], str)
    assert len(first_segment["text"]) > 0

    assert "tokens" in first_segment
    assert "temperature" in first_segment
    assert "avg_logprob" in first_segment
    assert "compression_ratio" in first_segment
    assert "no_speech_prob" in first_segment
    assert "id" in first_segment
    assert "seek" in first_segment


def test_groq_whisper_transcription_success_without_technical_information(youtube_video):
    """Test successful transcription of a YouTube video without technical information."""
    # Arrange
    tool = GroqWhisperTool()
    config = GroqWhisperConfig(
        file_path=youtube_video,
        model="whisper-large-v3",
        response_format="verbose_json",
        language="en",
        keep_technical_information=False,
    )
    # Act
    result = tool._run(config)

    # Print result for debugging
    logging.debug(f"Transcription result: {json.dumps(result, indent=2)}")

    # Assert
    assert isinstance(result, Dict)
    assert "full_transcription" in result
    assert "segments" in result
    assert "file_name" in result

    assert result["model_used"] == "whisper-large-v3"
    assert result["language"] == "en"
    assert result["response_format"] == "verbose_json"

    # Check the segments
    assert len(result["segments"]) > 0
    first_segment = result["segments"][0]

    logging.debug(f"First segment: {json.dumps(first_segment, indent=2)}")

    assert "start" in first_segment and isinstance(first_segment["start"], int | float)
    assert "end" in first_segment and isinstance(first_segment["end"], int | float)
    assert "text" in first_segment and isinstance(first_segment["text"], str)
    assert len(first_segment["text"]) > 0

    assert "tokens" not in first_segment
    assert "temperature" not in first_segment
    assert "avg_logprob" not in first_segment
    assert "compression_ratio" not in first_segment
    assert "no_speech_prob" not in first_segment
    assert "id" not in first_segment
    assert "seek" not in first_segment


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
