import os
import pytest
from pathlib import Path
from nagatoai_core.tool.common.audio.utils import preprocess_audio
from nagatoai_core.tool.lib.video.youtube.video_download import YouTubeVideoDownloadTool, YouTubeVideoDownloadConfig


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs"""
    return str(tmp_path)


def test_preprocess_audio_from_youtube(youtube_video):
    """Test preprocessing a valid audio file downloaded from YouTube"""
    input_path = Path(youtube_video)

    # Process the video to FLAC
    output_path = preprocess_audio(input_path, sample_rate=16000, channels=1)

    # Verify the output file exists and has content
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.suffix == ".flac"

    # Cleanup
    output_path.unlink(missing_ok=True)


def test_preprocess_audio_nonexistent_file(temp_output_dir):
    """Test that preprocessing a non-existent file raises FileNotFoundError"""
    nonexistent_file = Path(temp_output_dir) / "nonexistent.mp4"

    with pytest.raises(FileNotFoundError) as exc_info:
        preprocess_audio(nonexistent_file)

    assert "Input file not found" in str(exc_info.value)
