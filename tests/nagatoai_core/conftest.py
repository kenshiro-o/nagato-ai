"""
This module contains shared fixtures across all tests.
"""

# Standard Library
import logging

# Third Party
import pytest

# Nagato AI
from nagatoai_core.tool.lib.video.youtube.video_download import YouTubeVideoDownloadConfig, YouTubeVideoDownloadTool


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests"""
    logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(scope="session")
def youtube_video(tmp_path_factory):
    """
    Download a test YouTube video for audio preprocessing once per test session.
    """
    # Create a temporary directory that persists for the whole session.
    session_temp_dir = tmp_path_factory.mktemp("youtube_video")

    tool = YouTubeVideoDownloadTool()
    config = YouTubeVideoDownloadConfig(
        video_id="q7_5eCmu0MY",
        output_path=str(session_temp_dir),  # Ensure the path is a string if required
        file_name="test_video.mp4",
    )

    return tool._run(config)
