# Standard Library
import os

# Third Party
import pytest

# Nagato AI
from nagatoai_core.tool.lib.video.youtube.video_download import YouTubeVideoDownloadConfig, YouTubeVideoDownloadTool


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs"""
    return str(tmp_path)


@pytest.fixture
def video_download_tool():
    """Create an instance of the YouTubeVideoDownloadTool"""
    tool = YouTubeVideoDownloadTool()
    tool.po_token_file = "./tests/data/youtube_po_token.json"
    return tool


@pytest.mark.skip(reason="Skipping test_valid_video_download until issue in Github Actions is resolved")
def test_valid_video_download(video_download_tool, temp_output_dir):
    """Test downloading a valid YouTube video"""
    config = YouTubeVideoDownloadConfig(
        video_id="q7_5eCmu0MY", output_folder=temp_output_dir, file_name="test_video.mp4"
    )

    # Download the video
    output_path = video_download_tool._run(config)

    # Assert the file exists and has content
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    assert output_path.endswith("test_video.mp4")


def test_invalid_video_id(video_download_tool, temp_output_dir):
    """Test that an invalid video ID raises an appropriate error"""
    config = YouTubeVideoDownloadConfig(video_id="not_valid", output_folder=temp_output_dir, file_name="test_video.mp4")

    # Attempt to download invalid video
    with pytest.raises(RuntimeError) as exc_info:
        video_download_tool._run(config)

    assert "Error downloading YouTube video" in str(exc_info.value)
