"""
This module contains shared fixtures across all tests.
"""

# Standard Library
import logging

# Third Party
import pytest


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests"""
    logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(scope="session")
def youtube_video():
    """
    Share the path to a downloaded YouTube video for testing.
    """
    return "./tests/data/test_youtube_video.mp4"
