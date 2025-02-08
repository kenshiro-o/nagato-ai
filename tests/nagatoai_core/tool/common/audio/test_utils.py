# Standard Library
from pathlib import Path

# Third Party
import pytest

# Nagato AI
from nagatoai_core.tool.common.audio.audio_segment_with_timestamps import AudioSegmentWithOffsets
from nagatoai_core.tool.common.audio.utils import (
    find_longest_common_string_overlap,
    preprocess_audio,
    split_audio_in_chunks,
)


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


def test_preprocess_audio_success(youtube_video):
    """Test successful audio preprocessing from a video file"""
    # Process the test video
    output_path = preprocess_audio(youtube_video)

    try:
        # Verify the output file exists and has the correct extension
        assert output_path.exists()
        assert output_path.suffix == ".flac"
    finally:
        # Clean up the temporary file
        output_path.unlink(missing_ok=True)


def test_preprocess_audio_file_not_found():
    """Test preprocessing with non-existent file raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        preprocess_audio(Path("nonexistent_file.mp4"))


def test_split_audio_in_chunks_success(youtube_video):
    """Test successful splitting of audio into chunks"""
    # First preprocess the audio
    processed_path = preprocess_audio(youtube_video)

    try:
        # Split the audio into chunks
        chunks = split_audio_in_chunks(processed_path, chunk_length=600, overlap_seconds=10)

        # Verify we got a list of AudioSegmentWithOffsets
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, AudioSegmentWithOffsets) for chunk in chunks)

        # Verify chunk properties
        for i, chunk in enumerate(chunks):
            # Check if offsets are properly set
            if i > 0:
                # Verify overlap with previous chunk
                assert chunk.from_second_offset == (chunks[i - 1].to_second_offset - 10)

            # Verify chunk duration (should be <= 600 seconds)
            chunk_duration = chunk.to_second_offset - chunk.from_second_offset
            assert chunk_duration <= 600

            # Verify the audio segment exists
            assert chunk.audio is not None

    finally:
        # Clean up
        processed_path.unlink(missing_ok=True)


def test_split_audio_in_chunks_file_not_found():
    """Test splitting audio with non-existent file raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        split_audio_in_chunks(Path("nonexistent_file.mp4"))


def test_find_longest_common_string_overlap_different_lengths():
    """Test overlaps with strings of different lengths"""
    # Shorter overlap at end/start
    assert find_longest_common_string_overlap("hello world", "world peace") == 5
    # Longer first string
    assert find_longest_common_string_overlap("this is a test string", "string theory") == 6
    # Longer second string
    assert find_longest_common_string_overlap("end of", "of the long string") == 2
    # No overlap
    assert find_longest_common_string_overlap("hello", "world") == 0


def test_find_longest_common_string_overlap_with_punctuation():
    """Test overlaps with punctuation marks"""
    # Default behavior (remove_punctuation=True)
    assert find_longest_common_string_overlap("end of sentence.", "sentence! starts", remove_punctuation=True) == 8
    # Keep punctuation
    assert find_longest_common_string_overlap("end of sentence.", "sentence! starts", remove_punctuation=False) == 0
    # Mixed punctuation
    assert find_longest_common_string_overlap("hello... world", "world!!! next", remove_punctuation=True) == 5


def test_find_longest_common_string_overlap_whitespace():
    """Test overlaps with various whitespace configurations"""
    # Default behavior (ignore_leading_trailing_whitespace=True)
    assert find_longest_common_string_overlap("  hello world  ", "world peace  ") == 5

    # Don't ignore whitespace - it will match on the second whitespace in "  world" and and "world" itself
    assert (
        find_longest_common_string_overlap("hello world  ", "  world peace", ignore_leading_trailing_whitespace=False)
        == 2
    )
    # Multiple spaces
    assert (
        find_longest_common_string_overlap("end   of    text", "text   begins", ignore_leading_trailing_whitespace=True)
        == 4
    )


def test_find_longest_common_string_overlap_edge_cases():
    """Test edge cases for overlap detection"""
    # Empty strings
    assert find_longest_common_string_overlap("", "") == 0
    assert find_longest_common_string_overlap("hello", "") == 0
    assert find_longest_common_string_overlap("", "world") == 0
    # Single character overlap
    assert find_longest_common_string_overlap("a", "a") == 1
    # Complete overlap (identical strings)
    assert find_longest_common_string_overlap("hello", "hello") == 5
    # Case sensitivity
    assert find_longest_common_string_overlap("Hello World", "world peace") == 0
