# Standard Library
from pathlib import Path

# Third Party
import pytest

# Nagato AI
from nagatoai_core.tool.common.audio.audio_segment_with_timestamps import AudioSegmentWithOffsets
from nagatoai_core.tool.common.audio.utils import preprocess_audio, split_audio_in_chunks


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
