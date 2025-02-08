# Standard Library
import string
import subprocess
import tempfile
from pathlib import Path
from typing import List

# Third Party
from pydub import AudioSegment

# Nagato AI
from nagatoai_core.tool.common.audio.audio_segment_with_timestamps import AudioSegmentWithOffsets


def preprocess_audio(file_path: Path, sample_rate: int = 16000, channels: int = 1) -> Path:
    """
    Preprocess audio file to 16kHz mono FLAC using ffmpeg.
    FLAC provides lossless compression for faster upload times.

    :param file_path: The path to the file to preprocess (could be mp3, mp4, etc.)
    :param sample_rate: The sample rate in Hz for the output FLAC file. Default is 16000 Hz.
    :param channels: The number of channels for the output FLAC file. Default is 1 (mono).

    :raises FileNotFoundError: If the input file is not found.
    :raises RuntimeError: If the ffmpeg conversion fails.
    :return: The path to the preprocessed audio file.
    """
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as temp_file:
        output_path = Path(temp_file.name)

    print("Converting audio to 16kHz mono FLAC...")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-c:a",
                "flac",
                "-y",
                output_path,
            ],
            check=True,
        )
        return output_path
    # We'll raise an error if our FFmpeg conversion fails
    except subprocess.CalledProcessError as e:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")


def split_audio_in_chunks(
    audio_path: Path, chunk_length: int = 600, overlap_seconds: int = 10
) -> List[AudioSegmentWithOffsets]:
    """
    Split audio file into chunks with overlap.

    Args:
        audio_path: Path to audio file
        chunk_length: Length of each chunk in seconds
        overlap_seconds: Overlap between chunks in seconds

    Returns:
        List[AudioSegmentWithOffsets]: List of audio segments with their corresponding time offsets

    Raises:
        RuntimeError: If audio file fails to load
    """
    processed_path = None
    try:
        # Preprocess audio and get basic info
        processed_path = preprocess_audio(audio_path)
        audio = None

        try:
            audio = AudioSegment.from_file(processed_path, format="flac")
        except Exception as e:
            raise RuntimeError(f"Failed to process audio file with path {processed_path}") from e

        duration = len(audio)
        print(f"Audio duration: {duration/1000:.2f}s")

        # Calculate # of chunks
        chunk_ms = chunk_length * 1000
        overlap_ms = overlap_seconds * 1000
        total_chunks = (duration // (chunk_ms - overlap_ms)) + 1
        print(f"Processing {total_chunks} chunks...")

        audio_chunks: List[AudioSegmentWithOffsets] = []

        # Loop through each chunk and extract current chunk from audio
        for i in range(total_chunks):
            start = i * (chunk_ms - overlap_ms)
            end = min(start + chunk_ms, duration)

            print(f"\nProcessing chunk {i+1}/{total_chunks}")
            print(f"Time range: {start/1000:.1f}s - {end/1000:.1f}s")

            chunk = audio[start:end]
            audio_chunks.append(
                AudioSegmentWithOffsets(audio=chunk, from_second_offset=start / 1000, to_second_offset=end / 1000)
            )

        return audio_chunks

    finally:
        if processed_path:
            Path(processed_path).unlink(missing_ok=True)


def find_longest_common_string_overlap(
    text1: str, text2: str, ignore_leading_trailing_whitespace: bool = True, remove_punctuation: bool = True
) -> int:
    """
    Calculate the length of the longest overlap between the end of text1 and the beginning of text2.
    This overlap represents the longest common sequence shared at the boundary of the two segments.

    :param text1: The first text to compare
    :param text2: The second text to compare
    :param ignore_leading_trailing_whitespace: Whether to ignore leading and trailing whitespace
    :param remove_punctuation: Whether to remove punctuation
    :return: The length of the longest overlap
    """
    if ignore_leading_trailing_whitespace:
        text1 = text1.strip()
        text2 = text2.strip()

    # Also remove any punctuation from the text
    if remove_punctuation:
        text1 = text1.translate(str.maketrans("", "", string.punctuation))
        text2 = text2.translate(str.maketrans("", "", string.punctuation))

    max_overlap = min(len(text1), len(text2))
    for length in range(max_overlap, 0, -1):
        if text1[-length:] == text2[:length]:
            return length
    return 0
