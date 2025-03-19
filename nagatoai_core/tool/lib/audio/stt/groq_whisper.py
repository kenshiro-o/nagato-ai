# Standard Library
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Tuple, Type

# Third Party
from groq import Groq, RateLimitError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Nagato AI
# Company Libraries
from nagatoai_core.tool.abstract_tool import AbstractTool
from nagatoai_core.tool.common.audio.audio_segment_with_timestamps import AudioSegmentWithOffsets
from nagatoai_core.tool.common.audio.utils import (
    find_longest_common_string_overlap,
    preprocess_audio,
    split_audio_in_chunks,
)


class GroqWhisperConfig(BaseSettings, BaseModel):
    api_key: str = Field(
        ...,
        description="The Groq API key to use for authentication.",
        alias="GROQ_API_KEY",
        exclude_from_schema=True,
    )

    file_path: str = Field(
        ...,
        description="The full path to the audio/video file to transcribe.",
    )

    model: str = Field(
        "whisper-large-v3",
        description="The Whisper model to use for transcription. The only option available right now is 'whisper-large-v3'",
    )

    response_format: str = Field(
        "verbose_json",
        description="The format of the response. Possible values are 'json' and 'verbose_json'. Default is 'verbose_json'",
    )

    language: str = Field(
        "en",
        description="Optional language code for the audio language. Default is 'en' for English.",
    )

    temperature: float = Field(
        0.0,
        description="The sampling temperature for the model. Default is 0.0.",
    )

    keep_technical_information: bool = Field(
        False,
        description="Whether to keep technical information in the transcription result (e.g. tokens, temperature, avg_logprob, etc. ). Default is False.",
    )


class GroqWhisperTool(AbstractTool):
    """
    GroqWhisperTool is a tool that transcribes audio/video files using Whisper on Groq.
    """

    name: str = "groq_whisper_transcription"
    description: str = (
        """Transcribes an audio/video file using Whisper on Groq.
        Returns the transcription text and additional metadata.
        """
    )
    args_schema: Type[BaseModel] = GroqWhisperConfig

    def combine_segments_from_overlapping_chunks(
        self,
        chunk_a: Tuple[AudioSegmentWithOffsets, Dict[str, Any]],
        chunk_b: Tuple[AudioSegmentWithOffsets, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine segments from two overlapping chunks.
        """
        segments_a = chunk_a[1]["segments"]
        segments_b = chunk_b[1]["segments"]

        all_segments: List[Dict[str, Any]] = segments_a + segments_b

        # Find overlapping segments
        overlapping_segments: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for segment_a in segments_a:
            for segment_b in segments_b:
                if segment_a["end"] > segment_b["start"] and segment_a["start"] < segment_b["end"]:
                    overlapping_segments.append((segment_a, segment_b))

                    # If the segments are overlapping, remove them from all segments for now
                    all_segments = [s for s in all_segments if s != segment_a and s != segment_b]

        # Merge overlapping segments
        merged_segments = []
        for segment_a, segment_b in overlapping_segments:
            previous_merged_segment = merged_segments[-1] if merged_segments else None

            if (
                previous_merged_segment
                and previous_merged_segment["start"] < segment_b["start"]
                and previous_merged_segment["end"] >= segment_b["end"]
            ):
                logging.debug(
                    f"⚠️ Segments drifting not merging, segment_a: {json.dumps(segment_a, indent=2)}, segment_b: {json.dumps(segment_b, indent=2)}"
                )
                # merged_segments.append(segment_b)
                continue

            if (
                previous_merged_segment
                and previous_merged_segment["start"] == segment_b["start"]
                and previous_merged_segment["end"] == segment_b["end"]
            ):
                logging.debug(
                    f"❌ Previous merged segment start is equal to segment_b start (and end); not doing any merging, segment_a: {json.dumps(segment_a, indent=2)}, segment_b: {json.dumps(segment_b, indent=2)}"
                )
                continue

            if previous_merged_segment and previous_merged_segment["end"] > segment_a["start"]:
                logging.debug(
                    f"🎯 Previous merged segment end is greater than segment_a start; only using segment b, segment_a: {json.dumps(segment_a, indent=2)}, segment_b: {json.dumps(segment_b, indent=2)}"
                )
                merged_segments.append(segment_b)
                continue

            # Otherwise we can merge the segments
            merged_segment = self.merge_overlapping_segments(segment_a, segment_b)
            merged_segments.append(merged_segment)
            logging.debug(
                f"🚀 Merged non-drifting segments, segment_a: {json.dumps(segment_a, indent=2)}, segment_b: {json.dumps(segment_b, indent=2)}, merged_segment: {json.dumps(merged_segment, indent=2)}"
            )

        all_segments = merged_segments + all_segments

        # Sort the segments by start time
        all_segments = sorted(all_segments, key=lambda x: x["start"])

        return all_segments

    def merge_overlapping_segments(self, segment_a: Dict[str, Any], segment_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two overlapping segments by identifying and using the longest common overlap in their 'text' fields.
        The merged segment preserves the start time from segment_a and the end time from segment_b.

        :param segment_a: The first segment to merge
        :param segment_b: The second segment to merge
        :return: The merged segment
        """
        text_a = segment_a.get("text", "")
        text_b = segment_b.get("text", "")

        # Determine the overlapping length based on the longest common sequence at the boundary.
        overlap_len = find_longest_common_string_overlap(text_a, text_b)

        # Merge the texts by appending the non-overlapping portion of text_b to text_a.
        merged_text = text_a + text_b[overlap_len:]

        # Create the merged segment. We assume 'segment_a' provides the initial timing and metadata,
        # so we update its 'text' and use segment_b's 'end' value.
        merged_segment = segment_a.copy()
        merged_segment.update({"text": merged_text, "end": segment_b.get("end", segment_a.get("end"))})
        return merged_segment

    def _process_single_chunk(
        self, client: Groq, chunk: AudioSegmentWithOffsets, config: GroqWhisperConfig
    ) -> Dict[str, Any]:
        """
        Process a single chunk of audio using Whisper on Groq.
        """
        while True:
            with tempfile.NamedTemporaryFile(suffix=".flac") as temp_file:
                chunk.audio.export(temp_file.name, format="flac")

                try:
                    result = client.audio.transcriptions.create(
                        file=(os.path.basename(temp_file.name), open(temp_file.name, "rb")),
                        model=config.model,
                        language=config.language,
                        response_format=config.response_format,
                    )

                    output = {
                        "text": result.text,
                        "segments": result.segments,
                        "from_second": chunk.from_second_offset,
                        "to_second": chunk.to_second_offset,
                    }

                    if config.response_format == "verbose_json":
                        output["segments"] = result.segments

                    return output

                except RateLimitError:
                    print("Rate limit hit - retrying in 60 seconds...")
                    time.sleep(60)
                    continue

                except Exception as e:
                    print(f"Error transcribing chunk: {str(e)}")
                    raise

    def _run(self, config: GroqWhisperConfig) -> dict:
        """
        Transcribes an audio/video file using Whisper on Groq.
        :param config: The configuration for the GroqWhisperTool.
        :return: A dictionary containing the transcription text and additional metadata.
        """
        if not os.path.exists(config.file_path):
            raise FileNotFoundError(f"File not found: {config.file_path}")

        try:
            client = Groq(api_key=config.api_key)

            audio_preprocessed = preprocess_audio(config.file_path)
            chunk_length_seconds = 600
            overlap_seconds = 10
            chunks = split_audio_in_chunks(audio_preprocessed, chunk_length_seconds, overlap_seconds)

            transcription_results_with_chunks: List[Tuple[AudioSegmentWithOffsets, Dict[str, Any]]] = []
            all_combined_segments: List[Dict[str, Any]] = []
            for i, chunk in enumerate(chunks):
                transcribed_chunk = self._process_single_chunk(client, chunk, config)
                transcription_results_with_chunks.append((chunk, transcribed_chunk))

                if not "segments" in transcribed_chunk:
                    continue

                # Offset the start and end of each segment based on overlap and chunk index
                for segment in transcribed_chunk["segments"]:
                    segment["start"] += i * (chunk_length_seconds - overlap_seconds)
                    segment["end"] += i * (chunk_length_seconds - overlap_seconds)

                if i > 0:
                    combined_segments = self.combine_segments_from_overlapping_chunks(
                        transcription_results_with_chunks[i - 1], transcription_results_with_chunks[i]
                    )
                    logging.debug(f"Combined segments: {json.dumps(combined_segments, indent=2)}")
                    all_combined_segments.extend(combined_segments)

                output = {
                    "file_name": os.path.basename(config.file_path),
                    "model_used": config.model,
                    "language": config.language or "auto-detected",
                    "response_format": config.response_format,
                }

            # Special case for when there is only one chunk since no segment merging is required
            if len(transcription_results_with_chunks) == 1:
                all_combined_segments = transcription_results_with_chunks[0][1]["segments"]
                output["full_transcription"] = transcription_results_with_chunks[0][1]["text"]
            else:
                output["full_transcription"] = " ".join([segment["text"] for segment in all_combined_segments])

            if config.response_format == "verbose_json":
                output["segments"] = all_combined_segments
                # No need to keep the full transcription if we have the segments
                output.pop("full_transcription", None)

                if not config.keep_technical_information:
                    # Remove tokens, temperature, avg_logprob, compression_ratio, no_speech_prob from segments
                    for segment in output["segments"]:
                        segment.pop("id", None)
                        segment.pop("seek", None)
                        segment.pop("tokens", None)
                        segment.pop("temperature", None)
                        segment.pop("avg_logprob", None)
                        segment.pop("compression_ratio", None)
                        segment.pop("no_speech_prob", None)

            return output

        except Exception as e:
            raise RuntimeError(f"Error transcribing audio/video file: {str(e)}")
