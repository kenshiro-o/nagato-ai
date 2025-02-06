# Standard Library
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Type

# Third Party
from groq import Groq, RateLimitError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Nagato AI
# Company Libraries
from nagatoai_core.tool.abstract_tool import AbstractTool
from nagatoai_core.tool.common.audio.audio_segment_with_timestamps import AudioSegmentWithOffsets
from nagatoai_core.tool.common.audio.utils import preprocess_audio, split_audio_in_chunks


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

    prompt: str = Field(
        "Specify context or spelling",
        description="Optional context or spelling information to guide the transcription.",
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
            chunks = split_audio_in_chunks(audio_preprocessed)

            transcription_results = []
            for chunk in chunks:
                transcription_results.append(self._process_single_chunk(client, chunk, config))

            # with open(config.file_path, "rb") as file:
            #     transcription = client.audio.transcriptions.create(
            #         file=(os.path.basename(config.file_path), file.read()),
            #         model=config.model,
            #         prompt=config.prompt,
            #         response_format=config.response_format,
            #         language=config.language,
            #         temperature=config.temperature,
            #     )

            print(json.dumps(transcription_results, indent=4))
            return transcription_results

            # output = {
            #     "transcription": transcription.text,
            #     "file_name": os.path.basename(config.file_path),
            #     "model_used": config.model,
            #     "language": config.language or "auto-detected",
            #     "response_format": config.response_format,
            # }

            # if config.response_format == "verbose_json":
            #     output["segments"] = transcription.segments

            # return output

        except Exception as e:
            raise RuntimeError(f"Error transcribing audio/video file: {str(e)}")
