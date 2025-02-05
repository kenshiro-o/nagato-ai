# Standard Library
import os
from typing import Type
import tempfile
import subprocess
from pathlib import Path

# Third Party
from groq import Groq
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Nagato AI
# Company Libraries
from nagatoai_core.tool.abstract_tool import AbstractTool


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

    def _preprocess_audio(self) -> Path:
        """
        Preprocess audio file to 16kHz mono FLAC using ffmpeg.
        FLAC provides lossless compression for faster upload times.
        :return: The path to the preprocessed audio file.
        """
        input_path = Path(self.config.file_path)
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
                    "16000",
                    "-ac",
                    "1",
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

    def _run(self, config: GroqWhisperConfig) -> dict:
        """
        Transcribes an audio/video file using Whisper on Groq.
        :param config: The configuration for the GroqWhisperTool.
        :return: A dictionary containing the transcription text and additional metadata.
        """
        try:
            if not os.path.exists(config.file_path):
                return {"error": "File not found", "file_path": config.file_path}

            client = Groq(api_key=config.api_key)

            with open(config.file_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(config.file_path), file.read()),
                    model=config.model,
                    prompt=config.prompt,
                    response_format=config.response_format,
                    language=config.language,
                    temperature=config.temperature,
                )

            output = {
                "transcription": transcription.text,
                "file_name": os.path.basename(config.file_path),
                "model_used": config.model,
                "language": config.language or "auto-detected",
                "response_format": config.response_format,
            }

            if config.response_format == "verbose_json":
                output["segments"] = transcription.segments

            return output

        except Exception as e:
            raise RuntimeError(f"Error transcribing audio/video file: {str(e)}")
