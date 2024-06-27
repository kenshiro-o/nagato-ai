import os
from typing import Type
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
from openai import OpenAI

from nagatoai_core.tool.abstract_tool import AbstractTool


class OpenAIWhisperConfig(BaseSettings, BaseModel):
    api_key: str = Field(
        ...,
        description="The OpenAI API key to use for authentication.",
        alias="OPENAI_API_KEY",
        exclude_from_schema=True,
    )

    audio_full_path: str = Field(
        ...,
        description="The full path to the audio file to transcribe.",
    )

    model: str = Field(
        "whisper-1",
        description="The Whisper model to use for transcription. Default is 'whisper-1'.",
    )

    language: str = Field(
        "en",
        description="The language of the audio file. Default is 'en'. If not specified, Whisper will auto-detect the language.",
    )

    # TODO - do we need this?
    # prompt: str = Field(
    #     None,
    #     description="An optional text to guide the model's style or continue a previous audio segment.",
    # )

    response_format: str = Field(
        "verbose_json",
        description="The format of the transcript output. Options are 'json', 'text', 'srt', 'verbose_json', or 'vtt'. Default is 'verbose_json'.",
    )

    temperature: float = Field(
        0,
        description="The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Default is 0.",
    )


class OpenAIWhisperTool(AbstractTool):
    """
    OpenAIWhisperTool is a tool that transcribes audio files using OpenAI's Whisper model.
    """

    name: str = "openai_whisper_transcription"
    description: str = (
        """Transcribes an audio file using OpenAI's Whisper model.
        Returns the transcription text and additional metadata.
        """
    )
    args_schema: Type[BaseModel] = OpenAIWhisperConfig

    def _run(self, config: OpenAIWhisperConfig) -> dict:
        """
        Transcribes an audio file using OpenAI's Whisper model.
        :param config: The configuration for the OpenAIWhisperTool.
        :return: A dictionary containing the transcription text and additional metadata.
        """
        try:
            if not os.path.exists(config.audio_full_path):
                return {"error": "File not found", "file_path": config.audio_full_path}

            # Set up the OpenAI client
            client = OpenAI(api_key=config.api_key)

            # Open the audio file
            with open(config.audio_full_path, "rb") as audio_file:
                # Start the transcription
                transcript = client.audio.transcriptions.create(
                    model=config.model,
                    file=audio_file,
                    language=config.language,
                    # prompt=config.prompt,
                    response_format=config.response_format,
                    temperature=config.temperature,
                )

            # Prepare the result
            result = {
                "text": (
                    transcript.text if hasattr(transcript, "text") else str(transcript)
                ),
                "file_name": os.path.basename(config.audio_full_path),
                "model": config.model,
                "language": config.language or "auto-detected",
                "response_format": config.response_format,
                "temperature": config.temperature,
            }

            # If the response format is verbose_json, include additional details
            if config.response_format == "verbose_json" and hasattr(
                transcript, "segments"
            ):
                result["segments"] = transcript.segments

            return result

        except Exception as e:
            raise RuntimeError(f"Error transcribing audio file: {str(e)}")
