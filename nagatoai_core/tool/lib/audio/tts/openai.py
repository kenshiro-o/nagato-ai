from typing import Type

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
from openai import OpenAI

from nagatoai_core.tool.abstract_tool import AbstractTool


class OpenAITTSConfig(BaseSettings, BaseModel):
    api_key: str = Field(
        ...,
        description="The OpenAI API key to use for authentication.",
        alias="OPENAI_API_KEY",
        exclude_from_schema=True,
    )

    model: str = Field(
        "tts-1",
        description="The OpenAI TTS model to use for generating audio. Valid values are 'tts-1' or 'tts-1-hd'. Default is 'tts-1'.",
    )

    voice: str = Field(
        "alloy",
        description="The voice to use for the generated audio. Supported voices are 'alloy', 'echo', 'fable', 'onyx', 'nova', and 'shimmer'. Default is 'alloy'.",
    )

    input_text: str = Field(
        ...,
        description="The input text to convert to audio.",
    )

    output_file: str = Field(
        "output.mp3",
        description="The output file path to save the generated audio.",
    )


class OpenAITTSTool(AbstractTool):
    """
    OpenAITTS is a tool that generates an audio file from given input text using OpenAI's Text-to-Speech (TTS) API.
    """

    name: str = "openai_tts"
    description: str = (
        """Generates an audio file from the given input text using OpenAI's Text-to-Speech (TTS) API.
        Returns the path of the generated audio file.
        """
    )
    args_schema: Type[BaseModel] = OpenAITTSConfig

    def _run(self, config: OpenAITTSConfig) -> str:
        """
        Generates an audio file from the given input text using OpenAI's Text-to-Speech (TTS) API.
        :param config: The configuration for the OpenAITTS tool.
        :return: The path of the generated audio file.
        """
        client = OpenAI(api_key=config.api_key)

        response = client.audio.speech.create(
            model=config.model,
            voice=config.voice,
            input=config.input_text,
        )

        response.write_to_file(config.output_file)

        return config.output_file
