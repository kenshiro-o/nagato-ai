import requests
from typing import Type

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings

from nagatoai_core.tool.abstract_tool import AbstractTool


class ElevenLabsTTSConfig(BaseSettings, BaseModel):
    api_key: str = Field(
        ...,
        description="The Eleven Labs API key to use for authentication.",
        alias="ELEVENLABS_API_KEY",
        exclude_from_schema=True,
    )

    voice_id: str = Field(
        default="E215QqruhZm2aa5Ye6SW",
        description="The ID of the voice to use for text-to-speech. By default we use Mathilda's voice id (E215QqruhZm2aa5Ye6SW).",
    )

    input_text: str = Field(
        ...,
        description="The input text to convert to speech.",
    )

    output_file: str = Field(
        "output.mp3",
        description="The output file path to save the generated audio. By default we save the audio file as output.mp3.",
    )

    model_id: str = Field(
        "eleven_monolingual_v1",
        description="The ID of the model to use for text-to-speech. Possible values are 'eleven_monolingual_v1' or 'eleven_multilingual_v1', 'eleven_multilingual_v2', and 'eleven_turbo_v2'. By default we use 'eleven_monolingual_v1'.",
    )

    stability: float = Field(
        0.5,
        description="The stability value for the voice settings.",
    )

    similarity_boost: float = Field(
        0.5,
        description="The similarity boost value for the voice settings.",
    )


class ElevenLabsTTSTool(AbstractTool):
    """
    ElevenLabsTTS is a tool that generates an audio file from the given input text using the Eleven Labs API.
    """

    name: str = "elevenlabs_tts"
    description: str = (
        """Generates an audio file from the given input text using the Eleven Labs API.
        Returns the path of the generated audio file.
        """
    )
    args_schema: Type[BaseModel] = ElevenLabsTTSConfig

    def _run(self, config: ElevenLabsTTSConfig) -> str:
        """
        Generates an audio file from the given input text using the Eleven Labs API.
        :param config: The configuration for the ElevenLabsTTS tool.
        :return: The path of the generated audio file.
        """
        CHUNK_SIZE = 1024
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.api_key,
        }

        data = {
            "text": config.input_text,
            "model_id": config.model_id,
            "voice_settings": {
                "stability": config.stability,
                "similarity_boost": config.similarity_boost,
            },
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        with open(config.output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

        return config.output_file
