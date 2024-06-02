import subprocess
from typing import Type

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings

from nagatoai_core.tool.abstract_tool import AbstractTool


class AfPlayConfig(BaseSettings, BaseModel):
    audio_file: str = Field(
        ...,
        description="The path to the audio file to play.",
    )


class AfPlayTool(AbstractTool):
    """
    AfplayTool is a tool that plays an audio file using the 'afplay' command-line utility.
    """

    name: str = "afplay"
    description: str = (
        """Plays the specified audio file using the 'afplay' command-line utility.
        Returns a message indicating the result of the audio playback.
        """
    )
    args_schema: Type[BaseModel] = AfPlayConfig

    def _run(self, config: AfPlayConfig) -> str:
        """
        Plays the specified audio file using the 'afplay' command-line utility.
        :param config: The configuration for the AfplayTool.
        :return: A message indicating the result of the audio playback.
        """
        try:
            return_code = subprocess.call(["afplay", config.audio_file])
            if return_code == 0:
                return f"Audio playback completed successfully: {config.audio_file}"
            else:
                return f"Audio playback failed with return code {return_code}: {config.audio_file}"
        except FileNotFoundError:
            return f"Error: Audio file not found: {config.audio_file}"
        except Exception as e:
            return f"Error playing audio file {config.audio_file}: {str(e)}"
