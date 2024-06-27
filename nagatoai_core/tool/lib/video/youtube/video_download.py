from typing import Type
import os
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
from pytube import YouTube

from nagatoai_core.tool.abstract_tool import AbstractTool


class YouTubeVideoDownloadConfig(BaseSettings, BaseModel):
    video_id: str = Field(
        ...,
        description="The ID of the YouTube video to download.",
    )

    output_path: str = Field(
        ".",
        description="The output directory path to save the downloaded video.",
    )

    file_name: str = Field(
        None,
        description="The name of the file to save the video as. If not provided, the video's title will be used.",
    )


class YouTubeVideoDownloadTool(AbstractTool):
    """
    YouTubeVideoDownloadTool is a tool that downloads a video from YouTube given the video ID.
    """

    name: str = "youtube_video_download"
    description: str = (
        """Downloads a video from YouTube given the video ID and saves it to the specified output path with the specified file name.
        Returns the full path of the downloaded video file.
        """
    )
    args_schema: Type[BaseModel] = YouTubeVideoDownloadConfig

    def _run(self, config: YouTubeVideoDownloadConfig) -> str:
        """
        Downloads a video from YouTube using the given video ID and saves it to the specified output path with the specified file name.
        :param config: The configuration for the YouTubeVideoDownloadTool.
        :return: The full path of the downloaded video file.
        """
        try:
            # Construct the full YouTube URL
            video_url = f"https://www.youtube.com/watch?v={config.video_id}"

            # Create a YouTube object
            yt = YouTube(video_url)

            # Get the highest resolution progressive stream
            video = yt.streams.get_highest_resolution()

            # Determine the file name
            if config.file_name:
                file_name = config.file_name
                if not file_name.endswith(".mp4"):
                    file_name += ".mp4"
            else:
                file_name = f"{yt.title}.mp4"

            # Ensure the file name is valid
            file_name = "".join(
                c for c in file_name if c.isalnum() or c in (" ", "-", "_", ".")
            ).rstrip()

            # Construct the full output path
            full_path = os.path.join(config.output_path, file_name)

            # Download the video
            video.download(output_path=config.output_path, filename=file_name)

            return full_path
        except Exception as e:
            raise RuntimeError(f"Error downloading YouTube video: {str(e)}")
