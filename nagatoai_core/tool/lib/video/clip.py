import os
from typing import Type
import ffmpeg
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings

from nagatoai_core.tool.abstract_tool import AbstractTool


class VideoClipConfig(BaseSettings, BaseModel):
    full_path: str = Field(
        ...,
        description="The full path to the original video file.",
    )

    output_path_prefix: str = Field(
        "",
        description="The prefix to prepend to the output path of the clipped video file.",
    )

    output_path_suffix: str = Field(
        "_clip",
        description="The suffix to append to the output path of the clipped video file.",
    )

    from_timestamp: float = Field(
        ...,
        description="The start timestamp to clip the video from (in seconds).",
    )

    to_timestamp: float = Field(
        ...,
        description="The end timestamp to clip the video to (in seconds).",
    )


class VideoClipTool(AbstractTool):
    """
    VideoClipTool is a tool that creates video clips using ffmpeg.
    """

    name: str = "video_clip_creator"
    description: str = (
        """Creates a video clip from a specified start and end timestamp using ffmpeg.
        Returns information about the clipping process and the output file.
        """
    )
    args_schema: Type[BaseModel] = VideoClipConfig

    def _run(self, config: VideoClipConfig) -> dict:
        """
        Creates a video clip using ffmpeg.
        :param config: The configuration for the VideoClipTool.
        :return: A dictionary containing information about the clipping process and the output file.
        """
        try:
            if not os.path.exists(config.full_path):
                return {"error": "Input file not found", "input_path": config.full_path}

            # Prepare the output file path
            input_dir, input_filename = os.path.split(config.full_path)
            input_name, input_ext = os.path.splitext(input_filename)
            output_filename = f"{config.output_path_prefix}{input_name}{config.output_path_suffix}{input_ext}"
            output_path = os.path.join(input_dir, output_filename)

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Calculate duration
            probe = ffmpeg.probe(config.full_path)
            duration = float(probe["streams"][0]["duration"])

            # Perform the clipping
            input_stream = ffmpeg.input(
                config.full_path,
                ss=config.from_timestamp,
                t=config.to_timestamp - config.from_timestamp,
            )
            output_stream = ffmpeg.output(input_stream, output_path, c="copy")
            ffmpeg.run(
                output_stream,
                overwrite_output=True,
                capture_stdout=True,
                capture_stderr=True,
            )

            # Get information about the output file
            output_probe = ffmpeg.probe(output_path)
            output_duration = float(output_probe["streams"][0]["duration"])

            return {
                "status": "success",
                "input_file": config.full_path,
                "output_file": output_path,
                "from_timestamp": config.from_timestamp,
                "to_timestamp": config.to_timestamp,
                "original_duration": duration,
                "clip_duration": output_duration,
                "file_size_bytes": os.path.getsize(output_path),
            }

        except ffmpeg.Error as e:
            return {
                "error": "FFmpeg error",
                "stdout": e.stdout.decode("utf8"),
                "stderr": e.stderr.decode("utf8"),
            }
        except Exception as e:
            raise RuntimeError(f"Error creating video clip: {str(e)}")
