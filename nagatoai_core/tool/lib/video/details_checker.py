import os
from typing import Type
import json
import subprocess
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings

from nagatoai_core.tool.abstract_tool import AbstractTool


class VideoCheckerConfig(BaseSettings, BaseModel):
    full_path: str = Field(
        ...,
        description="The full path of the video file to check.",
    )


class VideoCheckerTool(AbstractTool):
    """
    VideoCheckerTool is a tool that checks the details of a video file using ffprobe.
    """

    name: str = "video_checker"
    description: str = (
        """Checks the details of a video file at the specified full path using ffprobe.
        Returns a dictionary containing various properties of the video file.
        """
    )
    args_schema: Type[BaseModel] = VideoCheckerConfig

    def _run(self, config: VideoCheckerConfig) -> dict:
        """
        Checks the details of a video file at the given path using ffprobe.
        :param config: The configuration for the VideoCheckerTool.
        :return: A dictionary containing detailed information about the video file.
        """
        try:
            if not os.path.exists(config.full_path):
                return {"exists": False, "error": "File not found"}

            # Construct the ffprobe command
            command = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                config.full_path,
            ]

            # Run the ffprobe command
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                return {
                    "error": "Failed to analyze video file",
                    "details": result.stderr,
                }

            # Parse the JSON output
            video_info = json.loads(result.stdout)

            # Extract relevant information
            formatted_info = {
                "exists": True,
                "filename": os.path.basename(config.full_path),
                "format": video_info.get("format", {}).get("format_name"),
                "duration": float(video_info.get("format", {}).get("duration", 0)),
                "size_bytes": int(video_info.get("format", {}).get("size", 0)),
                "bit_rate": int(video_info.get("format", {}).get("bit_rate", 0)),
                "streams": [],
            }

            for stream in video_info.get("streams", []):
                stream_info = {
                    "codec_type": stream.get("codec_type"),
                    "codec_name": stream.get("codec_name"),
                }
                if stream.get("codec_type") == "video":
                    stream_info.update(
                        {
                            "width": stream.get("width"),
                            "height": stream.get("height"),
                            "fps": eval(stream.get("avg_frame_rate", "0/1")),
                        }
                    )
                elif stream.get("codec_type") == "audio":
                    stream_info.update(
                        {
                            "sample_rate": stream.get("sample_rate"),
                            "channels": stream.get("channels"),
                        }
                    )
                formatted_info["streams"].append(stream_info)

            return formatted_info

        except Exception as e:
            raise RuntimeError(f"Error checking video file: {str(e)}")
