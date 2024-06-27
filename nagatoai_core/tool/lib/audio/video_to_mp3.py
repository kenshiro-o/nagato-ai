import os
from typing import Type
import ffmpeg
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings

from nagatoai_core.tool.abstract_tool import AbstractTool


class VideoToMP3Config(BaseSettings, BaseModel):
    input_path: str = Field(
        ...,
        description="The full path to the input video file.",
    )

    output_path: str = Field(
        ...,
        description="The full path for the output MP3 file.",
    )

    sampling_rate: int = Field(
        16000,
        description="The sampling rate in Hz for the output MP3 file. Default is 16000 Hz.",
    )


class VideoToMP3Tool(AbstractTool):
    """
    VideoToMP3Tool is a tool that converts a video file to an MP3 audio file.
    """

    name: str = "video_to_mp3_converter"
    description: str = (
        """Converts a video file to an MP3 audio file with specified sampling rate.
        Returns information about the conversion process.
        """
    )
    args_schema: Type[BaseModel] = VideoToMP3Config

    def _run(self, config: VideoToMP3Config) -> dict:
        """
        Converts a video file to an MP3 audio file.
        :param config: The configuration for the VideoToMP3Tool.
        :return: A dictionary containing information about the conversion process.
        """
        try:
            if not os.path.exists(config.input_path):
                return {
                    "error": "Input file not found",
                    "input_path": config.input_path,
                }

            output_dir = os.path.dirname(config.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                # If no directory is specified, use the current working directory
                config.output_path = os.path.join(os.getcwd(), config.output_path)

            # Perform the conversion
            (
                ffmpeg.input(config.input_path)
                .output(
                    config.output_path, acodec="libmp3lame", ar=config.sampling_rate
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            # Get information about the output file
            probe = ffmpeg.probe(config.output_path)
            audio_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "audio"
                ),
                None,
            )

            return {
                "status": "success",
                "input_file": os.path.basename(config.input_path),
                "output_file": os.path.basename(config.output_path),
                "sampling_rate": config.sampling_rate,
                "duration": float(probe["format"]["duration"]),
                "file_size_bytes": int(probe["format"]["size"]),
                "audio_codec": (
                    audio_stream["codec_name"] if audio_stream else "Unknown"
                ),
                "channels": audio_stream["channels"] if audio_stream else "Unknown",
            }

        except ffmpeg.Error as e:
            return {
                "error": "FFmpeg error",
                "stdout": e.stdout.decode("utf8"),
                "stderr": e.stderr.decode("utf8"),
            }
        except Exception as e:
            raise RuntimeError(f"Error converting video to MP3: {str(e)}")
