# Standard Library
import os
import re
from typing import Optional, Type

# Third Party
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pytubefix import YouTube, cipher
from pytubefix.cli import on_progress

# Nagato AI
# Company Libraries
from nagatoai_core.tool.abstract_tool import AbstractTool

# from pytube import YouTube
# from pytube.innertube import _default_clients
# from pytube import cipher
# from pytube.exceptions import RegexMatchError


# _default_clients["ANDROID"]["context"]["client"]["clientVersion"] = "19.08.35"
# _default_clients["IOS"]["context"]["client"]["clientVersion"] = "19.08.35"
# _default_clients["ANDROID_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
# _default_clients["IOS_EMBED"]["context"]["client"]["clientVersion"] = "19.08.35"
# _default_clients["IOS_MUSIC"]["context"]["client"]["clientVersion"] = "6.41"
# _default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]


def get_throttling_function_name(js: str, js_url: str) -> str:
    """Extract the name of the function that computes the throttling parameter.

    :param str js:
        The contents of the base.js asset file.
    :rtype: str
    :returns:
        The name of the function used to compute the throttling parameter.
    """
    function_patterns = [
        r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*' r"\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)",
        r"\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])\([a-z]\)",
    ]
    # logger.debug('Finding throttling function name')
    for pattern in function_patterns:
        regex = re.compile(pattern)
        function_match = regex.search(js)
        if function_match:
            # logger.debug("finished regex search, matched: %s", pattern)
            if len(function_match.groups()) == 1:
                return function_match.group(1)
            idx = function_match.group(2)
            if idx:
                idx = idx.strip("[]")
                array = re.search(
                    r"var {nfunc}\s*=\s*(\[.+?\]);".format(nfunc=re.escape(function_match.group(1))),
                    js,
                )
                if array:
                    array = array.group(1).strip("[]").split(",")
                    array = [x.strip() for x in array]
                    return array[int(idx)]

    # raise RegexMatchError(caller="get_throttling_function_name", pattern="multiple")
    raise RuntimeError("Failed to find throttling function name")


# cipher.get_throttling_function_name = get_throttling_function_name


class YouTubeVideoDownloadConfig(BaseSettings, BaseModel):
    video_id: str = Field(
        ...,
        description="The ID of the YouTube video to download.",
    )

    output_folder: str = Field(
        ".",
        description="The output directory path to save the downloaded video.",
    )

    file_name: str = Field(
        ...,
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

    po_token_file: Optional[str] = None

    def __init__(self):
        super().__init__()

    def _run(self, config: YouTubeVideoDownloadConfig) -> str:
        """
        Downloads a video from YouTube using the given video ID and saves it to the specified output path with the specified file name.
        :param config: The configuration for the YouTubeVideoDownloadTool.
        :return: The full path of the downloaded video file.
        """
        try:
            # Modify the YouTube object creation to use po_token
            if self.po_token_file:
                yt = YouTube(
                    f"https://www.youtube.com/watch?v={config.video_id}",
                    "WEB",
                    use_po_token=True,
                    token_file=self.po_token_file,
                )
            else:
                yt = YouTube(f"https://www.youtube.com/watch?v={config.video_id}")

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
            file_name = "".join(c for c in file_name if c.isalnum() or c in (" ", "-", "_", ".")).rstrip()

            # Construct the full output path
            full_path = os.path.join(config.output_folder, file_name)

            # Download the video
            video.download(output_path=config.output_folder, filename=file_name)

            return {"full_path": full_path}
        except Exception as e:
            raise RuntimeError(f"Error downloading YouTube video: {str(e)}")
