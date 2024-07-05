import os
import re
from typing import List, Optional, Type, Dict
from time import sleep
from enum import Enum
import uuid

from rich.console import Console
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from nagatoai_core.agent.factory import create_agent
from nagatoai_core.chain.chain import Chain, AgentParamConverter
from nagatoai_core.chain.ring import Ring
from nagatoai_core.chain.agent_link import AgentLink
from nagatoai_core.chain.tool_link import ToolLink
from nagatoai_core.chain.adapter_link import AdaptorLink
from nagatoai_core.tool.lib.human.input import HumanInputTool
from nagatoai_core.tool.lib.web.page_scraper import WebPageScraperTool
from nagatoai_core.tool.lib.web.serper_search import SerperSearchTool
from nagatoai_core.tool.lib.filesystem.text_file_reader import TextFileReaderTool
from nagatoai_core.tool.lib.filesystem.file_checker import FileCheckerTool
from nagatoai_core.tool.lib.time.time_offset import TimeOffsetTool
from nagatoai_core.tool.lib.time.time_now import TimeNowTool
from nagatoai_core.tool.lib.video.youtube.video_download import YouTubeVideoDownloadTool
from nagatoai_core.tool.lib.video.clip import VideoClipTool
from nagatoai_core.tool.lib.video.details_checker import VideoCheckerTool
from nagatoai_core.tool.lib.audio.stt.groq_whisper import GroqWhisperTool
from nagatoai_core.tool.lib.audio.stt.assemblyai import AssemblyAITranscriptionTool
from nagatoai_core.tool.lib.audio.stt.openai_whisper import OpenAIWhisperTool
from nagatoai_core.tool.lib.audio.tts.openai import OpenAITTSTool
from nagatoai_core.tool.lib.audio.video_to_mp3 import VideoToMP3Tool
from nagatoai_core.tool.lib.audio.tts.eleven_labs import ElevenLabsTTSTool
from nagatoai_core.tool.lib.audio.afplay import AfPlayTool


def main():
    """
    The program's main entry point.
    """
    load_dotenv()
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    # Create a list of tools
    yt_tool_link = ToolLink(
        id=str(uuid.uuid4()),
        name="YouTube Video Download",
        tool=YouTubeVideoDownloadTool,
    )

    input_data = {
        # "video_id": "rgeIb0saGbY",
        "video_id": "UakqL6Pj9xo",
        "output_path": "./",
        "file_name": "dwarkesh_chollet.mp4",
    }

    def v2mp3_param_conv(video_path: str) -> Dict:
        """
        Convert video path input into Dict for video to mp3 tool
        """
        return {"input_path": video_path, "output_path": "video_audio.mp3"}

    ad_link = AdaptorLink(
        name="Video to MP3 Adapter",
        adapter_fn=v2mp3_param_conv,
    )

    audio_extr_link = ToolLink(
        id=str(uuid.uuid4()), name="Audio Extraction", tool=VideoToMP3Tool
    )

    def transcribe_param_conv(audio_data: Dict) -> Dict:
        """
        Convert audio path input into Dict for transcription tool
        """
        return {"file_path": audio_data["output_file"]}

    transcribe_adapter = AdaptorLink(
        name="Transcription Adapter",
        adapter_fn=transcribe_param_conv,
    )

    # transcribe_link = ToolLink(
    #     id=str(uuid.uuid4()),
    #     name="Transcription",
    #     tool=GroqWhisperTool,
    # )

    transcribe_link = ToolLink(
        id=str(uuid.uuid4()),
        name="Transcription",
        tool=AssemblyAITranscriptionTool,
    )

    worker_agent = create_agent(
        anthropic_api_key,
        "claude-3-5-sonnet-20240620",
        "Worker agent",
        """
        You are a helpful assistant adept at analysing transcripts from podcasts and identifying the most important parts.
        """,
        "Worker",
    )

    wa_prompt = """Analyse the following transcript and identify the most important parts.
    Please return the key soundbites and the most important points in the following format.
    ---
    <key_moments>
        <key_moment>
            <transcript></transcript>
            <from_timestamp></from_timestamp>
            <to_timestamp></to_timestamp>
        </key_moment>
    </key_moments>
    ---
    Regarding the important points in the video, I would like you to focus on extracting contiguous sentences that cover these points:
    - what is the ARC challenges
    - why LLMs have not been able to solve the ARC challenge puzzles so far
    - what is the key insight that the LLMs are missing

    Feel free to add any further extracts that you think are important to the analysis.

    Put the text verbatim from the transcript inside the <transcript> tag.
    Put the start timestamp inside the <from_timestamp> tag and the end timestamp inside the <to_timestamp> tag.

    The soundbites can be composed of multiple contiguous sentences, and should ideally be shorter than 60 seconds.

    Analyse the transcript below:
    """

    ag_link = AgentLink(
        agent=worker_agent,
        input_prompt=wa_prompt,
    )

    def extract_key_moments_adapter(llm_output: str) -> List[Dict]:
        """
        Extract key moments from the transcript
        """
        soup = BeautifulSoup(llm_output, "html.parser")
        key_moments = soup.find_all("key_moment")
        key_moments_list = []
        for key_moment in key_moments:
            # tr = key_moment.find("transcript").text
            from_ts = key_moment.find("from_timestamp").text
            to_ts = key_moment.find("to_timestamp").text
            key_moments_list.append(
                {
                    # "transcript": tr,
                    "from_timestamp": from_ts,
                    "to_timestamp": to_ts,
                    "full_path": "./dwarkesh_chollet.mp4",
                    "output_path_prefix": f"clip_",
                    "output_path_suffix": f"_{from_ts}s_{to_ts}s",
                }
            )

        return key_moments_list

    km_adapter = AdaptorLink(
        name="Key Moments Adapter",
        adapter_fn=extract_key_moments_adapter,
    )

    def utterances_adapter(transcribe_output: Dict) -> Dict:
        """
        Convert utterances into a Dict for the Ring tool
        """
        return {
            # "utterances": transcribe_output["utterances"],
            "sentences": transcribe_output["sentences"],
            "file_name": transcribe_output["file_name"],
        }

    ut_adaptor = AdaptorLink(
        name="Utterances Adapter",
        adapter_fn=utterances_adapter,
    )

    rg_video_clip_link = ToolLink(
        id=str(uuid.uuid4()),
        name="Video Clip",
        tool=VideoClipTool,
    )

    conv_agent = create_agent(
        anthropic_api_key,
        "claude-3-5-sonnet-20240620",
        "Params converter agent",
        """
        You are a expert assistant adept at converting data from one format to another.
        """,
        "Params converter",
    )

    agnt_conv_link = AgentParamConverter(name="Agent Param Converter", agent=conv_agent)

    rg = Ring(agent_param_conv_link=agnt_conv_link, links=[rg_video_clip_link])

    # Uncomment the below to run the chain with explicit parameter conversion steps
    # ch = Chain(
    #     links=[
    #         yt_tool_link,
    #         ad_link,
    #         audio_extr_link,
    #         transcribe_adapter,
    #         transcribe_link,
    #         ag_link,
    #         km_adapter,
    #         rg,
    #     ],
    # )

    # Run this chain if you want to perform automatic parameter conversions using an agent
    ch = Chain(
        agent_param_conv_link=agnt_conv_link,
        links=[
            yt_tool_link,
            audio_extr_link,
            transcribe_link,
            ut_adaptor,
            ag_link,
            km_adapter,
            rg,
        ],
    )
    ch.run(input_data)


if __name__ == "__main__":
    main()
