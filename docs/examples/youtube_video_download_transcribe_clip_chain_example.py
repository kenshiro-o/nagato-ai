import os
from typing import List, Dict
import uuid

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from nagatoai_core.agent.factory import create_agent
from nagatoai_core.chain.chain import Chain, AgentParamConverter
from nagatoai_core.chain.ring import Ring
from nagatoai_core.chain.agent_link import AgentLink
from nagatoai_core.chain.tool_link import ToolLink
from nagatoai_core.chain.adapter_link import AdaptorLink
from nagatoai_core.tool.lib.video.youtube.video_download import YouTubeVideoDownloadTool
from nagatoai_core.tool.lib.video.clip import VideoClipTool
from nagatoai_core.tool.lib.audio.stt.assemblyai import AssemblyAITranscriptionTool
from nagatoai_core.tool.lib.audio.video_to_mp3 import VideoToMP3Tool
from nagatoai_core.prompt.template.prompt_template import PromptTemplate


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

    video_name = "pieter_levels_fridman.mp4"
    input_data = {
        "video_id": "oFtjKbXKqbg",
        "output_path": "./media/",
        "file_name": video_name,
    }

    # def v2mp3_param_conv(video_path: str) -> Dict:
    #     """
    #     Convert video path input into Dict for video to mp3 tool
    #     """
    #     return {"input_path": video_path, "output_path": "video_audio.mp3"}

    # ad_link = AdaptorLink(
    #     name="Video to MP3 Adapter",
    #     adapter_fn=v2mp3_param_conv,
    # )

    audio_extr_link = ToolLink(
        id=str(uuid.uuid4()), name="Audio Extraction", tool=VideoToMP3Tool
    )

    # def transcribe_param_conv(audio_data: Dict) -> Dict:
    #     """
    #     Convert audio path input into Dict for transcription tool
    #     """
    #     return {"file_path": audio_data["output_file"]}

    # transcribe_adapter = AdaptorLink(
    #     name="Transcription Adapter",
    #     adapter_fn=transcribe_param_conv,
    # )

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

    wa_prompt_template = PromptTemplate(
        template="""
        You are tasked with analyzing a transcript and identifying the most important parts. Your goal is to extract key soundbites and the most important points from the given transcript. Follow these instructions carefully:

        1. Review the following transcript sentences:
        <transcript_sentences>
        {sentences}
        </transcript_sentences>

        2. You will present your analysis in the following format:
        <key_moments>
            <key_moment>
                <transcript></transcript>
                <from_timestamp></from_timestamp>
                <to_timestamp></to_timestamp>
            </key_moment>
        </key_moments>

        3. Focus on extracting sentences that cover one or more of these points:
        - How to be productive as an indie hacker
        - What projects to work on
        - How to spot trends
        - How to build successful startups
        - How to use AI
        - Advice for entrepreneurs

        4. You may also include any other sentences that you deem important for the analysis.

        5. When selecting key moments:
        a. Copy the selected sentences verbatim from the transcript and place them inside the <transcript> tags.
        b. Put the start timestamp inside the <from_timestamp> tags.
        c. Put the end timestamp inside the <to_timestamp> tags.

        6. Guidelines for soundbites:
        - Soundbites can be composed of multiple contiguous sentences.
        - Aim for soundbites that are at most 90 seconds long.

        7. Present your final output as a series of <key_moment> entries, each containing the transcript, from_timestamp, and to_timestamp tags.

        Remember to carefully analyze the transcript and select the most relevant and impactful moments that align with the specified topics. Ensure that your chosen soundbites provide valuable insights and advice related to indie hacking, entrepreneurship, and startup success.
        """,
        data_light_schema={},
    )

    agent_soundbite_extractor_link = AgentLink(
        agent=worker_agent,
        input_prompt_template=wa_prompt_template,
    )

    def extract_key_moments_adapter(llm_output: str) -> List[Dict]:
        """
        Extract key moments from the transcript
        """
        print(f"Extracting key moments from transcript: {llm_output}")

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
                    "full_path": f"./media/{video_name}",
                    "output_path_prefix": f"clip_",
                    "output_path_suffix": f"_{from_ts}s_{to_ts}s",
                }
            )

        return key_moments_list

    km_adapter = AdaptorLink(
        name="Key Moments Adapter",
        adapter_fn=extract_key_moments_adapter,
    )

    def utterances_adapter_fn(transcribe_output: Dict) -> Dict:
        """
        Convert utterances into a Dict for the Ring tool
        """
        return {
            # "utterances": transcribe_output["utterances"],
            "sentences": transcribe_output["sentences"],
            "file_name": transcribe_output["file_name"],
        }

    utterances_adaptor = AdaptorLink(
        name="Utterances Adaptor",
        adapter_fn=utterances_adapter_fn,
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
            utterances_adaptor,
            agent_soundbite_extractor_link,
            km_adapter,
            rg,
        ],
    )
    ch.run(input_data)


if __name__ == "__main__":
    main()
