"""
Example of a graph composed of tool nodes and an agent node for processing YouTube videos.

This script sets up a graph with several nodes:
1. YouTube video download tool node
2. Groq Whisper transcription tool node
3. Human input tool node
4. Highlight generation agent node
5. Video path augmentation transformer
6. Unfold flow for processing multiple highlights
7. Parallel video clipping flow

The graph processes a YouTube video by:
1. Downloading it
2. Transcribing it
3. Getting user input on what to focus on
4. Generating key highlights from the transcript
5. Augmenting highlights with video path information
6. Unfolding the highlights list into individual items
7. Creating video clips in parallel for each highlight
"""

# Standard Library
import os
from typing import List
import traceback

# Third Party
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.agent.factory import create_agent
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.agent_node import AgentNode
from nagatoai_core.graph.tool_node_with_params_conversion import ToolNodeWithParamsConversion
from nagatoai_core.graph.types import NodeResult
from nagatoai_core.tool.lib.video.youtube.video_download import YouTubeVideoDownloadTool
from nagatoai_core.tool.lib.audio.stt.groq_whisper import GroqWhisperTool
from nagatoai_core.tool.provider.openai import OpenAIToolProvider
from nagatoai_core.graph.sequential_flow import SequentialFlow
from nagatoai_core.prompt.template.prompt_template import PromptTemplate
from nagatoai_core.graph.transformer_flow import TransformerFlow
from nagatoai_core.tool.lib.human.input import HumanInputTool, HumanInputConfig
from nagatoai_core.graph.tool_node import ToolNode
from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.unfold_flow import UnfoldFlow
from nagatoai_core.tool.lib.video.clip import VideoClipTool
from nagatoai_core.graph.parallel_flow import ParallelFlow


def main():
    """
    Main function to create and execute the graph for YouTube video processing.
    """
    # Set up API keys from environment
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")

    # Input data for the graph
    input_data = {"video_id": "xZX0vOqWsC8", "file_path": "./claude-3.7-sonnet-review.mp4"}

    # Initialize tools
    youtube_download_tool = YouTubeVideoDownloadTool()
    groq_whisper_tool = GroqWhisperTool()

    # Set the Groq API key for the whisper tool
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Create tool providers using OpenAIToolProvider
    youtube_download_tool_provider = OpenAIToolProvider(
        name=youtube_download_tool.name,
        description=youtube_download_tool.description,
        args_schema=youtube_download_tool.args_schema,
        tool=youtube_download_tool,
    )
    groq_whisper_tool_provider = OpenAIToolProvider(
        name=groq_whisper_tool.name,
        description=groq_whisper_tool.description,
        args_schema=groq_whisper_tool.args_schema,
        tool=groq_whisper_tool,
    )

    # Create agents
    # Agent for parameter conversion in tool nodes
    conversion_agent = create_agent(
        api_key=google_api_key,
        model="gemini-2.0-flash",
        role="Parameter Converter",
        role_description="You are a helpful assistant that converts parameters between different formats.",
        nickname="Converter",
    )

    # Agent for summary generation
    highlights_agent = create_agent(
        api_key=google_api_key,
        model="gemini-2.0-flash",
        role="Video Highlights Identifier",
        role_description="""You are an expert at identifying highlights in a video transcript.
        When given a transcript, provide a list of highlights that are relevant to the user's query.
        The highlights should be concise and to the point, and should capture the essence of the content
        in a well-structured format.
        Highlight length MUST BE at least 30 seconds and MUST NOT exceed 120 seconds (i.e. difference between end_offset_seconds and start_offset_seconds should not be between 30 and 120)
        """,
        nickname="Highlighter",
    )

    # Create nodes for the graph
    # 1. YouTube video download node
    youtube_download_node = ToolNodeWithParamsConversion(
        id="youtube_download_node",
        tool_provider=youtube_download_tool_provider,
        agent=conversion_agent,
        retries=2,
    )

    # 2. Groq Whisper transcription node
    groq_whisper_node = ToolNodeWithParamsConversion(
        id="groq_whisper_node",
        tool_provider=groq_whisper_tool_provider,
        agent=conversion_agent,
        retries=2,
    )

    sequential_yt_dl_transcribe_flow = SequentialFlow(
        id="sequential_yt_dl_transcribe_flow",
        nodes=[youtube_download_node, groq_whisper_node],
    )

    human_input_tool = HumanInputTool()

    tool_input_flow_param = ToolNode(
        id="tool_input_flow_param",
        tool_provider=OpenAIToolProvider(
            name=human_input_tool.name,
            description=human_input_tool.description,
            args_schema=human_input_tool.args_schema,
            tool=human_input_tool,
        ),
    )

    def functor_human_input_tool(inputs: List[NodeResult], flow: AbstractFlow) -> List[NodeResult]:
        flow_input = NodeResult(
            node_id=flow.id,
            result=HumanInputConfig(message="Enter what you would like the clips to focus on."),
            step=inputs[0].step + 1,
        )

        flow_result = flow.execute([flow_input])
        return inputs + flow_result

    sequential_human_input_flow = SequentialFlow(
        id="sequential_human_input_flow",
        nodes=[tool_input_flow_param],
    )

    class Highlight(BaseModel):
        start_offset_seconds: float = Field(
            description="The start offset of the highlight in the transcript in seconds."
        )
        end_offset_seconds: float = Field(description="The end offset of the highlight in the transcript in seconds.")
        transcript_text: str = Field(description="The transcript text of the highlight.")
        reason: str = Field(description="The reason for selecting the highlight.")

    class Highlights(BaseModel):
        highlights: List[Highlight] = Field(description="A list of highlights.")

    # 3. Highlights creation node with Gemini
    highlight_agent_node = AgentNode(
        id="highlight_agent_node",
        agent=highlights_agent,
        prompt_template=PromptTemplate(
            template="""{inputs[0].result}

            Please identify the highlights in the above transcript.
            Highlight the key points, main insights, and important information:
            Extract up to 5 highlights.

            Focus on the following key points: {inputs[1].result}
            """,
            data_light_schema={},
        ),
        temperature=0.3,
        max_tokens=10000,
        tools=[],
        output_schema=Highlights,
    )

    transformer_flow = TransformerFlow(
        id="transformer_flow",
        flow_param=sequential_human_input_flow,
        functor=functor_human_input_tool,
    )

    def video_path_augment_functor(inputs: List[NodeResult], flow: AbstractFlow) -> List[NodeResult]:
        # Get the video path from the input
        video_path = input_data["file_path"]

        # Extract the highlights from the input
        highlights_obj: Highlights = inputs[0].result

        highlights_with_video_path = [
            {
                "video_path": video_path,
                "highlight": highlight,
                "highlight_index": i,
            }
            for i, highlight in enumerate(highlights_obj.highlights)
        ]

        # Create a new input with the video path and highlights
        return [NodeResult(node_id=flow.id, result=highlights_with_video_path, step=inputs[0].step + 1)]

    transformer_flow_video_path_augment = TransformerFlow(
        id="transformer_flow_video_path_augment",
        flow_param=sequential_human_input_flow,  # Random - as this flow is not used
        functor=video_path_augment_functor,
    )

    unfold_flow = UnfoldFlow(id="unfold_flow")

    video_clip_tool = VideoClipTool()
    youtube_video_clip_tool_provider = OpenAIToolProvider(
        name=video_clip_tool.name,
        description=video_clip_tool.description,
        args_schema=video_clip_tool.args_schema,
        tool=video_clip_tool,
    )

    youtube_video_clip_tool = ToolNodeWithParamsConversion(
        id="youtube_video_clip_tool",
        tool_provider=youtube_video_clip_tool_provider,
        agent=conversion_agent,
        retries=2,
    )

    parallel_youtube_clip_flow = ParallelFlow(
        id="parallel_youtube_clip_flow",
        nodes=[youtube_video_clip_tool],
    )

    # Create the graph and establish directed edges between nodes
    graph = Graph()

    # Add edges to create a directed acyclic graph (DAG)
    # Step 1: YouTube download and transcription flow → Transformer flow (human input)
    graph.add_edge(sequential_yt_dl_transcribe_flow, transformer_flow)

    # Step 2: Transformer flow (human input) → Highlight agent node
    graph.add_edge(transformer_flow, highlight_agent_node)

    # Step 3: Highlight agent node → Transformer flow for video path augmentation
    graph.add_edge(highlight_agent_node, transformer_flow_video_path_augment)

    # Step 4: Transformer flow for video path augmentation → Unfold flow
    graph.add_edge(transformer_flow_video_path_augment, unfold_flow)

    # Step 5: Unfold flow → Parallel YouTube clip flow
    graph.add_edge(unfold_flow, parallel_youtube_clip_flow)

    # Compile the graph to verify it's a valid DAG and compute execution order
    graph.compile()

    # Create initial input for the graph
    initial_input = [NodeResult(node_id="input", result=input_data, step=0)]

    # Execute the graph
    print("Starting graph execution...")
    try:
        # Run the graph - this will execute all nodes in sequence
        results = graph.run(initial_input)

        # Only display the final results
        print("\n==== Graph Execution Results ====")

        # The results variable contains the output from the last node
        if results and len(results) > 0:
            if results[0].error:
                print(f"Error during execution: {results[0].error}")
            else:
                # Display the results
                result = results[0]
                if hasattr(result.result, "agent_response") and hasattr(result.result.agent_response, "content"):
                    print(f"Clips generated based on highlights:\n{result.result.agent_response.content}")
                else:
                    print(f"Result: {result.result}")
        else:
            print("No results were returned from the graph execution.")

        print("\nGraph execution completed successfully!")
    except Exception as e:
        print(f"Error executing graph: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    load_dotenv()
    print("*** Environment variables ***")
    print(f"*** Groq API key: {os.environ.get('GROQ_API_KEY')}")
    print(f"*** Google API key: {os.environ.get('GOOGLE_API_KEY')}")
    # Check for required environment variables
    required_envs = ["GROQ_API_KEY", "GOOGLE_API_KEY"]
    # missing_envs = [env for env in required_envs if not os.environ.get(env)]
    missing_envs = [env for env in required_envs if not os.getenv(env)]

    if missing_envs:
        print(f"Error: Missing required environment variables: {', '.join(missing_envs)}")
        print("Please set the following environment variables:")
        print("  - GROQ_API_KEY: Your Groq API key for transcription")
        print("  - GOOGLE_API_KEY: Your Google API key for summarization with Gemini")
        print("\nExample usage:")
        print("  export GROQ_API_KEY=your_groq_api_key")
        print("  export GOOGLE_API_KEY=your_google_api_key")
        print("  python youtube_transcription_summary_graph.py")
        exit(1)

    # Run the main function
    main()
