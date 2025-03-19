#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of a graph composed of tool nodes and an agent node for processing YouTube videos.

This script sets up a graph with three nodes:
1. YouTube video download tool node
2. Groq Whisper transcription tool node
3. Gemini Flash agent node for summarization

The graph processes a YouTube video by downloading it, transcribing it, and then
generating a summary of the transcript.
"""

# Standard Library
import os
from typing import Dict, List
import traceback

# Third Party
from dotenv import load_dotenv

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
        model="gemini-1.5-flash",
        role="Parameter Converter",
        role_description="You are a helpful assistant that converts parameters between different formats.",
        nickname="Converter",
    )

    # Agent for summary generation
    summary_agent = create_agent(
        api_key=google_api_key,
        model="gemini-2.0-flash",
        role="Video Summarizer",
        role_description="""You are an expert at summarizing video transcripts.
        When given a transcript, provide a concise yet comprehensive summary of the main points,
        key insights, and important details. Your summary should capture the essence of the content
        in a well-structured format.""",
        nickname="Summarizer",
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

    # 3. Summary generation node with Gemini
    summary_node = AgentNode(
        id="summary_node",
        agent=summary_agent,
        prompt_template=PromptTemplate(
            template="""{inputs[0].result}

            Please summarize the above transcript in a concise, well-structured manner.
            Highlight the key points, main insights, and important information:

            Focus on the following key points: {inputs[1].result}
            """,
            data_light_schema={},
        ),
        temperature=0.3,
        max_tokens=1000,
        tools=[],
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
            result=HumanInputConfig(message="Enter what what you would like he summary to focus on."),
            step=inputs[0].step + 1,
        )

        flow_result = flow.execute([flow_input])
        return inputs + flow_result

    sequential_human_input_flow = SequentialFlow(
        id="sequential_human_input_flow",
        nodes=[tool_input_flow_param],
    )

    sequential_summary_flow = SequentialFlow(
        id="sequential_summary_flow",
        nodes=[summary_node],
    )

    transformer_flow = TransformerFlow(
        id="transformer_flow",
        flow_param=sequential_human_input_flow,
        functor=functor_human_input_tool,
    )

    # Create the graph and establish directed edges between nodes
    graph = Graph()

    # Add edges to create a directed acyclic graph (DAG)
    # Step 1: YouTube download and transcription flow → Transformer flow (human input)
    graph.add_edge(sequential_yt_dl_transcribe_flow, transformer_flow)

    # Step 2: Transformer flow (human input) → Summary flow
    graph.add_edge(transformer_flow, sequential_summary_flow)

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

        # The results variable contains the output from the last node (summary node)
        if results and len(results) > 0:
            if results[0].error:
                print(f"Error during execution: {results[0].error}")
            else:
                # Display the summary results
                result = results[0]
                if hasattr(result.result, "agent_response") and hasattr(result.result.agent_response, "content"):
                    print(f"Summary:\n{result.result.agent_response.content}")
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
