import os
import re
from typing import List
from time import sleep

from rich.console import Console
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.factory import create_agent, get_agent_tool_provider
from nagatoai_core.common.common import print_exchange, send_agent_request
from nagatoai_core.mission.mission import Mission
from nagatoai_core.mission.task import Task, TaskOutcome
from nagatoai_core.runner.single_agent_task_runner import SingleAgentTaskRunner
from nagatoai_core.runner.single_critic_evaluator import SingleCriticEvaluator
from nagatoai_core.tool.registry import ToolRegistry
from nagatoai_core.tool.lib.readwise.book_finder import (
    ReadwiseDocumentFinderTool,
)
from nagatoai_core.tool.lib.readwise.highlights_lister import (
    ReadwiseHighightsListerTool,
)
from nagatoai_core.tool.lib.readwise.book_highlights_lister import (
    ReadwiseBookHighlightsListerTool,
)
from nagatoai_core.tool.lib.human.confirm import (
    HumanConfirmInputTool,
)
from nagatoai_core.tool.lib.human.input import HumanInputTool
from nagatoai_core.tool.lib.web.page_scraper import WebPageScraperTool
from nagatoai_core.tool.lib.web.serper_search import SerperSearchTool
from nagatoai_core.tool.lib.filesystem.text_file_reader import TextFileReaderTool
from nagatoai_core.tool.lib.filesystem.text_file_writer import TextFileWriterTool
from nagatoai_core.tool.lib.filesystem.file_checker import FileCheckerTool
from nagatoai_core.tool.lib.time.time_offset import TimeOffsetTool
from nagatoai_core.tool.lib.time.time_now import TimeNowTool
from nagatoai_core.tool.lib.video.youtube.video_download import YouTubeVideoDownloadTool
from nagatoai_core.tool.lib.video.details_checker import VideoCheckerTool
from nagatoai_core.tool.lib.audio.stt.groq_whisper import GroqWhisperTool
from nagatoai_core.tool.lib.audio.stt.assemblyai import AssemblyAITranscriptionTool
from nagatoai_core.tool.lib.audio.stt.openai_whisper import OpenAIWhisperTool
from nagatoai_core.tool.lib.audio.tts.openai import OpenAITTSTool
from nagatoai_core.tool.lib.audio.video_to_mp3 import VideoToMP3Tool
from nagatoai_core.tool.lib.audio.tts.eleven_labs import ElevenLabsTTSTool
from nagatoai_core.tool.lib.audio.afplay import AfPlayTool
from nagatoai_core.prompt.templates import (
    OBJECTIVE_PROMPT,
    COORDINATOR_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
)


def main():
    """
    main represents the main entry point of the program.
    """
    load_dotenv()

    # For pretty console logs
    console = Console()

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    coordinator_agent: Agent = create_agent(
        anthropic_api_key,
        "claude-3-5-sonnet-20240620",
        "Coordinator",
        COORDINATOR_SYSTEM_PROMPT,
        "Coordinator Agent",
    )

    researcher_agent = create_agent(
        anthropic_api_key,
        "claude-3-5-sonnet-20240620",
        "Researcher",
        RESEARCHER_SYSTEM_PROMPT,
        "Researcher Agent",
    )

    # researcher_agent = create_agent(
    #     google_api_key,
    #     "gemini-1.5-flash",
    #     "Researcher",
    #     RESEARCHER_SYSTEM_PROMPT,
    #     "Researcher Agent",
    # )

    # researcher_agent = create_agent(
    #     openai_api_key,
    #     "gpt-4o-mini-2024-07-18",
    #     "Researcher",
    #     RESEARCHER_SYSTEM_PROMPT,
    #     "Researcher Agent",
    # )

    # researcher_agent = create_agent(
    #     groq_api_key,
    #     "llama-3.1-70b-versatile",
    #     "Researcher",
    #     RESEARCHER_SYSTEM_PROMPT,
    #     "Researcher Agent",
    # )

    # critic_agent = create_agent(
    #     anthropic_api_key,
    #     "claude-3-haiku-20240307",
    #     "Critic",
    #     CRITIC_SYSTEM_PROMPT,
    #     "Critic Agent",
    # )

    # critic_agent = create_agent(
    #     google_api_key,
    #     "gemini-1.5-flash",
    #     "Critic",
    #     CRITIC_SYSTEM_PROMPT,
    #     "Critic Agent",
    # )

    critic_agent = create_agent(
        openai_api_key,
        "gpt-4o-mini-2024-07-18",
        "Critic",
        CRITIC_SYSTEM_PROMPT,
        "Critic Agent",
    )

    tool_registry = ToolRegistry()
    tool_registry.register_tool(ReadwiseDocumentFinderTool)
    tool_registry.register_tool(ReadwiseHighightsListerTool)
    tool_registry.register_tool(ReadwiseBookHighlightsListerTool)
    tool_registry.register_tool(HumanConfirmInputTool)
    tool_registry.register_tool(HumanInputTool)
    tool_registry.register_tool(WebPageScraperTool)
    tool_registry.register_tool(SerperSearchTool)
    tool_registry.register_tool(FileCheckerTool)
    tool_registry.register_tool(TextFileReaderTool)
    tool_registry.register_tool(TextFileWriterTool)
    tool_registry.register_tool(TimeNowTool)
    tool_registry.register_tool(TimeOffsetTool)
    tool_registry.register_tool(VideoCheckerTool)
    tool_registry.register_tool(YouTubeVideoDownloadTool)
    tool_registry.register_tool(GroqWhisperTool)
    tool_registry.register_tool(OpenAITTSTool)
    tool_registry.register_tool(AssemblyAITranscriptionTool)
    tool_registry.register_tool(VideoToMP3Tool)
    tool_registry.register_tool(OpenAIWhisperTool)
    tool_registry.register_tool(ElevenLabsTTSTool)
    tool_registry.register_tool(AfPlayTool)

    tools_available_str = ""
    for tool in tool_registry.get_all_tools():
        tool_instance = tool()
        tools_available_str += f"""
        <tool>
            <name>{tool_instance.name}</name>
            <description>{tool_instance.description}</description>
        </tool>
        """

    if tools_available_str:
        tools_available_str = (
            f"<tools_available>{tools_available_str}</tools_available>"
        )

    console.print(
        '[bold yellow]Please input a problem statement for Nagato to solve then press the "Enter" key twice:[/bold yellow]'
    )
    user_input = []

    while True:
        line = console.input()
        if line == "":
            break
        user_input.append(line)

    problem_statement = "\n".join(user_input)

    problem_statement_prompt = OBJECTIVE_PROMPT.format(
        problem_statement=problem_statement,
        tools_available=tools_available_str,
    )

    coordinator_exchange = send_agent_request(
        coordinator_agent, problem_statement_prompt, [], 0.6, 2000
    )

    soup = BeautifulSoup(coordinator_exchange.agent_response.content, "html.parser")
    objective = soup.find("objective").get_text(strip=True)
    tasks = soup.find_all("task")

    print_exchange(console, coordinator_agent, coordinator_exchange, "purple")

    tasks_list: List[Task] = []
    for task in tasks:
        goal = task.find("goal").get_text(strip=True)
        description = (
            task.find("description").get_text(strip=True)
            if task.find("description")
            else ""
        )

        # Find all the tools recommended in tools_recommended
        tools_recommended = task.find_all("tool")
        tools_recs = []
        for tool in tools_recommended:
            tools_recs.append(tool.get_text(strip=True))

        task = Task(goal=goal, description=description, tools_recommended=tools_recs)
        tasks_list.append(task)

    mission = Mission(
        problem_statement=problem_statement, objective=objective, tasks=tasks_list
    )
    sleep(3)

    for task in mission.tasks:
        console.print(f"[bright_cyan] \n\n🤖 Current task: {task}\n\n[/bright_cyan]")
        task_evaluator = SingleCriticEvaluator(critic_agent=critic_agent)

        task_runner = SingleAgentTaskRunner(
            previous_task=None,
            current_task=task,
            agents={"worker_agent": researcher_agent},
            tool_registry=tool_registry,
            agent_tool_providers={
                "worker_agent": get_agent_tool_provider(researcher_agent)
            },
            task_evaluator=task_evaluator,
        )

        task_runner.run()

        if task.result.outcome != TaskOutcome.MEETS_REQUIREMENT:
            # TODO - Should we exit the program if the task did not successfully complete?
            console.print(
                f"[bright_red]⚠️ Task {task} has not met the requirement. Moving on to next task...[/bright_red]"
            )

    markdown_output_str = f"# {mission.objective}\n\n"
    markdown_output_str += f"## Problem Statement\n\n{mission.problem_statement}\n\n"
    markdown_output_str += f"## Tasks\n\n"

    for task in mission.tasks:
        markdown_output_str += f"### {task.goal}\n\n"
        markdown_output_str += f"### Result\n\n{task.result.result}\n\n"

        markdown_output_str += f"### Result Assessment\n\n"
        markdown_output_str += (
            f"**Verdict**: {task.result.outcome.name.replace('_', ' ')}\n\n"
        )
        if task.result.outcome != TaskOutcome.MEETS_REQUIREMENT:
            markdown_output_str += f"**Feedback**: {task.result.evaluation}\n\n"

    # print_exchange(console, markdown_output_agent, markdown_exchange, "green3")

    # Sanitize/escape the objective string so that we can turn it into a file name
    objective_file_name = re.sub(r"[^a-zA-Z0-9]+", "_", objective)
    # Make sure the fle name is not longer than 25 characters
    objective_file_name = objective_file_name[:25] + ".md"

    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")

    # Write the agent's response in a markdown file
    with open(f"./outputs/{objective_file_name}", "w") as f:
        f.write(markdown_output_str)


if __name__ == "__main__":
    main()
