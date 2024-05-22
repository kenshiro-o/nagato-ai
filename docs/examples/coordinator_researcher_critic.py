import os
import re
from typing import List, Optional, Type
from time import sleep
from enum import Enum

from rich.console import Console
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.factory import create_agent
from nagatoai_core.agent.message import Exchange, Message, Sender, ToolResult
from nagatoai_core.common.common import print_exchange, send_agent_request
from nagatoai_core.mission.mission import Mission, MissionStatus
from nagatoai_core.mission.task import Task, TaskStatus, TaskOutcome, TaskResult
from nagatoai_core.runner.single_agent_task_runner import SingleAgentTaskRunner
from nagatoai_core.runner.single_critic_evaluator import SingleCriticEvaluator
from nagatoai_core.tool.registry import ToolRegistry
from nagatoai_core.tool.abstract_tool import AbstractTool
from nagatoai_core.tool.lib.readwise.book_finder import (
    ReadwiseDocumentFinderTool,
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
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider
from nagatoai_core.tool.provider.anthropic import AnthropicToolProvider
from nagatoai_core.tool.provider.openai import OpenAIToolProvider
from nagatoai_core.prompt.templates import (
    OBJECTIVE_PROMPT,
    COORDINATOR_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_TASK_PROMPT_WITH_EXAMPLE,
    RESEARCHER_TASK_PROMPT_NO_EXAMPLE,
    RESEARCHER_TASK_PROMPT_WITH_PREVIOUS_UNSATISFACTORY_TASK_RESULT,
    CRITIC_SYSTEM_PROMPT,
)


# Deprecated
def process_task(
    task: Task,
    task_result: Optional[TaskResult],
    worker_agent: Agent,
    tool_registry: ToolRegistry,
    tool_provider: Type[AbstractToolProvider],
    console: Console,
) -> List[Exchange]:
    """
    Processes a task by sending it to the researcher and the critic.
    :param task: The task to process.
    :param worker_agent: the worker agent
    """
    all_tools = tool_registry.get_all_tools()
    all_available_tools: List[AbstractToolProvider] = []
    for tool in all_tools:
        t = tool()
        at = tool_provider(
            tool=t, name=t.name, description=t.description, args_schema=t.args_schema
        )
        all_available_tools.append(at)
    # print(f"*** All Available Tools: {all_available_tools}")
    exchanges: List[Exchange] = []

    # Send the task to the researcher
    # We only need to send the examples in the first prompt to the agent (i.e. it has empty history).
    #  - If the agent has already seen the examples, we can skip sending them again.
    task_prompt = ""

    if task_result and task_result.outcome != TaskOutcome.MEETS_REQUIREMENT:
        task_prompt = (
            RESEARCHER_TASK_PROMPT_WITH_PREVIOUS_UNSATISFACTORY_TASK_RESULT.format(
                goal=task.goal,
                outcome=task_result.outcome,
                evaluation=task_result.evaluation,
                description=task.description,
            )
        )
    else:
        if len(worker_agent.history) > 0:
            task_prompt = RESEARCHER_TASK_PROMPT_NO_EXAMPLE.format(
                goal=task.goal, description=task.description
            )
        else:
            task_prompt = RESEARCHER_TASK_PROMPT_WITH_EXAMPLE.format(
                goal=task.goal, description=task.description
            )

    worker_exchange = send_agent_request(
        worker_agent, task_prompt, all_available_tools, 0.6, 2000
    )
    exchanges.append(worker_exchange)

    print_exchange(console, worker_agent, worker_exchange, "blue")

    while True:
        if not worker_exchange.agent_response.tool_calls:
            # print("*** No tool calls to process for task ** breaking out of loop ***")
            break

        for tool_call in worker_exchange.agent_response.tool_calls:
            # print(
            #     f"*** Agent to call tool: {tool_call.name} with parameters: {tool_call.parameters}"
            # )
            tool = tool_registry.get_tool(tool_call.name)
            tool_instance = tool()
            tool_params_schema = tool_instance.args_schema
            tool_params = tool_params_schema(**tool_call.parameters)

            tool_output = tool_instance._run(tool_params)
            # print(f"*** Tool Output: {tool_output}")

            tool_result = ToolResult(
                id=tool_call.id, name=tool_call.name, result=tool_output, error=None
            )

            tool_result_exchange = worker_agent.send_tool_run_results(
                [tool_result], 0.6, 2000
            )
            exchanges.append(tool_result_exchange)

            # print(
            #     f"*** Assistant from Tool Call reply: {tool_result_exchange.agent_response}"
            # )
            print_exchange(console, worker_agent, tool_result_exchange, "orange_red1")

        worker_exchange = exchanges[-1]

    return exchanges


def main():
    """
    main represents the main entry point of the program.
    """
    load_dotenv()

    # For pretty console logs
    console = Console()

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    coordinator_agent: Agent = create_agent(
        anthropic_api_key,
        "claude-3-opus-20240229",
        "Coordinator",
        COORDINATOR_SYSTEM_PROMPT,
        "Coordinator Agent",
    )

    researcher_agent = create_agent(
        anthropic_api_key,
        "claude-3-sonnet-20240229",
        "Researcher",
        RESEARCHER_SYSTEM_PROMPT,
        "Researcher Agent",
    )

    critic_agent = create_agent(
        anthropic_api_key,
        "claude-3-haiku-20240307",
        "Critic",
        CRITIC_SYSTEM_PROMPT,
        "Critic Agent",
    )

    tool_registry = ToolRegistry()
    tool_registry.register_tool(ReadwiseDocumentFinderTool)
    tool_registry.register_tool(ReadwiseBookHighlightsListerTool)
    tool_registry.register_tool(HumanConfirmInputTool)
    tool_registry.register_tool(HumanInputTool)
    tool_registry.register_tool(WebPageScraperTool)
    tool_registry.register_tool(SerperSearchTool)
    tool_registry.register_tool(TextFileReaderTool)

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

    problem_statement = input("Please enter a problem statement for Nagato to solve:")

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
        description = task.find("description").get_text(strip=True)

        task = Task(goal=goal, description=description)
        tasks_list.append(task)

    mission = Mission(
        problem_statement=problem_statement, objective=objective, tasks=tasks_list
    )
    sleep(3)

    for task in mission.tasks:
        task_evaluator = SingleCriticEvaluator(critic_agent=critic_agent)

        task_runner = SingleAgentTaskRunner(
            previous_task=None,
            current_task=task,
            agents={"worker_agent": researcher_agent},
            tool_registry=tool_registry,
            agent_tool_providers={"worker_agent": AnthropicToolProvider},
            task_evaluator=task_evaluator,
        )

        task_runner.run()

        if task.result.outcome != TaskOutcome.MEETS_REQUIREMENT:
            # TODO - Should we exit the program if the task did not successfully complete?
            print(
                f"⚠️ Task {task} has not met the requirement. Moving on to next task..."
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
