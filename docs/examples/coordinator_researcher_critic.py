import os
import re
from typing import List, Optional, Type
from time import sleep
from enum import Enum
from datetime import datetime, timezone

from openai import OpenAI
from groq import Groq
from anthropic import Anthropic
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.openai import OpenAIAgent
from nagatoai_core.agent.groq import GroqAgent
from nagatoai_core.agent.anthropic import AnthropicAgent
from nagatoai_core.agent.message import Exchange, Message, Sender, ToolResult
from nagatoai_core.mission.mission import Mission, MissionStatus
from nagatoai_core.mission.task import Task, TaskStatus, TaskOutcome, TaskResult
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
    CRITIC_PROMPT,
)


def send_agent_request(
    agent: Agent, prompt: str, task: Task, temperature: float, max_tokens: int
) -> Exchange:
    """
    Sends a request to the agent to generate a response.
    :param agent: The agent to send the request to.
    :param prompt: The prompt to send to the agent.
    :param task: The task to reason about.
    :param temperature: The temperature of the agent.
    :param max_tokens: The maximum number of tokens to generate.
    :return: The exchange object containing the user message and the agent response.
    """

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        # TextColumn("[bold blue]{task.fields[title]}"),
        transient=True,
    ) as progress:
        progress.add_task(f"[cyan]Sending request to agent {agent.name}", total=None)
        return agent.chat(prompt, task, temperature, max_tokens)


def print_exchange(console: Console, agent: Agent, exchange: Exchange, color: str):
    """
    Prints the exchange between the user and the agent.
    :param agent: The agent involved in the exchange.
    :param exchange: The exchange between the user and the agent.
    :param color: The color of the panel.
    """
    console.print(
        Panel(
            exchange.user_msg.content,
            title=f"<{agent.model}> {agent.name} Prompt",
            title_align="left",
            border_style=color,
        )
    )

    if exchange.agent_response.tool_calls:
        for tool_result in exchange.agent_response.tool_results:
            console.print(
                Panel(
                    str(tool_result.result),
                    title=f"<{agent.model}> {agent.name} Tool Result",
                    title_align="left",
                    border_style=color,
                )
            )
    else:
        console.print(
            Panel(
                exchange.agent_response.content,
                title=f"<{agent.model}> {agent.name} Response",
                title_align="left",
                border_style=color,
            )
        )


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


# def process_tasks()


def main():
    """
    main represents the main entry point of the program.
    """
    load_dotenv()

    # For pretty console logs
    console = Console()

    openai_client = OpenAI(
        organization=os.getenv("OPENAI_ORG_ID"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_client = Anthropic(api_key=anthropic_api_key)

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # coordinator_agent: Agent = OpenAIAgent(
    #     openai_client,
    #     "gpt-4-turbo-preview",
    #     "Coordinator",
    #     COORDINATOR_SYSTEM_PROMPT,
    #     "Coordinator Agent",
    # )

    coordinator_agent: Agent = AnthropicAgent(
        anthropic_client,
        "claude-3-opus-20240229",
        "Coordinator",
        COORDINATOR_SYSTEM_PROMPT,
        "Coordinator Agent",
    )

    # researcher_agent: Agent = AnthropicAgent(
    #     anthropic_client,
    #     "claude-3-sonnet-20240229",
    #     "Researcher",
    #     RESEARCHER_SYSTEM_PROMPT,
    #     "Researcher Agent",
    # )

    # researcher_agent: Agent = GroqAgent(
    #     groq_client,
    #     "llama3-70b-8192",
    #     "Researcher",
    #     RESEARCHER_SYSTEM_PROMPT,
    #     "Researcher Agent",
    # )

    # researcher_agent: Agent = AnthropicAgent(
    #     anthropic_client,
    #     "claude-3-opus-20240229",
    #     "Researcher",
    #     RESEARCHER_SYSTEM_PROMPT,
    #     "Researcher Agent",
    # )

    researcher_agent: Agent = OpenAIAgent(
        openai_client,
        "gpt-4-turbo-2024-04-09",
        "Researcher",
        RESEARCHER_SYSTEM_PROMPT,
        "Researcher Agent",
    )

    # critic_agent: Agent = AnthropicAgent(
    #     anthropic_client,
    #     "claude-3-opus-20240229",
    #     "Critic",
    #     CRITIC_SYSTEM_PROMPT,
    #     "Critic Agent",
    # )

    # critic_agent: Agent = AnthropicAgent(
    #     anthropic_client,
    #     "claude-3-sonnet-20240229",
    #     "Critic",
    #     CRITIC_SYSTEM_PROMPT,
    #     "Critic Agent",
    # )

    # critic_agent: Agent = GroqAgent(
    #     groq_client,
    #     "llama3-70b-8192",
    #     "Critic",
    #     CRITIC_SYSTEM_PROMPT,
    #     "Critic Agent",
    # )

    critic_agent: Agent = AnthropicAgent(
        anthropic_client,
        "claude-3-haiku-20240307",
        "Critic",
        CRITIC_SYSTEM_PROMPT,
        "Critic Agent",
    )

    # critic_agent: Agent = OpenAIAgent(
    #     openai_client,
    #     "gpt-4-turbo-2024-04-09",
    #     "Critic",
    #     CRITIC_SYSTEM_PROMPT,
    #     "Critic Agent",
    # )

    tool_registry = ToolRegistry()
    tool_registry.register_tool(ReadwiseDocumentFinderTool)
    tool_registry.register_tool(ReadwiseBookHighlightsListerTool)
    tool_registry.register_tool(HumanConfirmInputTool)
    tool_registry.register_tool(HumanInputTool)
    tool_registry.register_tool(WebPageScraperTool)
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
        task_pass = False
        task_retry_count = 0
        task_result: Optional[TaskResult] = None

        while not task_pass:
            task.start_time = datetime.now(timezone.utc)
            exchanges = process_task(
                task,
                task_result,
                researcher_agent,
                tool_registry,
                OpenAIToolProvider,
                # AnthropicToolProvider,
                console,
            )
            task.end_time = datetime.now(timezone.utc)

            task_result_str = ""
            # Construct critic prompt input containing all answers from exchanges of the current task
            for i, exchange in enumerate(exchanges):
                if exchange.user_msg.tool_results:
                    print(f"*** [{i}] Exchange has tool results ***")
                    task_result_str += (
                        "\n" + str(exchange.user_msg.tool_results[-1].result) + "\n\n"
                    )
                    task_result_str += (
                        "\n\n---\n" + exchange.agent_response.content + "\n\n"
                    )
                else:
                    print(f"*** [{i}] Exchange has no tool calls ***")
                    soup = BeautifulSoup(exchange.agent_response.content, "html.parser")
                    result_tag = soup.find("result")
                    if result_tag:
                        result_value = result_tag.get_text(strip=True)
                        # Skip if result is empty
                        if result_value:
                            task_result_str += (
                                "\n" + result_tag.get_text(strip=True) + "\n\n"
                            )

            sleep(3)

            # Critic agent can now be invoked
            critic_prompt = CRITIC_PROMPT.format(
                goal=task.goal, description=task.description, result=task_result_str
            )
            critic_exchange = send_agent_request(
                critic_agent, critic_prompt, [], 0.6, 2000
            )
            print_exchange(console, critic_agent, critic_exchange, "red")

            critic_soup = BeautifulSoup(
                critic_exchange.agent_response.content, "html.parser"
            )
            outcome = critic_soup.find("outcome").get_text(strip=True)
            evaluation = critic_soup.find("evaluation").get_text(strip=True)

            outcome_enum = TaskOutcome.from_str(outcome)
            task_result = TaskResult(
                result=task_result_str, evaluation=evaluation, outcome=outcome_enum
            )

            task.update(task_result)
            sleep(3)

            if task.result.outcome == TaskOutcome.MEETS_REQUIREMENT:
                task_pass = True
                print(f"✅ Task {task} has passed")
            else:
                task_retry_count += 1
                if task_retry_count >= 3:
                    print(f"Task {task} has been retried more than 3 times. Exiting...")
                    break
                print(f"❌ Task {task} has failed. Retrying...")

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
