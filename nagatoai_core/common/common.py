# Standard Library
from typing import List, Optional

# Third Party
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn

# Nagato AI
# Company Libraries
from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.message import Exchange
from nagatoai_core.mission.task import Task
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider


def send_agent_request(
    agent: Agent,
    task: Optional[Task],
    prompt: str,
    tools: List[AbstractToolProvider],
    temperature: float,
    max_tokens: int,
) -> Exchange:
    """
    Sends a request to the agent to generate a response.
    :param agent: The agent to send the request to.
    :param prompt: The prompt to send to the agent.
    :param tools: The tools to provide to the agent.
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
        return agent.chat(task, prompt, tools, temperature, max_tokens)


def print_exchange(console: Console, agent: Agent, exchange: Exchange, color: str, task_id: str = ""):
    """
    Prints the exchange between the user and the agent.
    :param agent: The agent involved in the exchange.
    :param exchange: The exchange between the user and the agent.
    :param color: The color of the panel.
    """
    console.print(
        Panel(
            exchange.user_msg.content,
            title=f"<model={agent.model}, task-id={task_id}> {agent.name} Prompt ",
            title_align="left",
            border_style=color,
        )
    )

    if exchange.agent_response.tool_calls:
        for tool_result in exchange.agent_response.tool_results:
            console.print(
                Panel(
                    str(tool_result.result),
                    title=f"<model={agent.model}, task-id={task_id}> {agent.name} Tool Result",
                    title_align="left",
                    border_style=color,
                )
            )
    else:
        console.print(
            Panel(
                exchange.agent_response.content,
                title=f"<model={agent.model}, task-id={task_id}> {agent.name} Response",
                title_align="left",
                border_style=color,
            )
        )
