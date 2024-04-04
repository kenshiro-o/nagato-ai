import re
from typing import List
from time import sleep

from openai import OpenAI
from anthropic import Anthropic

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn
from bs4 import BeautifulSoup


from nagato_ai.agent.agent import Agent
from nagato_ai.agent.openai import OpenAIAgent
from nagato_ai.agent.anthropic import AnthropicAgent
from nagato_ai.agent.message import Exchange
from nagato_ai.mission.mission import Mission
from nagato_ai.mission.task import Task, TaskOutcome, TaskResult
from nagato_ai.prompt.templates import (
    OBJECTIVE_PROMPT,
    COORDINATOR_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_TASK_PROMPT,
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
            title=f"{agent.name} Prompt",
            title_align="left",
            border_style=color,
        )
    )
    console.print(
        Panel(
            exchange.agent_response.content,
            title=f"{agent.name} Response",
            title_align="left",
            border_style=color,
        )
    )


def main():
    """
    main represents the main entry point of the program.
    """
    # For pretty console logs
    console = Console()

    openai_client = OpenAI(
        organization="<org_id>",
        api_key="<openai-api-key>",
    )

    anthropic_api_key = "<anthropic-api-key>"
    anthropic_client = Anthropic(api_key=anthropic_api_key)

    coordinator_agent: Agent = AnthropicAgent(
        anthropic_client,
        "claude-3-opus-20240229",
        "Coordinator",
        COORDINATOR_SYSTEM_PROMPT,
        "Coordinator Agent",
    )

    researcher_agent: Agent = AnthropicAgent(
        anthropic_client,
        "claude-3-sonnet-20240229",
        "Researcher",
        RESEARCHER_SYSTEM_PROMPT,
        "Researcher Agent",
    )

    # You can also use an OpenAI agent instead
    # researcher_agent: Agent = OpenAIAgent(
    #     openai_client,
    #     "gpt-4-turbo-preview",
    #     "Researcher",
    #     RESEARCHER_SYSTEM_PROMPT,
    #     "Researcher Agent",
    # )

    critic_agent: Agent = AnthropicAgent(
        anthropic_client,
        "claude-3-opus-20240229",
        "Critic",
        CRITIC_SYSTEM_PROMPT,
        "Critic Agent",
    )

    # You can also use an OpenAI agent instead
    # critic_agent: Agent = OpenAIAgent(
    #     openai_client,
    #     "gpt-4-turbo-preview",
    #     "Critic",
    #     CRITIC_SYSTEM_PROMPT,
    #     "Critic Agent",
    # )

    problem_statement = input("Please enter a problem statement for Nagato to solve:")

    problem_statement_prompt = OBJECTIVE_PROMPT.format(
        problem_statement=problem_statement
    )

    coordinator_exchange = send_agent_request(
        coordinator_agent, problem_statement_prompt, None, 0.6, 2000
    )

    soup = BeautifulSoup(coordinator_exchange.agent_response.content, "html.parser")
    objective = soup.find("objective").get_text(strip=True)
    tasks = soup.find_all("task")

    # Print the objective and tasks that the coordinator extracted from the problem statement
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
        task_prompt = RESEARCHER_TASK_PROMPT.format(
            goal=task.goal, description=task.description
        )
        exchange = send_agent_request(researcher_agent, task_prompt, task, 0.6, 2000)

        # Print the response from the agent that was assigned the current task
        print_exchange(console, researcher_agent, exchange, "blue")

        soup = BeautifulSoup(exchange.agent_response.content, "html.parser")
        goal = soup.find("goal").get_text(strip=True)
        result = soup.find("result").get_text(strip=True)
        sleep(3)

        critic_prompt = CRITIC_PROMPT.format(
            goal=goal, description=task.description, result=result
        )
        critic_exchange = send_agent_request(
            critic_agent, critic_prompt, task, 0.6, 2000
        )

        # Print the evaluation of the task from the critic agent
        print_exchange(console, critic_agent, critic_exchange, "red")

        critic_soup = BeautifulSoup(
            critic_exchange.agent_response.content, "html.parser"
        )
        outcome = critic_soup.find("outcome").get_text(strip=True)
        evaluation = critic_soup.find("evaluation").get_text(strip=True)

        outcome_enum = TaskOutcome.from_str(outcome)
        task_result = TaskResult(
            result=result, evaluation=evaluation, outcome=outcome_enum
        )

        task.update(task_result)
        sleep(3)

    # Save to file
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

    # Sanitize/escape the objective string so that we can turn it into a file name
    objective_file_name = re.sub(r"[^a-zA-Z0-9]+", "_", objective) + ".md"

    # Write the agent's response in a markdown file
    with open(f"./outputs/{objective_file_name}", "w") as f:
        f.write(markdown_output_str)


if __name__ == "__main__":
    main()
