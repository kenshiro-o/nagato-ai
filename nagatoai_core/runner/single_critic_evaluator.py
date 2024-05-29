from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Type
from rich.console import Console
from bs4 import BeautifulSoup


from nagatoai_core.mission.task import Task, TaskResult, TaskOutcome
from nagatoai_core.agent.message import Exchange
from nagatoai_core.agent.agent import Agent
from nagatoai_core.runner.task_evaluator import TaskEvaluator
from nagatoai_core.prompt.templates import (
    CRITIC_PROMPT,
)
from nagatoai_core.common.common import print_exchange, send_agent_request

DEFAULT_CRITIC_TEMPERATURE = 0.6


class SingleCriticEvaluator(TaskEvaluator):
    critic_agent: Agent

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

    def evaluate(
        self, task: Task, agents: Dict[str, Agent], exchanges: List[Exchange]
    ) -> TaskResult:
        """
        Evaluates the task
        """
        console = Console()

        # TODO - Create a class that can flatten exchanges into a multi-line string
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
                else:
                    # Try to extract the task_result parent tag instead
                    task_result_tag = soup.find("task_result")
                    if task_result_tag:
                        task_result_str += (
                            "\n" + task_result_tag.get_text(strip=True) + "\n\n"
                        )
                    else:
                        # If we can't find anything in task_result then just use the full agent response
                        task_result_str += (
                            "\n" + exchange.agent_response.content + "\n\n"
                        )

        critic_prompt = CRITIC_PROMPT.format(
            goal=task.goal, description=task.description, result=task_result_str
        )
        critic_exchange = send_agent_request(
            self.critic_agent, critic_prompt, [], DEFAULT_CRITIC_TEMPERATURE, 2000
        )
        print_exchange(console, self.critic_agent, critic_exchange, "red")

        # TODO - create an ExchangeDecoder class that can decode an exchange and return Any type (e.g. a Dict or tuple)
        critic_soup = BeautifulSoup(
            critic_exchange.agent_response.content, "html.parser"
        )
        outcome = critic_soup.find("outcome").get_text(strip=True)
        evaluation = critic_soup.find("evaluation").get_text(strip=True)

        outcome_enum = TaskOutcome.from_str(outcome)
        task_result = TaskResult(
            result=task_result_str, evaluation=evaluation, outcome=outcome_enum
        )

        return task_result
