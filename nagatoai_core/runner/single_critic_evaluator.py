from typing import List, Dict
from rich.console import Console
from bs4 import BeautifulSoup

from langfuse import Langfuse

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
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data)

        self.langfuse = None if not self.tracing_enabled else Langfuse()

    def evaluate(
        self, task: Task, agents: Dict[str, Agent], exchanges: List[Exchange]
    ) -> TaskResult:
        """
        Evaluates the task
        """
        console = Console()
        lf_trace = None
        if self.tracing_enabled:
            lf_trace = self.langfuse.trace(
                name=f"SingleCriticEvaluator - Task {task.goal}",
                session_id=task.id,
            )

        # TODO - Create a class that can flatten exchanges into a multi-line string
        task_result_str = ""
        # Construct critic prompt input containing all answers from exchanges of the current task
        for _, exchange in enumerate(exchanges):
            if exchange.user_msg.tool_results:
                task_result_str += (
                    "\n" + str(exchange.user_msg.tool_results[-1].result) + "\n\n"
                )
                task_result_str += (
                    "\n\n---\n" + exchange.agent_response.content + "\n\n"
                )
            else:
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
            self.critic_agent, task, critic_prompt, [], DEFAULT_CRITIC_TEMPERATURE, 2000
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

        if lf_trace:
            lf_trace.generation(
                start_time=critic_exchange.user_msg.created_at,
                model=self.critic_agent.model,
                name=f"{self.critic_agent.name} - critic exchange",
                input=critic_exchange.chat_history,
                model_parameters={
                    "max_tokens": critic_exchange.token_stats_and_params.max_tokens,
                    "temperature": critic_exchange.token_stats_and_params.temperature,
                },
                metadata={
                    "task_id": task.id,
                    "task_goal": task.goal,
                    "task_description": task.description,
                    "task_outcome": str(outcome_enum),
                },
                output=critic_exchange.agent_response.content,
                end_time=critic_exchange.agent_response.created_at,
                usage={
                    "input": critic_exchange.token_stats_and_params.input_tokens_used,
                    "output": critic_exchange.token_stats_and_params.output_tokens_used,
                    "unit:": "TOKENS",
                },
            )

        return task_result
