from typing import Union, Optional, List, Dict, Type
from datetime import datetime, timezone

from rich.console import Console

from nagatoai_core.runner.task_runner import TaskRunner
from nagatoai_core.agent.message import Exchange, ToolResult, ToolRun
from nagatoai_core.mission.task import TaskResult, TaskOutcome
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider
from nagatoai_core.prompt.templates import (
    RESEARCHER_TASK_PROMPT_WITH_EXAMPLE,
    RESEARCHER_TASK_PROMPT_NO_EXAMPLE,
    RESEARCHER_TASK_PROMPT_WITH_PREVIOUS_UNSATISFACTORY_TASK_RESULT,
)
from nagatoai_core.common.common import send_agent_request, print_exchange

DEFAULT_AGENT_TEMPERATURE = 0.6


class SingleAgentTaskRunner(TaskRunner):
    """
    SingleAgentTaskRunner is used to execute a given task using a single agent
    """

    def __init__(self, **data):
        super().__init__(**data)

        # check if there is only one agent
        if len(self.agents) != 1:
            raise ValueError("SingleAgentTaskRunner requires only one agent")

    def _run(self) -> List[Exchange]:
        """
        The workhorse of the run method
        """
        console = Console()

        # Pick the only agent from the dictionary
        agent = list(self.agents.values())[0]
        tool_provider = list(self.agent_tool_providers.values())[0]

        tools = self.tool_registry.get_all_tools()
        available_tools: List[AbstractToolProvider] = []
        for tool in tools:
            t = tool()
            at = tool_provider(
                tool=t,
                name=t.name,
                description=t.description,
                args_schema=t.args_schema,
            )
            available_tools.append(at)

        exchanges: List[Exchange] = []

        # TODO In the future implement a PromptBuilder class that is responsible for generating prompts given:
        #  - the current task
        #  - the previous task
        #  - the current task exchange history
        task_prompt = ""
        if (
            self.current_task.result
            and self.current_task.result.outcome != TaskOutcome.MEETS_REQUIREMENT
        ):
            task_prompt = (
                RESEARCHER_TASK_PROMPT_WITH_PREVIOUS_UNSATISFACTORY_TASK_RESULT.format(
                    goal=self.current_task.goal,
                    outcome=self.current_task.result.outcome,
                    evaluation=self.current_task.result.evaluation,
                    description=self.current_task.description,
                )
            )
        else:
            if len(agent.history) > 0:
                task_prompt = RESEARCHER_TASK_PROMPT_NO_EXAMPLE.format(
                    goal=self.current_task.goal,
                    description=self.current_task.description,
                )
            else:
                task_prompt = RESEARCHER_TASK_PROMPT_WITH_EXAMPLE.format(
                    goal=self.current_task.goal,
                    description=self.current_task.description,
                )

        exchange = send_agent_request(
            agent, task_prompt, available_tools, DEFAULT_AGENT_TEMPERATURE, 2000
        )
        exchanges.append(exchange)

        print_exchange(console, agent, exchange, "blue")

        latest_exchange = exchange
        while True:
            if not latest_exchange.agent_response.tool_calls:
                # print("*** No tool calls to process for task ** breaking out of loop ***")
                break

            # TODO - Move this block into a separate function
            #  where we can tell the agent that the tool name or parameter is incorrect
            #  and get the agent to suggest the appropriate tool name and parameters
            for tool_call in latest_exchange.agent_response.tool_calls:
                # print(
                #     f"*** Agent to call tool: {tool_call.name} with parameters: {tool_call.parameters}"
                # )
                tool = self.tool_registry.get_tool(tool_call.name)
                tool_instance = tool()
                tool_params_schema = tool_instance.args_schema
                tool_params = tool_params_schema(**tool_call.parameters)

                tool_run = self.tool_cache.get_tool_run(
                    tool_call.name, tool_call.parameters
                )

                # Make sure that the tool run result is not an error or Exception
                if tool_run and not isinstance(tool_run.result.result, BaseException):
                    tool_output = tool_run.result.result
                    print(
                        f"**** Using tool run result from cache [tool-call={tool_run.call.name}, params={tool_run.call.parameters} ****"
                    )
                else:
                    tool_output = tool_instance._run(tool_params)
                    print(
                        f"**** Tool run result not in cache. Running tool... [tool-call={tool_call.name}, params={tool_call.parameters} ****"
                    )

                # print(f"*** Tool Output: {tool_output}")

                tool_result = ToolResult(
                    id=tool_call.id, name=tool_call.name, result=tool_output, error=None
                )

                tool_run = ToolRun(id=tool_call.id, call=tool_call, result=tool_result)

                tool_result_exchange = agent.send_tool_run_results(
                    [tool_result], DEFAULT_AGENT_TEMPERATURE, 2000
                )
                exchanges.append(tool_result_exchange)

                self.tool_cache.add_tool_run(tool_run)

                # print(
                #     f"*** Assistant from Tool Call reply: {tool_result_exchange.agent_response}"
                # )
                print_exchange(console, agent, tool_result_exchange, "orange_red1")

            latest_exchange = exchanges[-1]

        return exchanges

    def run(self) -> List[Exchange]:
        """
        Runs the task
        """
        task_pass = False
        task_retry_count = 0
        task_result: Optional[TaskResult] = None

        all_exchanges: List[Exchange] = []

        while not task_pass:
            self.current_task.start_time = datetime.now(timezone.utc)
            exchanges = self._run()
            all_exchanges.extend(exchanges)
            self.current_task.end_time = datetime.now(timezone.utc)

            task_result = self.task_evaluator.evaluate(
                self.current_task, self.agents, exchanges
            )
            self.current_task.update(task_result)

            if self.current_task.result.outcome == TaskOutcome.MEETS_REQUIREMENT:
                task_pass = True
                print(f"✅ Task {self.current_task} has passed")
            else:
                task_retry_count += 1
                if task_retry_count >= 3:
                    print(
                        f"Task {self.current_task} has been retried more than 3 times. Exiting..."
                    )
                    break
                print(f"❌ Task {self.current_task} has failed. Retrying...")

        return all_exchanges
