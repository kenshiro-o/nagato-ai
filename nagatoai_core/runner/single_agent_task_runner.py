# Standard Library
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Type, Union

# Third Party
from langfuse import get_client, observe
from rich.console import Console

# Nagato AI
# Company Libraries
from nagatoai_core.agent.agent import Agent  # Import the Agent class
from nagatoai_core.agent.message import Exchange, ToolCall, ToolResult, ToolRun  # Added ToolCall here
from nagatoai_core.common.common import print_exchange, send_agent_request
from nagatoai_core.common.structured_logger import StructuredLogger
from nagatoai_core.mission.task import Task, TaskOutcome, TaskResult
from nagatoai_core.prompt.templates import (
    RESEARCHER_TASK_PROMPT_NO_EXAMPLE,
    RESEARCHER_TASK_PROMPT_WITH_EXAMPLE,
    RESEARCHER_TASK_PROMPT_WITH_PREVIOUS_UNSATISFACTORY_TASK_RESULT,
)
from nagatoai_core.runner.task_runner import TaskRunner
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider

DEFAULT_AGENT_TEMPERATURE = 0.6


def generate_tools_recommended_xml_str(task: Task) -> str:
    """
    Generate an XML string for the tools recommended for a task
    """
    tools_recommended_xml = ""
    for tool in task.tools_recommended:
        tools_recommended_xml += f"<tool><name>{tool}</name></tool>"

    return tools_recommended_xml


class SingleAgentTaskRunner(TaskRunner):
    """
    SingleAgentTaskRunner is used to execute a given task using a single agent
    """

    class Config:
        # arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data)

        # check if there is only one agent
        if len(self.agents) != 1:
            raise ValueError("SingleAgentTaskRunner requires only one agent")

        self.langfuse = None if not self.tracing_enabled else get_client()
        self.logger = StructuredLogger.get_logger({})

    # TODO - Refactor code so that this function is called whenever a tool call must be run
    def _run_tool_call(
        self,
        tool_call: ToolCall,
        agent: Agent,
        exchanges: List[Exchange],
        console: Console,
    ) -> Exchange:
        """
        Runs a tool call. Unused for now.
        """
        tool = self.tool_registry.get_tool(tool_call.name)
        tool_instance = tool()
        tool_params_schema = tool_instance.args_schema
        tool_params = tool_params_schema(**tool_call.parameters)

        tool_run = self.tool_cache.get_tool_run(tool_call.name, tool_call.parameters)

        # Make sure that the tool run result is not an error or Exception
        if tool_run and not isinstance(tool_run.result.result, BaseException):
            tool_output = tool_run.result.result
        else:
            tool_output = tool_instance._run(tool_params)

        tool_result = ToolResult(id=tool_call.id, name=tool_call.name, result=tool_output, error=None)

        tool_run = ToolRun(id=tool_call.id, call=tool_call, result=tool_result)

        tool_result_exchange = agent.send_tool_run_results([tool_result], DEFAULT_AGENT_TEMPERATURE, 2000)
        exchanges.append(tool_result_exchange)

        self.tool_cache.add_tool_run(tool_run)

        # print(
        #     f"*** Assistant from Tool Call reply: {tool_result_exchange.agent_response}"
        # )
        print_exchange(
            console,
            agent,
            tool_result_exchange,
            "orange_red1",
            task_id=self.current_task.id,
        )

        return tool_result_exchange

    @observe()
    def _run(self) -> List[Exchange]:
        """
        The workhorse of the run method
        """
        console = Console()

        if self.tracing_enabled and self.langfuse:
            # Update trace attributes for the current execution
            self.langfuse.update_current_trace(
                name=f"SingleAgentTaskRunner - Task {self.current_task.goal}",
                session_id=self.current_task.group_id,
                metadata={
                    "task_id": self.current_task.id,
                    "task_goal": self.current_task.goal,
                    "task_description": self.current_task.description,
                },
                input={
                    "goal": self.current_task.goal,
                    "description": self.current_task.description,
                },
            )

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

        t_recs = generate_tools_recommended_xml_str(self.current_task)

        # TODO In the future implement a PromptBuilder class that is responsible for generating prompts given:
        #  - the current task
        #  - the previous task
        #  - the current task exchange history
        task_prompt = ""
        if self.current_task.result and self.current_task.result.outcome != TaskOutcome.MEETS_REQUIREMENT:
            task_prompt = RESEARCHER_TASK_PROMPT_WITH_PREVIOUS_UNSATISFACTORY_TASK_RESULT.format(
                goal=self.current_task.goal,
                outcome=self.current_task.result.outcome,
                evaluation=self.current_task.result.evaluation,
                description=self.current_task.description,
                tools_recommended=t_recs,
            )
        else:
            if len(agent.history) > 0:
                task_prompt = RESEARCHER_TASK_PROMPT_NO_EXAMPLE.format(
                    goal=self.current_task.goal,
                    description=self.current_task.description,
                    tools_recommended=t_recs,
                )
            else:
                task_prompt = RESEARCHER_TASK_PROMPT_WITH_EXAMPLE.format(
                    goal=self.current_task.goal,
                    description=self.current_task.description,
                    tools_recommended=t_recs,
                )

        if self.tracing_enabled and self.langfuse:
            with self.langfuse.start_as_current_generation(
                name=f"{agent.name} - initial exchange",
                model=agent.model,
                model_parameters={
                    "max_tokens": 4096,
                    "temperature": DEFAULT_AGENT_TEMPERATURE,
                },
            ) as generation:
                exchange = send_agent_request(
                    agent,
                    self.current_task,
                    task_prompt,
                    available_tools,
                    DEFAULT_AGENT_TEMPERATURE,
                    4096,
                )
                exchanges.append(exchange)

                # Update generation with model call details
                generation.update(
                    input=exchange.chat_history,
                    output=exchange.agent_response.content,
                    metadata={
                        "model": agent.model,
                        "max_tokens": exchange.token_stats_and_params.max_tokens,
                        "temperature": exchange.token_stats_and_params.temperature,
                        "task_id": self.current_task.id,
                        "task_goal": self.current_task.goal,
                        "task_description": self.current_task.description,
                    },
                    usage={
                        "input": exchange.token_stats_and_params.input_tokens_used,
                        "output": exchange.token_stats_and_params.output_tokens_used,
                        "unit": "TOKENS",
                    },
                    start_time=exchange.user_msg.created_at,
                    end_time=exchange.agent_response.created_at,
                )
        else:
            exchange = send_agent_request(
                agent,
                self.current_task,
                task_prompt,
                available_tools,
                DEFAULT_AGENT_TEMPERATURE,
                4096,
            )
            exchanges.append(exchange)

        print_exchange(console, agent, exchange, "blue", task_id=self.current_task.id)

        latest_exchange = exchange
        while True:
            if not latest_exchange.agent_response.tool_calls:
                # print("*** No tool calls to process for task ** breaking out of loop ***")
                break

            # TODO - Move this block into the method _run_tool_call
            #  where we can tell the agent that the tool name or parameter is incorrect
            #  and get the agent to suggest the appropriate tool name and parameters

            tool_results: List[ToolResult] = []
            tool_runs: List[ToolRun] = []
            for tool_call in latest_exchange.agent_response.tool_calls:
                print(f"*** Agent to call tool: {tool_call.name} with parameters: {tool_call.parameters}")
                tool = self.tool_registry.get_tool(tool_call.name)
                tool_instance = tool()
                tool_params_schema = tool_instance.args_schema
                tool_params = tool_params_schema(**tool_call.parameters)

                tool_run = self.tool_cache.get_tool_run(tool_call.name, tool_call.parameters)

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

                print(f"*** Tool with name {tool_call.name} output: {tool_output}")

                tool_result = ToolResult(id=tool_call.id, name=tool_call.name, result=tool_output, error=None)
                tool_results.append(tool_result)

                tool_run = ToolRun(id=tool_call.id, call=tool_call, result=tool_result)
                tool_runs.append(tool_run)

                # Sleep in between tool calls
                time.sleep(10)

            print(f"*** Executed {len(tool_runs)} tool runs with results: {tool_results}")

            if self.tracing_enabled and self.langfuse:
                with self.langfuse.start_as_current_generation(
                    name=f"{agent.name} - tool run - {tool_call.name} (id={tool_call.id})",
                    model=agent.model,
                    model_parameters={
                        "max_tokens": 4096,
                        "temperature": DEFAULT_AGENT_TEMPERATURE,
                    },
                ) as generation:
                    tool_result_exchange = agent.send_tool_run_results(
                        self.current_task,
                        # [tool_result],
                        tool_results,
                        available_tools,
                        DEFAULT_AGENT_TEMPERATURE,
                        4096,
                    )
                    exchanges.append(tool_result_exchange)

                    # Update generation with tool run details
                    generation.update(
                        input=tool_result_exchange.chat_history,
                        output=tool_result_exchange.agent_response.content,
                        metadata={
                            "model": agent.model,
                            "max_tokens": tool_result_exchange.token_stats_and_params.max_tokens,
                            "temperature": tool_result_exchange.token_stats_and_params.temperature,
                            "task_id": self.current_task.id,
                            "task_goal": self.current_task.goal,
                            "task_description": self.current_task.description,
                        },
                        usage_details={
                            "input": tool_result_exchange.token_stats_and_params.input_tokens_used,
                            "output": tool_result_exchange.token_stats_and_params.output_tokens_used,
                            "unit": "TOKENS",
                        },
                        start_time=tool_result_exchange.user_msg.created_at,
                        end_time=tool_result_exchange.agent_response.created_at,
                    )
            else:
                tool_result_exchange = agent.send_tool_run_results(
                    self.current_task,
                    # [tool_result],
                    tool_results,
                    available_tools,
                    DEFAULT_AGENT_TEMPERATURE,
                    4096,
                )
                exchanges.append(tool_result_exchange)

            for tool_run in tool_runs:
                self.tool_cache.add_tool_run(tool_run)

            # print(
            #     f"*** Assistant from Tool Call reply: {tool_result_exchange.agent_response}"
            # )
            print_exchange(
                console,
                agent,
                tool_result_exchange,
                "orange_red1",
                task_id=self.current_task.id,
            )

            latest_exchange = exchanges[-1]

        # Update trace with final output
        if self.tracing_enabled and self.langfuse:
            self.langfuse.update_current_trace(
                output={
                    "exchanges_count": len(exchanges),
                    "final_response": latest_exchange.agent_response.content if exchanges else None,
                }
            )

        return exchanges

    @observe()
    def run(self) -> List[Exchange]:
        """
        Runs the task
        """
        if self.tracing_enabled and self.langfuse:
            # Update trace attributes for the overall task run
            self.langfuse.update_current_trace(
                name=f"SingleAgentTaskRunner.run - {self.current_task.goal}",
                session_id=self.current_task.group_id,
                metadata={
                    "task_id": self.current_task.id,
                    "task_goal": self.current_task.goal,
                    "task_description": self.current_task.description,
                    "tracing_enabled": self.tracing_enabled,
                },
                input={
                    "task": {
                        "goal": self.current_task.goal,
                        "description": self.current_task.description,
                        "tools_recommended": self.current_task.tools_recommended,
                    }
                },
            )

        task_pass = False
        task_retry_count = 0
        task_result: Optional[TaskResult] = None

        all_exchanges: List[Exchange] = []

        while not task_pass:
            self.current_task.start_time = datetime.now(timezone.utc)
            exchanges = self._run()
            all_exchanges.extend(exchanges)
            self.current_task.end_time = datetime.now(timezone.utc)

            task_result = self.task_evaluator.evaluate(self.current_task, self.agents, exchanges)
            self.current_task.update(task_result)

            if self.current_task.result.outcome == TaskOutcome.MEETS_REQUIREMENT:
                task_pass = True
                self.logger.info(
                    "✅Task completed successfully",
                    task_id=self.current_task.id,
                    task_goal=self.current_task.goal,
                    task_description=self.current_task.description,
                )
            else:
                task_retry_count += 1
                if task_retry_count >= 3:
                    self.logger.error(
                        "❌ Task has been retried more than 3 times. Exiting...",
                        task_id=self.current_task.id,
                        task_goal=self.current_task.goal,
                        task_description=self.current_task.description,
                    )
                    break
                self.logger.warning(
                    "⚠️ Task has failed. Retrying...",
                    task_id=self.current_task.id,
                    task_goal=self.current_task.goal,
                    task_description=self.current_task.description,
                )

        # Update trace with final outcome
        if self.tracing_enabled and self.langfuse:
            self.langfuse.update_current_trace(
                output={
                    "task_result": {
                        "outcome": str(task_result.outcome) if task_result else None,
                        "evaluation": task_result.evaluation if task_result else None,
                        "retry_count": task_retry_count,
                    },
                    "total_exchanges": len(all_exchanges),
                    "task_passed": task_pass,
                }
            )

        return all_exchanges
