# Standard Library
from abc import ABC, abstractmethod
from typing import Any, List

# Third Party
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

# Nagato AI
# Company Libraries
from nagatoai_core.chain.agent_param_converter import AgentParamConverter
from nagatoai_core.tool.provider.openai import OpenAIToolProvider


class Link(ABC, BaseModel):
    """
    Link represent a component of a chain
    """

    id: str = Field(
        None,
        description="The id of the link",
    )
    name: str = Field(
        None,
        description="The name of the link",
    )
    description: str = Field(
        None,
        description="The description of the link",
    )

    @abstractmethod
    def forward(self, input_data: Any) -> Any:
        """
        Forward the data through the link
        :param input_data: The input data to forward through the link
        :return: The output data obtained after processing the input data through the link
        """
        pass

    @abstractmethod
    def category(self) -> str:
        """
        Get the category of the link
        :return: The category of the link
        """
        pass


class Chain(BaseModel):
    """
    Chain represents a series of links that can be used to process data.
    """

    links: List[Link]
    agent_param_conv_link: AgentParamConverter = Field(
        None,
        description="The link that will convert the agent parameters to the tool parameters. Set to nil if you want to manually convert the agent parameters to the tool parameters.",
    )
    retries: int = Field(
        0,
        description="The number of retries to attempt if the a link fails",
    )

    def add_link(self, link: Link):
        """
        Add a link to the chain
        """
        self.links.append(link)

    def _run_link(self, link: Link, input_data: Any, console: Console, attempt_nb: int) -> Any:
        """
        Run a link with the given input data
        :param link: The link to run
        :param input_data: The input data to run the link with
        :param console: The console to print messages to
        :param attempt_nb: The number of the attempt
        :return: The output data obtained after processing the input data through the link
        """
        try:
            console.print(
                Panel(
                    f"Running link {link.name} with input data: {input_data}",
                    title=f"🔗 Chain runtime <Attempt {attempt_nb}> - pre-link execution 🔗",
                    title_align="left",
                    border_style="blue",
                )
            )

            data_converted = False
            if link.category() == "TOOL_LINK" and self.agent_param_conv_link is not None:
                # tool_link: ToolLink = link
                # TODO - Find a way to set the type to ToolLink without circular imports
                tool_instance = link.tool()

                provider = OpenAIToolProvider(
                    tool=tool_instance,
                    name=tool_instance.name,
                    description=tool_instance.description,
                    args_schema=tool_instance.args_schema,
                )
                schema = provider.schema()

                conv_data = {
                    "input_data": input_data,
                    "target_schema": schema,
                }
                console.print(
                    Panel(
                        f"Invoking tool param conversion agent link with conversion data: {conv_data}",
                        title=f"🛠️ Chain runtime <Attempt {attempt_nb}> - implicit tool param conversion agent 🛠️",
                        title_align="left",
                        border_style="orange_red1",
                    )
                )

                input_data = self.agent_param_conv_link.convert(conv_data)
                data_converted = True

            if data_converted:
                console.print(
                    Panel(
                        f"Running link {link.name} with converted input data: {input_data}",
                        title=f"🔗 Chain runtime <Attempt {attempt_nb}> - post data conversion 🔗",
                        title_align="left",
                        border_style="blue",
                    )
                )

            input_data = link.forward(input_data)

            console.print(
                Panel(
                    f"Finished running link {link.name} with output data: {input_data}",
                    title=f"🔗 Chain runtime <Attempt {attempt_nb}> - <{link.name}> - post link execution 🔗",
                    title_align="left",
                    border_style="green",
                )
            )

            return input_data
        except Exception as e:
            console.print(
                Panel(
                    f"An error occurred while running link {link.name}: {e}",
                    title=f"🚨 Chain runtime <Attempt {attempt_nb}> - error",
                    title_align="left",
                    border_style="red",
                )
            )
            raise e

    def run(self, input_data: Any) -> Any:
        """
        Runs the chain. Each link within the chain will be sequentially executed.
        :param input_data: The input data to start the chain with.
        :return: The output data obtained after processing the input data through the whole chain.
        """
        console = Console()

        tries = self.retries + 1
        for link in self.links:
            for i in range(tries):
                try:
                    input_data = self._run_link(link, input_data, console, i + 1)
                    break
                except Exception as e:
                    if i == tries - 1:
                        raise e

        return input_data
