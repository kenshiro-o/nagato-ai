# Standard Library
import traceback
from collections.abc import Mapping
from typing import Any, Iterable, List

# Third Party
from pydantic import Field
from rich.console import Console
from rich.panel import Panel

# Nagato AI
# Company Libraries
from nagatoai_core.chain.agent_param_converter import AgentParamConverter
from nagatoai_core.chain.chain import Link
from nagatoai_core.tool.provider.openai import OpenAIToolProvider


class Ring(Link):
    """
    Ring represents a special link that iterates over the input data and returns the results in a similar iterable as the input data.
    You can think of a link as a mini chain
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

    # TODO Create a separate internal method for running a link -> this will allow us to do retries etc.

    def _run_link(self, link: Link, data: Any, console: Console, attempt_nb: int) -> Any:
        """
        Run a link and handle any exceptions that occur
        :param link: The link to run
        :param data: The data to run the link with
        :param console: The console to print the logs to
        :param attempt_nb: The number of the current attempt
        :return: The output of the link
        """
        try:
            # console.print(
            #     Panel(
            #         f"Invoking link {link.name} with input data: {data}",
            #         title="🪐 Ring runtime - pre-link execution 🪐",
            #         title_align="left",
            #         border_style="deep_sky_blue4",
            #     )
            # )

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
                    "input_data": data,
                    "target_schema": schema,
                }

                console.print(
                    Panel(
                        f"Invoking tool param conversion agent link with conversion data: {conv_data}",
                        title="🛠️ Ring runtime <Attempt {attempt_nb}> - tool param conversion agent 🛠️",
                        title_align="left",
                        border_style="purple",
                    )
                )

                data = self.agent_param_conv_link.convert(conv_data)

                console.print(
                    Panel(
                        f"Completed tool param conversion agent link with converted data: {data}",
                        title="🛠️ Ring runtime <Attempt {attempt_nb}> - tool param conversion agent 🛠️",
                        title_align="left",
                        border_style="medium_purple1",
                    )
                )

            data = link.forward(data)
            if link.category() == "AGENT_LINK":
                console.print(
                    Panel(
                        f"Finished running link {link.name} with output data: {data}",
                        title=f"🪐 Ring runtime <Attempt {attempt_nb}> - <{link.name}> - post link execution 🪐",
                        title_align="left",
                        border_style="dodger_blue2",
                    )
                )

            return data
        except Exception as le:
            print(traceback.format_exc())
            console.print(
                Panel(
                    f"Skipping item.... Error running ring links for input data {data}: {le}",
                    title=f"🛠️ Ring runtime <Attempt {attempt_nb}> - Error 🛠️",
                    title_align="left",
                    border_style="red",
                )
            )
            # TODO - instead of None should we set data to some error value so that we can identify it later?
            raise le

    def forward(self, input_data: Iterable) -> Any:
        """
        Forward the data through the ring
        :param input_data: The input data we are iterating over
        :return: The resulting data after iterating over the input data
        """
        is_dict = False
        output = []
        if isinstance(input_data, Mapping):
            output = {}
            is_dict = True

        console = Console()
        console.print(
            Panel(
                f"Processing ring {self.name} with input data: {input_data}",
                title="🪐 Ring runtime - pre-link iteration 🪐",
                title_align="left",
                border_style="purple",
            )
        )

        for data in input_data:

            initial_data = data
            tries = self.retries + 1

            for i in range(tries):
                try:
                    data = self._run_link(self.links[0], data, console, i + 1)
                    break
                except Exception as e:
                    if i == tries - 1:
                        raise e

            # Now add the result at the end of the link pass to the output
            if is_dict:
                # Use the key of the initial data as the key for the output
                output[initial_data[0]] = data
            else:
                output.append(data)

        console.print(
            Panel(
                f" Finished processing ring {self.name} with output data: {output}",
                title="🪐 Ring runtime - post-link execution 🪐",
                title_align="left",
                border_style="green",
            )
        )

        return output

    def category(self) -> str:
        """
        Get the category of the link
        :return: The category of the link
        """
        return "RING_LINK"
