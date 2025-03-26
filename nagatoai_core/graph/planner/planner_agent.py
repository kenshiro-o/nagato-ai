# Standard Library
import logging
from typing import List, Optional

# Third Party
from pydantic import BaseModel, ConfigDict

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.graph.plan.plan import Plan
from nagatoai_core.graph.plan.validator import XMLPlanValidator
from nagatoai_core.graph.plan.xml_parser import XMLPlanParser
from nagatoai_core.graph.planner.objective import Objective
from nagatoai_core.mission.task import Task
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider

MODELS_AVAILABLE = [
    "o1",
    "o3-mini",
    "o3-mini-high",
    "claude-3-7-sonnet-20250219",
    "gpt-4o-mini",
    "gpt-4o",
    "gemini-2.0-flash",
]


class PlannerAgent(BaseModel):
    """
    An agent that generates plans from human prompts.

    The PlannerAgent is responsible for:
    1. Generating an objective based on human input
    2. Generating an XML plan based on the objective and human input
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Agent

    def generate_objective(
        self,
        task: Optional[Task],
        prompt: str,
        tools: List[AbstractToolProvider],
        temperature: float = 1,
        max_tokens: int = 1000,
    ) -> Objective:
        """
        Generate an objective based on the human prompt.

        Args:
            task: The current task (if any)
            prompt: The human prompt
            tools: Available tools
            temperature: The temperature for generation
            max_tokens: Maximum tokens for generation

        Returns:
            An objective statement as a string
        """
        objective_prompt = f"""
        Based on the following human request, generate a clear, concise objective statement.
        The objective should capture the core goal of what the human wants to achieve.

        Human request: {prompt}
        """

        exchange = self.agent.chat(
            task, objective_prompt, tools, temperature, max_tokens, target_output_schema=Objective
        )
        return exchange.agent_response.content

    def generate_plan_xml(
        self,
        prompt: str,
        objective: Objective,
        tools: List[AbstractToolProvider],
        temperature: float = 1,
        max_tokens: int = 6000,
    ) -> str:
        """
        Generate an XML plan based on the human prompt and objective.

        Args:
            task: The current task (if any)
            prompt: The human prompt
            objective: The generated objective
            tools: Available tools
            temperature: The temperature for generation
            max_tokens: Maximum tokens for generation

        Returns:
            A valid XML plan string
        """

        milestones = "\n".join([f"- {milestone}" for milestone in objective.milestones])

        models_available_xml_str = ""
        for model in MODELS_AVAILABLE:
            models_available_xml_str += f"<model>{model}</model>"

        models_available_xml_str = f"<models_available>{models_available_xml_str}</models_available>"

        # Read the schema plan.xsd file and insert it into the prompt
        xsd_content = ""
        with open("nagatoai_core/graph/plan/plan.xsd", "r") as file:
            xsd_content = file.read()

        tools_available_xml_str = ""
        for tool in tools:
            tools_available_xml_str += f"""
            <tool>
                <name>{tool.name}</name>
                <description>{tool.description}</description>
            </tool>
            """
        if tools_available_xml_str:
            tools_available_xml_str = f"<tools_available>{tools_available_xml_str}</tools_available>"

        plan_prompt = f"""
        Based on the following human request and objective, generate a detailed plan in XML format.

        <human_request>
        {prompt}
        </human_request>

        <objective>
        {objective.objective}
        </objective>

        <milestones>
        {milestones}
        </milestones>

        {models_available_xml_str}

        {tools_available_xml_str}

        Here is more information about how the building blocks for a plan work:
        - agent_node: Executes AI agents with configurable parameters
        - tool_node_with_params_conversion: Executes tools with parameter conversion capabilities
        - Flow Nodes are special nodes that manage subgraphs:
            - sequential_flow: Executes nodes in sequence
            - parallel_flow: Executes nodes in parallel
            - conditional_flow: Executes nodes based on conditions
            - transformer_flow: Transforms data using custom functions
            - unfold_flow: Expands lists into individual items

        Leverage the tools available if it makes sense to have nodes that use tools.
        The milestones represent potential steps needed to complete the objective. However, they may not be all needed.
        IMPORTANT: Only use the models available when specifying the model to use for an agent.
        IMPORTANT: For the node tool_node_with_params_conversion, make sure that the agent used is either gemini-2.0-flash, or claude-3-7-sonnet-20250219 .
        IMPORTANT: Inside the <graph> tag, make sure that we only have edges that connect top-level nodes to other top-level nodes (i.e. no edges fron sub-nodes contained within a sequential or parallel flow).
        The XML plan should follow this structure:
        ```xml
        {xsd_content}
        ```

        Ensure the plan is detailed and addresses all aspects of the objective.
        Make sure to add the namespace to the plan tag.
        """

        logging.debug(f"Plan prompt: {plan_prompt}")

        exchange = self.agent.chat(None, plan_prompt, tools, temperature, max_tokens)
        xml_plan = exchange.agent_response.content

        logging.debug(f"XML Plan generated from agent: {xml_plan}")

        # Extract the XML content if it's wrapped in code blocks
        if "```xml" in xml_plan and "```" in xml_plan:
            start_idx = xml_plan.find("```xml") + 6
            end_idx = xml_plan.rfind("```")
            xml_plan = xml_plan[start_idx:end_idx].strip()
        elif "```" in xml_plan:
            start_idx = xml_plan.find("```") + 3
            end_idx = xml_plan.rfind("```")
            xml_plan = xml_plan[start_idx:end_idx].strip()

        # Validate the XML
        xml_plan_validator = XMLPlanValidator()
        is_valid, errors = xml_plan_validator.validate_string(xml_plan)

        if not is_valid:
            # If invalid, attempt to fix common issues
            logging.warning(f"Invalid XML plan generated. Errors: {errors}")
            xml_plan = self._fix_xml_issues(xml_plan)

            # Validate again
            is_valid, errors = xml_plan_validator.validate_string(xml_plan)
            if not is_valid:
                logging.error(f"Failed to generate a valid XML plan. Errors: {errors}")
                raise ValueError(f"Failed to generate a valid XML plan: {errors}")

        return xml_plan

    def _fix_xml_issues(self, xml_plan: str) -> str:
        """
        Attempt to fix common XML issues.

        Args:
            xml_plan: The XML plan string with potential issues

        Returns:
            A potentially fixed XML plan string
        """
        # Ensure proper XML declaration
        if not xml_plan.startswith("<?xml"):
            xml_plan = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_plan

        # Ensure namespace is included
        if "<plan>" in xml_plan and "xmlns=" not in xml_plan:
            xml_plan = xml_plan.replace("<plan>", '<plan xmlns="http://nagatoai.com/schema/plan">')

        return xml_plan

    def parse_plan_to_object(self, xml_plan: str) -> Plan:
        """
        Parse an XML plan string into a Plan object.

        Args:
            xml_plan: The XML plan string

        Returns:
            A Plan object
        """
        xml_parser = XMLPlanParser()
        return xml_parser.parse(xml_plan)
