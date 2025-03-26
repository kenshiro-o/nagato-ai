# Standard Library
import importlib
import logging
import os
import re
from typing import Dict, Tuple, Type

# Third Party
from bs4 import BeautifulSoup
from pydantic import BaseModel

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.anthropic import AnthropicAgent
from nagatoai_core.agent.deepseek import DeepSeekAgent
from nagatoai_core.agent.factory import create_agent, get_agent_tool_provider, get_agent_type
from nagatoai_core.agent.google import GoogleAgent
from nagatoai_core.agent.groq import GroqAgent
from nagatoai_core.agent.openai import OpenAIAgent
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.agent_node import AgentNode
from nagatoai_core.graph.conditional_flow import ConditionalFlow
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.parallel_flow import ParallelFlow
from nagatoai_core.graph.plan.base_model_generator import BaseModelGenerator
from nagatoai_core.graph.plan.plan import Plan
from nagatoai_core.graph.sequential_flow import SequentialFlow
from nagatoai_core.graph.tool_node_with_params_conversion import ToolNodeWithParamsConversion
from nagatoai_core.graph.transformer_flow import TransformerFlow
from nagatoai_core.graph.types import ComparisonType, PredicateJoinType
from nagatoai_core.graph.unfold_flow import UnfoldFlow
from nagatoai_core.prompt.template.prompt_template import PromptTemplate
from nagatoai_core.tool.lib.audio.afplay import AfPlayTool
from nagatoai_core.tool.lib.audio.stt.assemblyai import AssemblyAITranscriptionTool
from nagatoai_core.tool.lib.audio.stt.groq_whisper import GroqWhisperTool
from nagatoai_core.tool.lib.audio.stt.openai_whisper import OpenAIWhisperTool
from nagatoai_core.tool.lib.audio.tts.eleven_labs import ElevenLabsTTSTool
from nagatoai_core.tool.lib.audio.tts.openai import OpenAITTSTool
from nagatoai_core.tool.lib.audio.video_to_mp3 import VideoToMP3Tool
from nagatoai_core.tool.lib.filesystem.file_checker import FileCheckerTool
from nagatoai_core.tool.lib.filesystem.text_file_reader import TextFileReaderTool
from nagatoai_core.tool.lib.filesystem.text_file_writer import TextFileWriterTool
from nagatoai_core.tool.lib.human.confirm import HumanConfirmInputTool
from nagatoai_core.tool.lib.human.input import HumanInputTool
from nagatoai_core.tool.lib.readwise.book_finder import ReadwiseDocumentFinderTool
from nagatoai_core.tool.lib.readwise.book_highlights_lister import ReadwiseBookHighlightsListerTool
from nagatoai_core.tool.lib.readwise.highlights_lister import ReadwiseHighightsListerTool
from nagatoai_core.tool.lib.time.time_now import TimeNowTool
from nagatoai_core.tool.lib.time.time_offset import TimeOffsetTool
from nagatoai_core.tool.lib.video.details_checker import VideoCheckerTool
from nagatoai_core.tool.lib.video.youtube.video_download import YouTubeVideoDownloadTool
from nagatoai_core.tool.lib.web.page_scraper import WebPageScraperTool
from nagatoai_core.tool.lib.web.serper_search import SerperSearchTool
from nagatoai_core.tool.registry import ToolRegistry

AVAILABLE_FLOW_TYPES = [
    "sequential_flow",
    "parallel_flow",
    "conditional_flow",
    "transformer_flow",
    "unfold_flow",
]

AVAILABLE_BASE_NODE_TYPES = ["agent_node", "tool_node_with_params_conversion"]

AVAILABLE_ALL_NODE_TYPES = AVAILABLE_BASE_NODE_TYPES + AVAILABLE_FLOW_TYPES


class XMLPlanParser:
    """
    Parser for XML plan files that define agents and their configurations.

    This class parses XML plan definitions and creates Agent objects based on
    the specifications in the XML.
    """

    def __init__(self):
        """Initialize the XML Plan Parser."""
        super().__init__()

    def parse(self, xml_str: str) -> Plan:
        """
        Parse an XML string into a Plan object.

        Args:
            xml_str: The XML string to parse

        Returns:
            A Plan object containing the parsed agents and output schemas

        Raises:
            ValueError: If the XML is not valid or required elements are missing
        """
        soup = BeautifulSoup(xml_str, "xml")

        # Parse agents
        agents_node = soup.find("agents")
        if agents_node is None:
            raise ValueError("No agents node found in the XML plan")
        agents = self.parse_agents(agents_node)

        # Parse output schemas
        output_schemas = {}
        output_schemas_node = soup.find("output_schemas")
        if output_schemas_node:
            output_schemas = self.parse_output_schemas(output_schemas_node)

        # Parse nodes
        nodes = {}
        nodes_node = soup.find("nodes")
        if nodes_node:
            nodes = self.parse_nodes(nodes_node, agents, output_schemas)

        graph_node = soup.find("graph")
        graph = self.parse_graph(graph_node, nodes)

        return Plan(agents=agents, output_schemas=output_schemas, graph=graph)

    def parse_agents(self, agents_node: BeautifulSoup) -> Dict[str, Agent]:
        """
        Parse the agents node and extract all agent definitions.

        Args:
            agents_node: The agents XML node

        Returns:
            Dictionary mapping agent names to Agent objects
        """
        agents = {}

        for agent_node in agents_node.find_all("agent"):
            name, agent = self.parse_agent(agent_node)
            agents[name] = agent

        return agents

    def parse_agent(self, agent_node: BeautifulSoup) -> Tuple[str, Agent]:
        """
        Parse an agent node and create an Agent instance.

        Args:
            agent_node: The agent XML node

        Returns:
            An Agent instance

        Raises:
            ValueError: If required attributes are missing
        """
        name = agent_node.get("name")
        if name is None:
            raise ValueError("Missing required attribute 'name' in agent node")

        model = self._get_node_text(agent_node, "model")
        role = self._get_node_text(agent_node, "role")
        role_description = self._get_node_text(agent_node, "role_description")
        nickname = self._get_node_text(agent_node, "nickname")

        api_key = self._get_api_key(model)

        return name, create_agent(
            api_key=api_key, model=model, role=role, role_description=role_description, nickname=nickname
        )

    def parse_output_schemas(self, output_schemas_node: BeautifulSoup) -> Dict[str, Type[BaseModel]]:
        """
        Parse the output_schemas node and convert JSON schemas to Pydantic models.

        Args:
            output_schemas_node: The output_schemas XML node

        Returns:
            Dictionary mapping schema names to Pydantic model classes

        Raises:
            ValueError: If a schema is invalid
        """
        schemas = {}

        for schema_node in output_schemas_node.find_all("output_schema"):
            name = schema_node.get("name")
            if name is None:
                raise ValueError("Missing required attribute 'name' in output_schema node")

            # Extract the JSON schema from the node text
            json_schema = schema_node.text.strip()
            if not json_schema:
                raise ValueError(f"Empty schema definition for '{name}'")

            try:
                # Use BaseModelGenerator to create a Pydantic model from the JSON schema
                model_class = BaseModelGenerator.generate(json_schema)
                schemas[name] = model_class
            except Exception as e:
                raise ValueError(f"Failed to parse schema '{name}': {str(e)}")

        return schemas

    def _get_node_text(self, parent_node: BeautifulSoup, node_name: str) -> str:
        """
        Get the text content of a child node.

        Args:
            parent_node: The parent XML element
            node_name: The name of the child node

        Returns:
            The text content of the child node

        Raises:
            ValueError: If the child node doesn't exist
        """
        node = parent_node.find(node_name)
        if node is None:
            raise ValueError(f"Missing required node: {node_name}")
        return node.text.strip() if node.text else ""

    def _get_api_key(self, model: str) -> str:
        """
        Get the appropriate API key based on the model name.

        Args:
            model: The model name

        Returns:
            The API key for the model

        Raises:
            ValueError: If the API key is not found
        """
        agent_type = get_agent_type(model).__name__

        if agent_type == "OpenAIAgent":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            return api_key

        if agent_type == "DeepSeekAgent":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable not set")
            return api_key

        if agent_type == "AnthropicAgent":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return api_key

        if agent_type == "GroqAgent":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            return api_key

        if agent_type == "GoogleAgent":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            return api_key

        raise ValueError(f"Unsupported model: {model}")

    def parse_nodes(
        self, nodes_node: BeautifulSoup, agents: Dict[str, Agent], output_schemas: Dict[str, Type[BaseModel]]
    ) -> Dict[str, AbstractNode]:
        """
        Parse the nodes section of an XML plan.

        Args:
            nodes_node: The XML nodes element
            agents: Dictionary of pre-parsed Agent instances
            output_schemas: Dictionary of output schema models

        Returns:
            Dictionary mapping node IDs to AbstractNode instances
        """
        # Add a whole bunch of tools to the tool registry
        # TODO - in the future we should have a separate helper method that lives elsewhere and is responsible for populating the tool registry
        # with all the tools that are available in the system
        tool_registry = ToolRegistry()
        tool_registry.register_tool(ReadwiseDocumentFinderTool)
        tool_registry.register_tool(ReadwiseHighightsListerTool)
        tool_registry.register_tool(ReadwiseBookHighlightsListerTool)
        tool_registry.register_tool(HumanConfirmInputTool)
        tool_registry.register_tool(HumanInputTool)
        tool_registry.register_tool(WebPageScraperTool)
        tool_registry.register_tool(SerperSearchTool)
        tool_registry.register_tool(FileCheckerTool)
        tool_registry.register_tool(TextFileReaderTool)
        tool_registry.register_tool(TextFileWriterTool)
        tool_registry.register_tool(TimeNowTool)
        tool_registry.register_tool(TimeOffsetTool)
        tool_registry.register_tool(VideoCheckerTool)
        tool_registry.register_tool(YouTubeVideoDownloadTool)
        tool_registry.register_tool(GroqWhisperTool)
        tool_registry.register_tool(OpenAITTSTool)
        tool_registry.register_tool(AssemblyAITranscriptionTool)
        tool_registry.register_tool(VideoToMP3Tool)
        tool_registry.register_tool(OpenAIWhisperTool)
        tool_registry.register_tool(ElevenLabsTTSTool)
        tool_registry.register_tool(AfPlayTool)
        nodes = {}

        for agent_node in nodes_node.find_all("agent_node"):
            node = self.parse_agent_node(agent_node, agents, output_schemas, tool_registry)
            nodes[node.id] = node

        # Parse tool_node_with_params_conversion nodes
        for tool_node in nodes_node.find_all("tool_node_with_params_conversion"):
            node = self.parse_tool_node_with_params_conversion(tool_node, agents, tool_registry)
            nodes[node.id] = node

        # Parse sequential_flow nodes
        for sequential_flow in nodes_node.find_all("sequential_flow"):
            node = self.parse_sequential_flow(sequential_flow, agents, output_schemas, tool_registry)
            nodes[node.id] = node

        # Parse parallel_flow nodes
        for parallel_flow in nodes_node.find_all("parallel_flow"):
            node = self.parse_parallel_flow(parallel_flow, agents, output_schemas, tool_registry)
            nodes[node.id] = node

        # Parse conditional_flow nodes
        for conditional_flow in nodes_node.find_all("conditional_flow"):
            node = self.parse_conditional_flow(conditional_flow, agents, output_schemas, tool_registry)
            nodes[node.id] = node

        # Parse transformer_flow nodes
        for transformer_flow in nodes_node.find_all("transformer_flow"):
            node = self.parse_transformer_flow(transformer_flow, agents, output_schemas, tool_registry)
            nodes[node.id] = node

        # Parse unfold_flow nodes
        for unfold_flow in nodes_node.find_all("unfold_flow"):
            node = self.parse_unfold_flow(unfold_flow, agents, output_schemas, tool_registry)
            nodes[node.id] = node

        return nodes

    def parse_agent_node(
        self,
        agent_node_bs: BeautifulSoup,
        agents: Dict[str, Agent],
        output_schemas: Dict[str, Type[BaseModel]],
        tool_registry: ToolRegistry,
    ) -> AgentNode:
        """
        Parse an agent_node XML element into an AgentNode instance.

        Args:
            agent_node_bs: The agent_node XML element
            agents: Dictionary of pre-parsed Agent instances
            output_schemas: Dictionary of output schema models
            tool_registry: ToolRegistry instance

        Returns:
            An AgentNode instance

        Raises:
            ValueError: If required attributes are missing or references invalid
        """
        # Extract required attributes
        node_id = agent_node_bs.get("id")
        if not node_id:
            raise ValueError("Missing required attribute 'id' in agent_node")

        node_name = agent_node_bs.get("name", node_id)

        # Get agent reference
        agent_elem = agent_node_bs.find("agent")
        if not agent_elem:
            raise ValueError(f"Missing agent reference in agent_node {node_id}")

        agent_name = agent_elem.get("name")
        if not agent_name or agent_name not in agents:
            raise ValueError(f"Invalid or missing agent name '{agent_name}' in agent_node {node_id}")

        agent = agents[agent_name]

        tool_provider_class = get_agent_tool_provider(agent)

        # Process prompt template if present
        prompt_template = None
        template_elem = agent_node_bs.find("prompt_template")
        if template_elem:
            template_text = self._get_node_text(template_elem, "template")
            # TODO: Parse data_light_schema if needed for more complex prompts
            prompt_template = PromptTemplate(template=template_text)

        # Process tools if present
        tools = None
        tools_elem = agent_node_bs.find("tools")
        if tools_elem:
            tools = []
            for tool_elem in tools_elem.find_all("tool"):
                tool_name = tool_elem.get("name")
                if not tool_name:
                    raise ValueError(f"Missing tool name in agent_node {node_id}")

                tool = tool_registry.get_tool(tool_name)
                logging.info(f"Tool found: tool_name={tool_name}, tool={tool}")
                logging.info(f"Tool provider class: {tool_provider_class}")

                tool_provider_instance = tool_provider_class(
                    tool=tool,
                    name=tool_name,
                    description=tool.description,
                    args_schema=tool.args_schema,
                )
                tools.append(tool_provider_instance)

        # Extract numerical parameters with defaults
        temperature = 0.7
        temp_elem = agent_node_bs.find("temperature")
        if temp_elem and temp_elem.text.strip():
            temperature = float(temp_elem.text.strip())

        max_tokens = 150
        tokens_elem = agent_node_bs.find("max_tokens")
        if tokens_elem and tokens_elem.text.strip():
            max_tokens = int(tokens_elem.text.strip())

        # Get output schema if specified
        output_schema = None
        schema_elem = agent_node_bs.find("output_schema")
        if schema_elem:
            schema_name = schema_elem.get("name")
            if schema_name and schema_name in output_schemas:
                output_schema = output_schemas[schema_name]
            else:
                raise ValueError(f"Invalid or missing output schema name '{schema_name}' in agent_node {node_id}")

        # Create and return the AgentNode
        return AgentNode(
            id=node_id,
            name=node_name,
            agent=agent,
            prompt_template=prompt_template,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            output_schema=output_schema,
        )

    def parse_tool_node_with_params_conversion(
        self,
        tool_node_bs: BeautifulSoup,
        agents: Dict[str, Agent],
        tool_registry: ToolRegistry,
    ) -> ToolNodeWithParamsConversion:
        """
        Parse a tool_node_with_params_conversion XML element into a ToolNodeWithParamsConversion instance.

        Args:
            tool_node_bs: The tool_node_with_params_conversion XML element
            agents: Dictionary of pre-parsed Agent instances
            tool_registry: ToolRegistry instance

        Returns:
            A ToolNodeWithParamsConversion instance

        Raises:
            ValueError: If required attributes are missing or references invalid
        """
        # Extract required attributes
        node_id = tool_node_bs.get("id")
        if not node_id:
            raise ValueError("Missing required attribute 'id' in tool_node_with_params_conversion")

        node_name = tool_node_bs.get("name", node_id)

        # Get agent reference
        agent_elem = tool_node_bs.find("agent")
        if not agent_elem:
            raise ValueError(f"Missing agent reference in tool_node_with_params_conversion {node_id}")

        agent_name = agent_elem.get("name")
        if not agent_name or agent_name not in agents:
            raise ValueError(
                f"Invalid or missing agent name '{agent_name}' in tool_node_with_params_conversion {node_id}"
            )

        agent = agents[agent_name]

        # Get tool reference
        tool_elem = tool_node_bs.find("tool")
        if not tool_elem:
            raise ValueError(f"Missing tool reference in tool_node_with_params_conversion {node_id}")

        tool_name = tool_elem.get("name")
        if not tool_name:
            raise ValueError(f"Missing tool name in tool_node_with_params_conversion {node_id}")

        tool = tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in tool registry")

        # Get optional parameters
        retries = 1
        retries_elem = tool_node_bs.find("retries")
        if retries_elem and retries_elem.text.strip():
            retries = int(retries_elem.text.strip())

        clear_memory = True
        clear_memory_elem = tool_node_bs.find("clear_memory_after_conversion")
        if clear_memory_elem and clear_memory_elem.text.strip().lower() == "false":
            clear_memory = False

        # Make sure to instantiate the tool
        tool_instance = tool()

        # Create tool provider
        tool_provider_class = get_agent_tool_provider(agent)
        tool_provider_instance = tool_provider_class(
            tool=tool_instance,
            name=tool_instance.name,
            description=tool_instance.description,
            args_schema=tool_instance.args_schema,
        )

        # Create and return the ToolNodeWithParamsConversion
        return ToolNodeWithParamsConversion(
            id=node_id,
            name=node_name,
            agent=agent,
            tool_provider=tool_provider_instance,
            retries=retries,
            clear_memory_after_conversion=clear_memory,
        )

    def parse_sequential_flow(
        self,
        sequential_flow_bs: BeautifulSoup,
        agents: Dict[str, Agent],
        output_schemas: Dict[str, Type[BaseModel]],
        tool_registry: ToolRegistry,
    ) -> SequentialFlow:
        """Parse a sequential flow node.

        Args:
            sequential_flow_bs: Tag containing sequential flow node.
            agents: Agents to use when parsing nodes.
            output_schemas: Output schemas to use when parsing nodes.
            tool_registry: Tool registry to use when parsing nodes.

        Returns:
            SequentialFlow node.
        """
        # Check required attributes
        if "id" not in sequential_flow_bs.attrs:
            raise ValueError("Missing required attribute 'id' in sequential_flow")

        flow_id = sequential_flow_bs["id"]
        flow_name = sequential_flow_bs.get("name", "")

        # Parse nodes
        nodes_bs = sequential_flow_bs.find("nodes")
        if not nodes_bs:
            raise ValueError(f"No nodes found in sequential_flow with id {flow_id}")

        parsed_nodes = self.parse_nodes(nodes_bs, agents, output_schemas)
        nodes_list = list(parsed_nodes.values())

        return SequentialFlow(id=flow_id, name=flow_name, nodes=nodes_list)

    def parse_parallel_flow(
        self,
        parallel_flow_xml: BeautifulSoup,
        agents: Dict[str, Agent],
        output_schemas: Dict[str, Type[BaseModel]],
        tool_registry: ToolRegistry,
    ) -> ParallelFlow:
        """
        Parse a parallel_flow XML element into a ParallelFlow instance.

        Args:
            parallel_flow_xml: The parallel_flow XML element
            agents: Dictionary of pre-parsed Agent instances
            output_schemas: Dictionary of output schema models
            tool_registry: ToolRegistry instance

        Returns:
            A ParallelFlow instance

        Raises:
            ValueError: If required attributes are missing or references invalid
        """
        # Extract required attributes
        flow_id = parallel_flow_xml.get("id")
        if not flow_id:
            raise ValueError("Missing required attribute 'id' in parallel_flow")

        node_name = parallel_flow_xml.get("name", flow_id)

        # Parse max_workers if specified
        max_workers = 4  # Default value
        max_workers_elem = parallel_flow_xml.find("max_workers")
        if max_workers_elem and max_workers_elem.text.strip():
            max_workers = int(max_workers_elem.text.strip())

        # Parse nested nodes
        nested_nodes = []
        nodes_elem = parallel_flow_xml.find("nodes")
        if nodes_elem:
            nested_nodes_dict = self.parse_nodes(nodes_elem, agents, output_schemas)
            # Convert the nodes dictionary to a list while preserving the order defined in XML
            for node_elem in nodes_elem.find_all(AVAILABLE_ALL_NODE_TYPES, recursive=False):
                node_id = node_elem.get("id")
                if node_id in nested_nodes_dict:
                    nested_nodes.append(nested_nodes_dict[node_id])

        # Create and return the ParallelFlow
        return ParallelFlow(id=flow_id, name=node_name, nodes=nested_nodes, max_workers=max_workers)

    def parse_conditional_flow(
        self,
        conditional_flow_bs: BeautifulSoup,
        agents: Dict[str, Agent],
        output_schemas: Dict[str, Type[BaseModel]],
        tool_registry: ToolRegistry,
    ) -> ConditionalFlow:
        """Parse a conditional flow node.

        Args:
            conditional_flow_bs: Tag containing conditional flow node.
            agents: Agents to use when parsing nodes.
            output_schemas: Output schemas to use when parsing nodes.
            tool_registry: Tool registry to use when parsing nodes.

        Returns:
            ConditionalFlow node.
        """
        # Extract required attributes
        flow_id = conditional_flow_bs.get("id")
        if not flow_id:
            raise ValueError("Missing required attribute 'id' in conditional_flow")

        flow_name = conditional_flow_bs.get("name", flow_id)

        # Extract configuration parameters
        broadcast_comparison = False
        broadcast_elem = conditional_flow_bs.find("broadcast_comparison")
        if broadcast_elem and broadcast_elem.text.strip().lower() == "true":
            broadcast_comparison = True

        input_index = 0
        index_elem = conditional_flow_bs.find("input_index")
        if index_elem and index_elem.text.strip():
            input_index = int(index_elem.text.strip())

        input_attribute = None
        attr_elem = conditional_flow_bs.find("input_attribute")
        if attr_elem and attr_elem.text.strip():
            input_attribute = attr_elem.text.strip()

        comparison_value = None
        value_elem = conditional_flow_bs.find("comparison_value")
        if value_elem and value_elem.text.strip():
            comparison_value = value_elem.text.strip()

        comparison_type = ComparisonType.EQUAL
        type_elem = conditional_flow_bs.find("comparison_type")
        if type_elem and type_elem.text.strip():
            comparison_type = ComparisonType[type_elem.text.strip()]

        # Handle custom comparison function
        custom_comparison_fn = None
        custom_fn_elem = conditional_flow_bs.find("custom_comparison_function")
        if custom_fn_elem:
            module_name = self._get_node_text(custom_fn_elem, "module")
            function_name = self._get_node_text(custom_fn_elem, "function")
            custom_comparison_fn = self._get_comparison_function(module_name, function_name)

        predicate_join_type = PredicateJoinType.AND
        join_elem = conditional_flow_bs.find("predicate_join_type")
        if join_elem and join_elem.text.strip():
            predicate_join_type = PredicateJoinType[join_elem.text.strip()]

        # Parse positive path
        positive_path = None
        positive_path_elem = conditional_flow_bs.find("positive_path", recursive=False)
        if positive_path_elem:
            # Check for each possible node type in positive_path
            child_nodes = positive_path_elem.find_all(AVAILABLE_ALL_NODE_TYPES, recursive=False)
            if child_nodes:
                # Parse the first child node
                node_elem = child_nodes[0]
                tag_name = node_elem.name

                if tag_name == "agent_node":
                    positive_path = self.parse_agent_node(node_elem, agents, output_schemas, tool_registry)
                elif tag_name == "tool_node_with_params_conversion":
                    positive_path = self.parse_tool_node_with_params_conversion(node_elem, agents, tool_registry)
                elif tag_name == "sequential_flow":
                    positive_path = self.parse_sequential_flow(node_elem, agents, output_schemas, tool_registry)
                elif tag_name == "parallel_flow":
                    positive_path = self.parse_parallel_flow(node_elem, agents, output_schemas, tool_registry)
                elif tag_name == "conditional_flow":
                    positive_path = self.parse_conditional_flow(node_elem, agents, output_schemas, tool_registry)
                elif tag_name == "unfold_flow":
                    positive_path = self.parse_unfold_flow(node_elem, agents, output_schemas, tool_registry)
                else:
                    raise ValueError(f"Invalid node type '{tag_name}' in positive_path of conditional_flow {flow_id}")

        # Parse negative path (similar to positive path)
        negative_path = None
        negative_path_elem = conditional_flow_bs.find("negative_path", recursive=False)
        if negative_path_elem:
            # Similar implementation as positive_path
            child_nodes = negative_path_elem.find_all(AVAILABLE_ALL_NODE_TYPES, recursive=False)
            if child_nodes:
                # Parse the first child node
                node_elem = child_nodes[0]
                tag_name = node_elem.name

                if tag_name == "agent_node":
                    negative_path = self.parse_agent_node(node_elem, agents, output_schemas, tool_registry)
                elif tag_name == "tool_node_with_params_conversion":
                    negative_path = self.parse_tool_node_with_params_conversion(node_elem, agents, tool_registry)
                elif tag_name == "sequential_flow":
                    negative_path = self.parse_sequential_flow(node_elem, agents, output_schemas, tool_registry)
                elif tag_name == "parallel_flow":
                    negative_path = self.parse_parallel_flow(node_elem, agents, output_schemas, tool_registry)
                elif tag_name == "conditional_flow":
                    negative_path = self.parse_conditional_flow(node_elem, agents, output_schemas, tool_registry)
                elif tag_name == "unfold_flow":
                    negative_path = self.parse_unfold_flow(node_elem, agents, output_schemas, tool_registry)
                else:
                    raise ValueError(f"Invalid node type '{tag_name}' in negative_path of conditional_flow {flow_id}")

        # Create and return the ConditionalFlow
        return ConditionalFlow(
            id=flow_id,
            name=flow_name,
            broadcast_comparison=broadcast_comparison,
            input_index=input_index,
            input_attribute=input_attribute,
            comparison_value=comparison_value,
            comparison_type=comparison_type,
            custom_comparison_fn=custom_comparison_fn,
            predicate_join_type=predicate_join_type,
            positive_path=positive_path,
            negative_path=negative_path,
        )

    def parse_transformer_flow(
        self,
        transformer_flow_bs: BeautifulSoup,
        agents: Dict[str, Agent],
        output_schemas: Dict[str, Type[BaseModel]],
        tool_registry: ToolRegistry,
    ) -> TransformerFlow:
        """Parse a transformer flow node.

        Args:
            transformer_flow_bs: Tag containing transformer flow node.
            agents: Agents to use when parsing nodes.
            output_schemas: Output schemas to use when parsing nodes.
            tool_registry: Tool registry to use when parsing nodes.

        Returns:
            TransformerFlow node.

        Raises:
            ValueError: If required attributes or elements are missing.
        """
        # Extract required attributes
        flow_id = transformer_flow_bs.get("id")
        if not flow_id:
            raise ValueError("Missing required attribute 'id' in transformer_flow")

        flow_name = transformer_flow_bs.get("name", flow_id)

        # Parse flow_param
        flow_param_elem = transformer_flow_bs.find("flow_param", recursive=False)
        if not flow_param_elem:
            raise ValueError("Missing required element 'flow_param' in transformer_flow")

        # The flow_param can contain any valid flow type
        flow_param = None
        for flow_type in AVAILABLE_FLOW_TYPES:
            flow_elem = flow_param_elem.find(flow_type, recursive=False)
            if flow_elem:
                if flow_type == "sequential_flow":
                    flow_param = self.parse_sequential_flow(flow_elem, agents, output_schemas, tool_registry)
                elif flow_type == "parallel_flow":
                    flow_param = self.parse_parallel_flow(flow_elem, agents, output_schemas, tool_registry)
                elif flow_type == "conditional_flow":
                    flow_param = self.parse_conditional_flow(flow_elem, agents, output_schemas, tool_registry)
                elif flow_type == "transformer_flow":
                    flow_param = self.parse_transformer_flow(flow_elem, agents, output_schemas, tool_registry)
                elif flow_type == "unfold_flow":
                    flow_param = self.parse_unfold_flow(flow_elem, agents, output_schemas, tool_registry)
                break

        if not flow_param:
            raise ValueError("No valid flow type found in flow_param")

        # Parse functor
        functor_elem = transformer_flow_bs.find("functor")
        if not functor_elem:
            raise ValueError("Missing required element 'functor' in transformer_flow")

        module_name = self._get_node_text(functor_elem, "module")
        function_name = self._get_node_text(functor_elem, "function")

        try:
            module = importlib.import_module(module_name)
            functor = getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import functor {function_name} from {module_name}: {str(e)}")

        # Create and return the TransformerFlow
        return TransformerFlow(
            id=flow_id,
            name=flow_name,
            flow_param=flow_param,
            functor=functor,
        )

    def parse_unfold_flow(
        self,
        unfold_flow_bs: BeautifulSoup,
        agents: Dict[str, Agent],
        output_schemas: Dict[str, Type[BaseModel]],
        tool_registry: ToolRegistry,
    ) -> UnfoldFlow:
        """Parse an unfold flow node.

        Args:
            unfold_flow_bs: Tag containing unfold flow node.
            agents: Agents to use when parsing nodes.
            output_schemas: Output schemas to use when parsing nodes.
            tool_registry: Tool registry to use when parsing nodes.

        Returns:
            UnfoldFlow node.

        Raises:
            ValueError: If required attributes are missing.
        """
        # Extract required attributes
        flow_id = unfold_flow_bs.get("id")
        if not flow_id:
            raise ValueError("Missing required attribute 'id' in unfold_flow")

        flow_name = unfold_flow_bs.get("name", flow_id)

        # Parse optional boolean parameters with defaults
        preserve_metadata = True
        preserve_metadata_elem = unfold_flow_bs.find("preserve_metadata")
        if preserve_metadata_elem and preserve_metadata_elem.text.strip().lower() == "false":
            preserve_metadata = False

        skip_empty_lists = False
        skip_empty_lists_elem = unfold_flow_bs.find("skip_empty_lists")
        if skip_empty_lists_elem and skip_empty_lists_elem.text.strip().lower() == "true":
            skip_empty_lists = True

        wrap_non_list_inputs = True
        wrap_non_list_inputs_elem = unfold_flow_bs.find("wrap_non_list_inputs")
        if wrap_non_list_inputs_elem and wrap_non_list_inputs_elem.text.strip().lower() == "false":
            wrap_non_list_inputs = False

        # Create and return the UnfoldFlow
        return UnfoldFlow(
            id=flow_id,
            name=flow_name,
            preserve_metadata=preserve_metadata,
            skip_empty_lists=skip_empty_lists,
            wrap_non_list_inputs=wrap_non_list_inputs,
        )

    def parse_graph(self, graph_node: BeautifulSoup, node_dict: Dict[str, AbstractNode]) -> Graph:
        """
        Parse a graph structure from XML and build a Graph object.

        Args:
            graph_node: The graph XML element
            node_dict: Dictionary mapping node IDs to AbstractNode instances

        Returns:
            A Graph instance with all edges defined in the XML

        Raises:
            ValueError: If a referenced node doesn't exist in node_dict
        """

        # Create a new Graph instance
        graph = Graph()

        # Find the graph element
        if not graph_node:
            raise ValueError("No graph element found in XML")

        # Find the edges element
        edges_elem = graph_node.find("edges")
        if not edges_elem:
            # No edges defined, but this might be valid (single node graph)
            return graph

        # Process each edge
        for edge_elem in edges_elem.find_all("edge"):
            from_id = edge_elem.get("from")
            to_id = edge_elem.get("to")

            if not from_id or not to_id:
                raise ValueError("Edge missing 'from' or 'to' attribute")

            # Verify nodes exist
            if from_id not in node_dict:
                raise ValueError(f"Source node '{from_id}' not found in node dictionary")

            if to_id not in node_dict:
                raise ValueError(f"Target node '{to_id}' not found in node dictionary")

            # Add the edge to the graph
            graph.add_edge(node_dict[from_id], node_dict[to_id])

        # Compile the graph to check for cycles and compute execution order
        graph.compile()

        return graph

    def _get_comparison_function(self, module_name, function_name):
        """
        Dynamically import a comparison function.

        Args:
            module_name: The full module path
            function_name: The function name in the module

        Returns:
            The imported callable function
        """
        try:
            module = importlib.import_module(module_name)
            return getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import comparison function {function_name} from {module_name}: {str(e)}")
