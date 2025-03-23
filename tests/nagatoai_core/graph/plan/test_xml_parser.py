"""Unit tests for XMLPlanParser class."""

# Standard Library
import logging
import os
from unittest.mock import Mock, patch

# Third Party
import pytest
from bs4 import BeautifulSoup
from pydantic import BaseModel

# Nagato AI
from nagatoai_core.agent.agent import Agent
from nagatoai_core.agent.openai import OpenAIAgent
from nagatoai_core.graph.agent_node import AgentNode
from nagatoai_core.graph.conditional_flow import ConditionalFlow
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.parallel_flow import ParallelFlow
from nagatoai_core.graph.plan.plan import Plan
from nagatoai_core.graph.plan.xml_parser import XMLPlanParser
from nagatoai_core.graph.sequential_flow import SequentialFlow
from nagatoai_core.graph.tool_node_with_params_conversion import ToolNodeWithParamsConversion
from nagatoai_core.graph.types import ComparisonType, PredicateJoinType
from nagatoai_core.tool.lib.human.input import HumanInputTool
from nagatoai_core.tool.lib.readwise.book_finder import ReadwiseDocumentFinderTool
from nagatoai_core.tool.provider.openai import OpenAIToolProvider


@pytest.fixture
def parser():
    """Return an instance of XMLPlanParser for testing."""
    return XMLPlanParser()


@pytest.fixture
def agent_xml():
    """Return a BeautifulSoup object representing a single agent XML node."""
    xml = """
    <agent name="agent1">
        <model>gpt-4o</model>
        <role>Main Agent</role>
        <role_description>You are a useful assistant that can help with tasks and questions.</role_description>
        <nickname>my_agent</nickname>
    </agent>
    """
    return BeautifulSoup(xml, "xml").find("agent")


@pytest.fixture
def agents_xml():
    """Return a BeautifulSoup object representing multiple agents XML node."""
    xml = """
    <agents>
        <agent name="agent1">
            <model>gpt-4o</model>
            <role>Main Agent</role>
            <role_description>You are a useful assistant that can help with tasks and questions.</role_description>
            <nickname>agent1</nickname>
        </agent>
        <agent name="agent2">
            <model>claude-3-opus-20240229</model>
            <role>Secondary Agent</role>
            <role_description>You are an assistant that specializes in creative tasks.</role_description>
            <nickname>agent2</nickname>
        </agent>
    </agents>
    """
    return BeautifulSoup(xml, "xml").find("agents")


@pytest.fixture
def mock_agent():
    """Return a mock Agent instance."""
    mock = Mock(spec=Agent)
    return mock


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_type")
@patch("nagatoai_core.graph.plan.xml_parser.create_agent")
@patch("nagatoai_core.graph.plan.xml_parser.os.getenv")
def test_parse_agent(mock_getenv, mock_create_agent, mock_get_agent_type, parser: XMLPlanParser, agent_xml, mock_agent):
    """Test parsing a single agent XML node."""

    mock_get_agent_type.return_value = OpenAIAgent
    mock_getenv.return_value = "fake-api-key"
    mock_create_agent.return_value = mock_agent

    # Call the method
    name, agent = parser.parse_agent(agent_xml)

    # Verify results
    assert name == "agent1"
    assert agent == mock_agent
    mock_get_agent_type.assert_called_once_with("gpt-4o")
    mock_getenv.assert_called_once_with("OPENAI_API_KEY")
    mock_create_agent.assert_called_once_with(
        api_key="fake-api-key",
        model="gpt-4o",
        role="Main Agent",
        role_description="You are a useful assistant that can help with tasks and questions.",
        nickname="my_agent",
    )


@patch("nagatoai_core.graph.plan.xml_parser.XMLPlanParser.parse_agent")
def test_parse_agents(mock_parse_agent, parser: XMLPlanParser, agents_xml, mock_agent):
    """Test parsing multiple agent XML nodes."""
    # Set up mocks
    mock_agent1 = Mock(spec=Agent)
    mock_agent1.name = "my_agent1"

    mock_agent2 = Mock(spec=Agent)
    mock_agent2.name = "my_agent2"

    mock_parse_agent.side_effect = [("agent1", mock_agent1), ("agent2", mock_agent2)]

    # Call the method
    agents = parser.parse_agents(agents_xml)

    # Verify results
    assert len(agents) == 2
    assert agents["agent1"] == mock_agent1
    assert agents["agent2"] == mock_agent2
    assert mock_parse_agent.call_count == 2


def test_parse_agent_missing_name_attribute(parser: XMLPlanParser):
    """Test parsing agent node with missing name attribute."""
    xml = """
    <agent>
        <model>gpt-4o</model>
        <role>Main Agent</role>
        <role_description>Description</role_description>
        <nickname>agent1</nickname>
    </agent>
    """
    agent_node = BeautifulSoup(xml, "xml").find("agent")

    with pytest.raises(ValueError, match="Missing required attribute 'name' in agent node"):
        name, agent = parser.parse_agent(agent_node)


def test_get_node_text_missing_node(parser: XMLPlanParser, agent_xml: BeautifulSoup):
    """Test getting text from a missing node."""
    with pytest.raises(ValueError, match="Missing required node: non_existent_node"):
        parser._get_node_text(agent_xml, "non_existent_node")


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_type")
@patch("nagatoai_core.graph.plan.xml_parser.os.getenv")
def test_get_api_key_missing_env_var(mock_getenv, mock_get_agent_type, parser):
    """Test getting API key when environment variable is not set."""

    mock_get_agent_type.return_value = OpenAIAgent
    mock_getenv.return_value = None

    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
        parser._get_api_key("gpt-4o")


def test_parse_output_schemas():
    """Test parsing of output schemas from XML."""
    xml_str = """
    <plan>
        <agents>
            <agent name="test_agent">
                <model>gpt-4o</model>
                <role>assistant</role>
                <role_description>A helpful assistant</role_description>
                <nickname>Test</nickname>
            </agent>
        </agents>
        <output_schemas>
            <output_schema name="Person">
                {
                  "type": "object",
                  "properties": {
                    "name": {
                      "type": "string",
                      "description": "The person's name"
                    },
                    "age": {
                      "type": "integer",
                      "minimum": 0,
                      "description": "The person's age"
                    },
                    "email": {
                      "type": "string",
                      "format": "email",
                      "description": "The person's email address"
                    }
                  },
                  "required": ["name", "age"]
                }
            </output_schema>
            <output_schema name="Highlights">
                {
                  "type": "object",
                  "properties": {
                    "highlights": {
                      "type": "array",
                      "description": "A list of highlights.",
                      "items": {
                        "type": "object",
                        "properties": {
                          "start_offset_seconds": {
                            "type": "number",
                            "description": "The start offset of the highlight in the transcript in seconds."
                          },
                          "end_offset_seconds": {
                            "type": "number",
                            "description": "The end offset of the highlight in the transcript in seconds."
                          },
                          "transcript_text": {
                            "type": "string",
                            "description": "The transcript text of the highlight."
                          },
                          "reason": {
                            "type": "string",
                            "description": "The reason for selecting the highlight."
                          }
                        },
                        "required": ["start_offset_seconds", "end_offset_seconds", "transcript_text", "reason"]
                      }
                    }
                  },
                  "required": ["highlights"]
                }
            </output_schema>
        </output_schemas>

        <graph>
        </graph>
    </plan>
    """

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        parser = XMLPlanParser()
        plan = parser.parse(xml_str)

        # Verify the output_schemas were parsed
        assert "Person" in plan.output_schemas
        assert "Highlights" in plan.output_schemas

        # Check Person model
        Person = plan.output_schemas["Person"]
        assert issubclass(Person, BaseModel)

        # Create a valid Person
        person = Person(name="John Doe", age=30, email="john@example.com")
        assert person.name == "John Doe"
        assert person.age == 30
        assert person.email == "john@example.com"

        # Check validation works
        with pytest.raises(ValueError):
            Person(name="John Doe", age=-1)  # Invalid age

        # Check Highlights model
        Highlights = plan.output_schemas["Highlights"]
        assert issubclass(Highlights, BaseModel)

        # Create valid Highlights
        highlights = Highlights(
            highlights=[
                {
                    "start_offset_seconds": 10.5,
                    "end_offset_seconds": 20.75,
                    "transcript_text": "This is an important part.",
                    "reason": "Key information",
                }
            ]
        )

        # Verify nested structure
        assert len(highlights.highlights) == 1
        assert highlights.highlights[0].start_offset_seconds == 10.5
        assert highlights.highlights[0].transcript_text == "This is an important part."

        # Test validation of nested structure
        with pytest.raises(ValueError):
            Highlights(
                highlights=[
                    {
                        "start_offset_seconds": 10.5,
                        # Missing end_offset_seconds
                        "transcript_text": "This is an important part.",
                        "reason": "Key information",
                    }
                ]
            )


@pytest.fixture
def agent_node_xml():
    """Return XML for a basic agent node with minimal configuration."""
    xml = """
    <agent_node id="node1" name="Test Node">
        <agent name="agent1" />
        <temperature>0.5</temperature>
        <max_tokens>200</max_tokens>
    </agent_node>
    """
    return BeautifulSoup(xml, "xml").find("agent_node")


@pytest.fixture
def agent_node_with_tools_xml():
    """Return XML for an agent node with tools."""
    xml = """
    <agent_node id="node1" name="Test Node">
        <agent name="agent1" />
        <tools>
            <tool name="ReadwiseDocumentFinderTool" />
            <tool name="HumanInputTool" />
        </tools>
    </agent_node>
    """
    return BeautifulSoup(xml, "xml").find("agent_node")


@pytest.fixture
def agent_node_with_schema_xml():
    """Return XML for an agent node with output schema."""
    xml = """
    <agent_node id="node1" name="Test Node">
        <agent name="agent1" />
        <output_schema name="Person" />
    </agent_node>
    """
    return BeautifulSoup(xml, "xml").find("agent_node")


@pytest.fixture
def agent_node_invalid_tool_xml():
    """Return XML for an agent node with an invalid tool."""
    xml = """
    <agent_node id="node1" name="Test Node">
        <agent name="agent1" />
        <tools>
            <tool name="NonExistentTool" />
        </tools>
    </agent_node>
    """
    return BeautifulSoup(xml, "xml").find("agent_node")


@pytest.fixture
def agent_node_invalid_schema_xml():
    """Return XML for an agent node with an invalid schema."""
    xml = """
    <agent_node id="node1" name="Test Node">
        <agent name="agent1" />
        <output_schema name="NonExistentSchema" />
    </agent_node>
    """
    return BeautifulSoup(xml, "xml").find("agent_node")


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_agent_node_valid(mock_get_agent_tool_provider, parser, agent_node_xml, mock_agent):
    """Test parsing a valid agent node with existing agent."""
    # Setup
    agents = {"agent1": mock_agent}
    output_schemas = {}
    tool_registry = Mock()

    mock_tool_provider = Mock()
    mock_get_agent_tool_provider.return_value = mock_tool_provider

    # Execute
    agent_node = parser.parse_agent_node(agent_node_xml, agents, output_schemas, tool_registry)

    # Verify
    assert agent_node.id == "node1"
    assert agent_node.name == "Test Node"
    assert agent_node.agent == mock_agent
    assert agent_node.temperature == 0.5
    assert agent_node.max_tokens == 200
    assert agent_node.tools is None
    assert agent_node.output_schema is None


def test_parse_agent_node_invalid_agent(parser, agent_node_xml):
    """Test parsing an agent node with non-existent agent."""
    agents = {"different_agent": Mock()}
    output_schemas = {}
    tool_registry = Mock()

    with pytest.raises(ValueError, match="Invalid or missing agent name 'agent1'"):
        parser.parse_agent_node(agent_node_xml, agents, output_schemas, tool_registry)


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_agent_node_with_tools(mock_get_agent_tool_provider, parser, agent_node_with_tools_xml, mock_agent):
    """Test parsing an agent node with tools."""
    # Setup
    agents = {"agent1": mock_agent}
    output_schemas = {}

    # Create actual tool instances
    human_input_tool = HumanInputTool()
    readwise_tool = ReadwiseDocumentFinderTool()

    # Create a mock registry that returns actual tool instances
    tool_registry = Mock()

    def get_tool_side_effect(name):
        if name == "ReadwiseDocumentFinderTool":
            return readwise_tool
        elif name == "HumanInputTool":
            return human_input_tool
        else:
            raise ValueError(f"Unknown tool: {name}")

    tool_registry.get_tool.side_effect = get_tool_side_effect

    # Use the real OpenAIToolProvider
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Execute
    agent_node = parser.parse_agent_node(agent_node_with_tools_xml, agents, output_schemas, tool_registry)

    # Verify
    assert agent_node.id == "node1"
    assert agent_node.tools is not None
    assert len(agent_node.tools) == 2
    assert tool_registry.get_tool.call_count == 2

    # Verify the tools are of the expected type
    assert isinstance(agent_node.tools[0], OpenAIToolProvider)
    assert isinstance(agent_node.tools[1], OpenAIToolProvider)

    # Verify the tool instances were used correctly
    assert agent_node.tools[0].name == "ReadwiseDocumentFinderTool" or agent_node.tools[0].name == "HumanInputTool"
    assert agent_node.tools[1].name == "ReadwiseDocumentFinderTool" or agent_node.tools[1].name == "HumanInputTool"


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_agent_node_without_schema(mock_get_agent_tool_provider, parser, agent_node_xml, mock_agent):
    """Test parsing an agent node without output schema."""
    agents = {"agent1": mock_agent}
    output_schemas = {"Person": type("Person", (BaseModel,), {})}
    tool_registry = Mock()

    # Mock the tool provider class
    # Nagato AI

    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Execute
    agent_node = parser.parse_agent_node(agent_node_xml, agents, output_schemas, tool_registry)

    # Verify
    assert agent_node.output_schema is None


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_agent_node_with_invalid_tool(
    mock_get_agent_tool_provider, parser, agent_node_invalid_tool_xml, mock_agent
):
    """Test parsing an agent node with non-existent tool."""
    agents = {"agent1": mock_agent}
    output_schemas = {}

    tool_registry = Mock()
    tool_registry.get_tool.side_effect = ValueError("Tool NonExistentTool not found")

    with pytest.raises(ValueError, match="Tool NonExistentTool not found"):
        parser.parse_agent_node(agent_node_invalid_tool_xml, agents, output_schemas, tool_registry)


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_agent_node_with_schema(mock_get_agent_tool_provider, parser, agent_node_with_schema_xml, mock_agent):
    """Test parsing an agent node with existing output schema."""
    # Setup
    agents = {"agent1": mock_agent}

    # Mock schema class
    mock_schema = type("Person", (BaseModel,), {})
    output_schemas = {"Person": mock_schema}

    tool_registry = Mock()
    mock_get_agent_tool_provider.return_value = Mock()

    # Execute
    agent_node = parser.parse_agent_node(agent_node_with_schema_xml, agents, output_schemas, tool_registry)

    # Verify
    assert agent_node.id == "node1"
    assert agent_node.output_schema == mock_schema


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_agent_node_without_schema(mock_get_agent_tool_provider, parser, agent_node_xml, mock_agent):
    """Test parsing an agent node without output schema."""
    agents = {"agent1": mock_agent}
    output_schemas = None
    tool_registry = Mock()

    # Use the real OpenAIToolProvider
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    agent_node = parser.parse_agent_node(agent_node_xml, agents, output_schemas, tool_registry)

    assert agent_node.output_schema is None


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_agent_node_with_invalid_schema(
    mock_get_agent_tool_provider, parser, agent_node_invalid_schema_xml, mock_agent
):
    """Test parsing an agent node with non-existent output schema."""
    # Setup
    agents = {"agent1": mock_agent}
    output_schemas = {"Person": type("Person", (BaseModel,), {})}
    tool_registry = Mock()

    # Use the real OpenAIToolProvider
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Execute and verify that it raises the expected error
    with pytest.raises(ValueError, match="Invalid or missing output schema name 'NonExistentSchema'"):
        parser.parse_agent_node(agent_node_invalid_schema_xml, agents, output_schemas, tool_registry)


@pytest.fixture
def graph_xml():
    """Return a simple XML string for a graph with two connected nodes."""
    return """
    <graph>
      <edges>
        <edge from="node1" to="node2" />
        <edge from="node2" to="node3" />
      </edges>
    </graph>
    """


@pytest.fixture
def invalid_graph_xml():
    """Return an XML string for a graph with an invalid node reference."""
    return """
    <graph>
      <edges>
        <edge from="node1" to="invalid_node" />
      </edges>
    </graph>
    """


@pytest.fixture
def empty_graph_xml():
    """Return an XML string for a graph with no edges."""
    return """
    <graph>
      <edges>
      </edges>
    </graph>
    """


def test_parse_graph(parser, graph_xml):
    """Test parsing a graph structure from XML."""
    # Create mock nodes
    node1 = Mock()
    node1.id = "node1"

    node2 = Mock()
    node2.id = "node2"

    node3 = Mock()
    node3.id = "node3"

    # Create node dictionary
    node_dict = {"node1": node1, "node2": node2, "node3": node3}

    bs = BeautifulSoup(graph_xml, "xml")
    graph_node = bs.find("graph")

    # Parse the graph
    graph = parser.parse_graph(graph_node, node_dict)

    # Verify the graph structure
    assert isinstance(graph, Graph)
    assert len(graph.nodes_set) == 3
    assert "node1" in graph.adjacency_list
    assert "node2" in graph.adjacency_list
    assert "node2" in graph.adjacency_list["node1"]
    assert "node3" in graph.adjacency_list["node2"]
    assert graph.compiled is True


def test_parse_graph_invalid_node(parser, invalid_graph_xml):
    """Test parsing a graph with an invalid node reference."""
    # Create mock node
    node1 = Mock()
    node1.id = "node1"

    # Create node dictionary with only one node
    node_dict = {"node1": node1}

    bs = BeautifulSoup(invalid_graph_xml, "xml")
    graph_node = bs.find("graph")

    # Expect error for missing node
    with pytest.raises(ValueError, match="Target node 'invalid_node' not found in node dictionary"):
        parser.parse_graph(graph_node, node_dict)


def test_parse_graph_empty(parser, empty_graph_xml):
    """Test parsing a graph with no edges."""
    # Create mock nodes
    node1 = Mock()
    node1.id = "node1"

    node2 = Mock()
    node2.id = "node2"

    # Create node dictionary
    node_dict = {"node1": node1, "node2": node2}

    # Parse the graph
    bs = BeautifulSoup(empty_graph_xml, "xml")
    graph_node = bs.find("graph")
    graph = parser.parse_graph(graph_node, node_dict)

    # Verify the graph is empty
    assert isinstance(graph, Graph)
    assert len(graph.nodes_set) == 0
    assert len(graph.adjacency_list) == 0
    assert graph.compiled is True


@pytest.fixture
def tool_node_with_params_conversion_xml():
    """Return XML for a basic tool node with params conversion."""
    xml = """
    <tool_node_with_params_conversion id="node1" name="Test Tool Node">
        <agent name="agent1" />
        <tool name="HumanInputTool" />
    </tool_node_with_params_conversion>
    """
    return BeautifulSoup(xml, "xml").find("tool_node_with_params_conversion")


@pytest.fixture
def tool_node_with_params_conversion_with_options_xml():
    """Return XML for a tool node with params conversion including optional parameters."""
    xml = """
    <tool_node_with_params_conversion id="node1" name="Test Tool Node">
        <agent name="agent1" />
        <tool name="HumanInputTool" />
        <retries>3</retries>
        <clear_memory_after_conversion>false</clear_memory_after_conversion>
    </tool_node_with_params_conversion>
    """
    return BeautifulSoup(xml, "xml").find("tool_node_with_params_conversion")


@pytest.fixture
def tool_node_without_id_xml():
    """Return XML for a tool node with params conversion missing the required id attribute."""
    xml = """
    <tool_node_with_params_conversion name="Test Tool Node">
        <agent name="agent1" />
        <tool name="HumanInputTool" />
    </tool_node_with_params_conversion>
    """
    return BeautifulSoup(xml, "xml").find("tool_node_with_params_conversion")


@pytest.fixture
def tool_node_without_agent_xml():
    """Return XML for a tool node with params conversion missing the agent element."""
    xml = """
    <tool_node_with_params_conversion id="node1" name="Test Tool Node">
        <tool name="HumanInputTool" />
    </tool_node_with_params_conversion>
    """
    return BeautifulSoup(xml, "xml").find("tool_node_with_params_conversion")


@pytest.fixture
def tool_node_with_invalid_agent_xml():
    """Return XML for a tool node with params conversion with an invalid agent reference."""
    xml = """
    <tool_node_with_params_conversion id="node1" name="Test Tool Node">
        <agent name="nonexistent_agent" />
        <tool name="HumanInputTool" />
    </tool_node_with_params_conversion>
    """
    return BeautifulSoup(xml, "xml").find("tool_node_with_params_conversion")


@pytest.fixture
def tool_node_without_tool_xml():
    """Return XML for a tool node with params conversion missing the tool element."""
    xml = """
    <tool_node_with_params_conversion id="node1" name="Test Tool Node">
        <agent name="agent1" />
    </tool_node_with_params_conversion>
    """
    return BeautifulSoup(xml, "xml").find("tool_node_with_params_conversion")


@pytest.fixture
def tool_node_with_invalid_tool_xml():
    """Return XML for a tool node with params conversion with an invalid tool reference."""
    xml = """
    <tool_node_with_params_conversion id="node1" name="Test Tool Node">
        <agent name="agent1" />
        <tool name="NonexistentTool" />
    </tool_node_with_params_conversion>
    """
    return BeautifulSoup(xml, "xml").find("tool_node_with_params_conversion")


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_tool_node_with_params_conversion_valid(
    mock_get_agent_tool_provider, parser, tool_node_with_params_conversion_xml, mock_agent
):
    """Test parsing a valid tool node with params conversion."""
    # Setup
    agents = {"agent1": mock_agent}

    # Create actual tool instance
    human_input_tool = HumanInputTool()

    # Create a mock registry that returns actual tool instance
    tool_registry = Mock()
    tool_registry.get_tool.return_value = human_input_tool

    # Use the real OpenAIToolProvider
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Execute
    tool_node = parser.parse_tool_node_with_params_conversion(
        tool_node_with_params_conversion_xml, agents, tool_registry
    )

    # Verify
    assert isinstance(tool_node, ToolNodeWithParamsConversion)
    assert tool_node.id == "node1"
    assert tool_node.name == "Test Tool Node"
    assert tool_node.agent == mock_agent
    assert tool_node.retries == 1  # Default value
    assert tool_node.clear_memory_after_conversion is True  # Default value
    assert tool_registry.get_tool.call_count == 1
    assert tool_registry.get_tool.call_args[0][0] == "HumanInputTool"


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_tool_node_with_params_conversion_with_options(
    mock_get_agent_tool_provider, parser, tool_node_with_params_conversion_with_options_xml, mock_agent
):
    """Test parsing a tool node with params conversion with optional parameters."""
    # Setup
    agents = {"agent1": mock_agent}

    # Create actual tool instance
    human_input_tool = HumanInputTool()

    # Create a mock registry that returns actual tool instance
    tool_registry = Mock()
    tool_registry.get_tool.return_value = human_input_tool

    # Use the real OpenAIToolProvider
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Execute
    tool_node = parser.parse_tool_node_with_params_conversion(
        tool_node_with_params_conversion_with_options_xml, agents, tool_registry
    )

    # Verify
    assert isinstance(tool_node, ToolNodeWithParamsConversion)
    assert tool_node.id == "node1"
    assert tool_node.retries == 3  # Custom value
    assert tool_node.clear_memory_after_conversion is False  # Custom value


def test_parse_tool_node_without_id(parser, tool_node_without_id_xml):
    """Test parsing a tool node with params conversion missing the id attribute."""
    agents = {"agent1": Mock()}
    tool_registry = Mock()

    with pytest.raises(ValueError, match="Missing required attribute 'id' in tool_node_with_params_conversion"):
        parser.parse_tool_node_with_params_conversion(tool_node_without_id_xml, agents, tool_registry)


def test_parse_tool_node_without_agent(parser, tool_node_without_agent_xml):
    """Test parsing a tool node with params conversion missing the agent element."""
    agents = {"agent1": Mock()}
    tool_registry = Mock()

    with pytest.raises(ValueError, match="Missing agent reference in tool_node_with_params_conversion node1"):
        parser.parse_tool_node_with_params_conversion(tool_node_without_agent_xml, agents, tool_registry)


def test_parse_tool_node_with_invalid_agent(parser, tool_node_with_invalid_agent_xml):
    """Test parsing a tool node with params conversion with an invalid agent reference."""
    agents = {"agent1": Mock()}
    tool_registry = Mock()

    with pytest.raises(
        ValueError, match="Invalid or missing agent name 'nonexistent_agent' in tool_node_with_params_conversion node1"
    ):
        parser.parse_tool_node_with_params_conversion(tool_node_with_invalid_agent_xml, agents, tool_registry)


def test_parse_tool_node_without_tool(parser, tool_node_without_tool_xml):
    """Test parsing a tool node with params conversion missing the tool element."""
    agents = {"agent1": Mock()}
    tool_registry = Mock()

    with pytest.raises(ValueError, match="Missing tool reference in tool_node_with_params_conversion node1"):
        parser.parse_tool_node_with_params_conversion(tool_node_without_tool_xml, agents, tool_registry)


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_tool_node_with_invalid_tool(
    mock_get_agent_tool_provider, parser, tool_node_with_invalid_tool_xml, mock_agent
):
    """Test parsing a tool node with params conversion with an invalid tool reference."""
    # Setup
    agents = {"agent1": mock_agent}

    # Mock the tool registry to return None for nonexistent tool
    tool_registry = Mock()
    tool_registry.get_tool.return_value = None

    # Use the real OpenAIToolProvider
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    with pytest.raises(ValueError, match="Tool 'NonexistentTool' not found in tool registry"):
        parser.parse_tool_node_with_params_conversion(tool_node_with_invalid_tool_xml, agents, tool_registry)


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_nodes_with_tool_node_with_params_conversion(mock_get_agent_tool_provider, parser, mock_agent):
    """Test parsing nodes that include a tool_node_with_params_conversion."""
    # Setup
    xml = """
    <nodes>
        <agent_node id="agent_node1" name="Agent Node">
            <agent name="agent1" />
        </agent_node>
        <tool_node_with_params_conversion id="tool_node1" name="Tool Node">
            <agent name="agent1" />
            <tool name="HumanInputTool" />
            <retries>2</retries>
        </tool_node_with_params_conversion>
    </nodes>
    """
    nodes_bs = BeautifulSoup(xml, "xml").find("nodes")

    # Create mock agents
    agents = {"agent1": mock_agent}

    # Create mock output schemas
    output_schemas = {}

    # Mock the tool provider class
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Configure the tool registry to return real tools
    human_input_tool = HumanInputTool()

    # Create mock registry for parsing tool_node_with_params_conversion
    with patch("nagatoai_core.tool.registry.ToolRegistry.get_tool") as mock_get_tool:
        mock_get_tool.return_value = human_input_tool

        # Execute
        nodes = parser.parse_nodes(nodes_bs, agents, output_schemas)

    # Verify
    assert len(nodes) == 2
    assert "agent_node1" in nodes
    assert "tool_node1" in nodes
    assert isinstance(nodes["agent_node1"], AgentNode)
    assert isinstance(nodes["tool_node1"], ToolNodeWithParamsConversion)
    assert nodes["tool_node1"].retries == 2


@pytest.fixture
def sequential_flow_xml():
    """Return XML for a basic sequential flow with a single agent node."""
    xml = """
    <sequential_flow id="seq_flow_1" name="Sequential Flow">
        <nodes>
            <agent_node id="agent_node1" name="Agent Node">
                <agent name="agent1" />
            </agent_node>
        </nodes>
    </sequential_flow>
    """
    return BeautifulSoup(xml, "xml").find("sequential_flow")


@pytest.fixture
def sequential_flow_with_multiple_nodes_xml():
    """Return XML for a sequential flow with multiple nodes."""
    xml = """
    <sequential_flow id="seq_flow_1" name="Sequential Flow">
        <nodes>
            <agent_node id="agent_node1" name="Agent Node 1">
                <agent name="agent1" />
            </agent_node>
            <tool_node_with_params_conversion id="tool_node1" name="Tool Node">
                <agent name="agent1" />
                <tool name="HumanInputTool" />
            </tool_node_with_params_conversion>
        </nodes>
    </sequential_flow>
    """
    return BeautifulSoup(xml, "xml").find("sequential_flow")


@pytest.fixture
def sequential_flow_with_nested_flow_xml():
    """Return XML for a sequential flow with a nested sequential flow."""
    xml = """
    <sequential_flow id="outer_flow" name="Outer Flow">
        <nodes>
            <agent_node id="agent_node1" name="Agent Node">
                <agent name="agent1" />
            </agent_node>
            <sequential_flow id="inner_flow" name="Inner Flow">
                <nodes>
                    <agent_node id="inner_agent_node" name="Inner Agent Node">
                        <agent name="agent1" />
                    </agent_node>
                </nodes>
            </sequential_flow>
        </nodes>
    </sequential_flow>
    """
    return BeautifulSoup(xml, "xml").find("sequential_flow")


@pytest.fixture
def sequential_flow_without_id_xml():
    """Return XML for a sequential flow missing the required id attribute."""
    xml = """
    <sequential_flow name="Sequential Flow">
        <nodes>
            <agent_node id="agent_node1" name="Agent Node">
                <agent name="agent1" />
            </agent_node>
        </nodes>
    </sequential_flow>
    """
    return BeautifulSoup(xml, "xml").find("sequential_flow")


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_sequential_flow_valid(mock_get_agent_tool_provider, parser, sequential_flow_xml, mock_agent):
    """Test parsing a valid sequential flow with a single agent node."""
    # Setup
    agents = {"agent1": mock_agent}
    output_schemas = {}
    tool_registry = Mock()

    # Mock the tool provider class
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Execute
    # We need to patch the parse_agent_node method since it's called by parse_nodes
    with patch.object(parser, "parse_agent_node") as mock_parse_agent_node:
        mock_agent_node = Mock(spec=AgentNode)
        mock_agent_node.id = "agent_node1"
        mock_parse_agent_node.return_value = mock_agent_node

        sequential_flow = parser.parse_sequential_flow(sequential_flow_xml, agents, output_schemas, tool_registry)

    # Verify
    assert isinstance(sequential_flow, SequentialFlow)
    assert sequential_flow.id == "seq_flow_1"
    assert sequential_flow.name == "Sequential Flow"
    assert len(sequential_flow.nodes) == 1
    assert sequential_flow.nodes[0] == mock_agent_node


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_sequential_flow_with_multiple_nodes(
    mock_get_agent_tool_provider, parser, sequential_flow_with_multiple_nodes_xml, mock_agent
):
    """Test parsing a sequential flow with multiple nodes."""
    # Setup
    agents = {"agent1": mock_agent}
    output_schemas = {}
    tool_registry = Mock()

    # Mock the tool provider class
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Create mocks for the nodes
    mock_agent_node = Mock(spec=AgentNode)
    mock_agent_node.id = "agent_node1"

    mock_tool_node = Mock(spec=ToolNodeWithParamsConversion)
    mock_tool_node.id = "tool_node1"

    # Execute
    with (
        patch.object(parser, "parse_agent_node") as mock_parse_agent_node,
        patch.object(parser, "parse_tool_node_with_params_conversion") as mock_parse_tool_node,
    ):
        mock_parse_agent_node.return_value = mock_agent_node
        mock_parse_tool_node.return_value = mock_tool_node

        sequential_flow = parser.parse_sequential_flow(
            sequential_flow_with_multiple_nodes_xml, agents, output_schemas, tool_registry
        )

    # Verify
    assert isinstance(sequential_flow, SequentialFlow)
    assert sequential_flow.id == "seq_flow_1"
    assert len(sequential_flow.nodes) == 2
    assert sequential_flow.nodes[0] == mock_agent_node
    assert sequential_flow.nodes[1] == mock_tool_node


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_sequential_flow_with_nested_flow(
    mock_get_agent_tool_provider, parser, sequential_flow_with_nested_flow_xml, mock_agent
):
    """Test parsing a sequential flow with a nested sequential flow."""
    # Setup
    agents = {"agent1": mock_agent}
    output_schemas = {}
    tool_registry = Mock()

    # Mock the tool provider class
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Create mocks for the nodes
    mock_agent_node = Mock(spec=AgentNode)
    mock_agent_node.id = "agent_node1"

    mock_inner_agent_node = Mock(spec=AgentNode)
    mock_inner_agent_node.id = "inner_agent_node"

    # The inner flow will be an actual SequentialFlow, not a mock
    inner_flow = SequentialFlow(id="inner_flow", name="Inner Flow", nodes=[mock_inner_agent_node])

    # Execute
    with patch.object(parser, "parse_nodes") as mock_parse_nodes:
        # When parse_nodes is called for the outer flow's nodes,
        # return a dict with the agent_node and inner_flow
        mock_parse_nodes.return_value = {"agent_node1": mock_agent_node, "inner_flow": inner_flow}

        sequential_flow = parser.parse_sequential_flow(
            sequential_flow_with_nested_flow_xml, agents, output_schemas, tool_registry
        )

    # Verify
    assert isinstance(sequential_flow, SequentialFlow)
    assert sequential_flow.id == "outer_flow"
    assert len(sequential_flow.nodes) == 2
    assert sequential_flow.nodes[0] == mock_agent_node
    assert sequential_flow.nodes[1].id == "inner_flow"
    assert isinstance(sequential_flow.nodes[1], SequentialFlow)
    assert len(sequential_flow.nodes[1].nodes) == 1
    assert sequential_flow.nodes[1].nodes[0] == mock_inner_agent_node


def test_parse_sequential_flow_without_id(parser, sequential_flow_without_id_xml):
    """Test parsing a sequential flow missing the id attribute."""
    agents = {"agent1": Mock()}
    output_schemas = {}
    tool_registry = Mock()

    with pytest.raises(ValueError, match="Missing required attribute 'id' in sequential_flow"):
        parser.parse_sequential_flow(sequential_flow_without_id_xml, agents, output_schemas, tool_registry)


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_nodes_with_sequential_flow(mock_get_agent_tool_provider, parser, mock_agent):
    """Test parsing nodes that include a sequential flow."""
    # Setup
    xml = """
    <nodes>
        <agent_node id="agent_node1" name="Agent Node">
            <agent name="agent1" />
        </agent_node>
        <sequential_flow id="seq_flow" name="Sequential Flow">
            <nodes>
                <agent_node id="flow_agent_node" name="Flow Agent Node">
                    <agent name="agent1" />
                </agent_node>
            </nodes>
        </sequential_flow>
    </nodes>
    """
    nodes_bs = BeautifulSoup(xml, "xml").find("nodes")

    # Create mock agents
    agents = {"agent1": mock_agent}

    # Create mock output schemas
    output_schemas = {}

    # Mock the tool provider class
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Execute
    # We need to patch the necessary methods to avoid deep recursion
    with patch.object(parser, "parse_agent_node") as mock_parse_agent_node:
        # Create mock agent nodes
        mock_agent_node1 = Mock(spec=AgentNode)
        mock_agent_node1.id = "agent_node1"

        mock_flow_agent_node = Mock(spec=AgentNode)
        mock_flow_agent_node.id = "flow_agent_node"

        # Configure the mock to return different values for different calls
        mock_parse_agent_node.side_effect = lambda *args, **kwargs: (
            mock_agent_node1 if args[0].get("id") == "agent_node1" else mock_flow_agent_node
        )

        nodes = parser.parse_nodes(nodes_bs, agents, output_schemas)

    # Verify
    # With the current implementation, we get 3 nodes:
    # 1. The top-level agent node
    # 2. The sequential flow node
    # 3. The agent node inside the sequential flow is also added to the top level
    assert len(nodes) == 3
    assert "agent_node1" in nodes
    assert "seq_flow" in nodes
    assert "flow_agent_node" in nodes
    assert isinstance(nodes["agent_node1"], Mock)  # It's our mock AgentNode
    assert isinstance(nodes["seq_flow"], SequentialFlow)
    assert len(nodes["seq_flow"].nodes) == 1
    assert nodes["seq_flow"].nodes[0].id == "flow_agent_node"


@pytest.fixture
def parallel_flow_xml():
    """Return XML for a basic parallel flow with a single agent node."""
    xml = """
    <parallel_flow id="parallel_flow_1" name="Parallel Flow">
        <nodes>
            <agent_node id="agent_node1" name="Agent Node">
                <agent name="agent1" />
            </agent_node>
        </nodes>
    </parallel_flow>
    """
    return BeautifulSoup(xml, "xml").find("parallel_flow")


@pytest.fixture
def parallel_flow_with_multiple_nodes_xml():
    """Return XML for a parallel flow with multiple nodes."""
    xml = """
    <parallel_flow id="parallel_flow_1" name="Parallel Flow">
        <nodes>
            <agent_node id="agent_node1" name="Agent Node 1">
                <agent name="agent1" />
            </agent_node>
            <tool_node_with_params_conversion id="tool_node1" name="Tool Node">
                <agent name="agent1" />
                <tool name="HumanInputTool" />
            </tool_node_with_params_conversion>
        </nodes>
    </parallel_flow>
    """
    return BeautifulSoup(xml, "xml").find("parallel_flow")


@pytest.fixture
def parallel_flow_with_custom_workers_xml():
    """Return XML for a parallel flow with custom max_workers value."""
    xml = """
    <parallel_flow id="parallel_flow_1" name="Parallel Flow">
        <max_workers>8</max_workers>
        <nodes>
            <agent_node id="agent_node1" name="Agent Node">
                <agent name="agent1" />
            </agent_node>
        </nodes>
    </parallel_flow>
    """
    return BeautifulSoup(xml, "xml").find("parallel_flow")


@pytest.fixture
def parallel_flow_with_nested_flow_xml():
    """Return XML for a parallel flow with a nested sequential flow."""
    xml = """
    <parallel_flow id="outer_flow" name="Outer Flow">
        <nodes>
            <agent_node id="agent_node1" name="Agent Node">
                <agent name="agent1" />
            </agent_node>
            <sequential_flow id="inner_flow" name="Inner Flow">
                <nodes>
                    <agent_node id="inner_agent_node" name="Inner Agent Node">
                        <agent name="agent1" />
                    </agent_node>
                </nodes>
            </sequential_flow>
        </nodes>
    </parallel_flow>
    """
    return BeautifulSoup(xml, "xml").find("parallel_flow")


@pytest.fixture
def parallel_flow_without_id_xml():
    """Return XML for a parallel flow missing the required id attribute."""
    xml = """
    <parallel_flow name="Parallel Flow">
        <nodes>
            <agent_node id="agent_node1" name="Agent Node">
                <agent name="agent1" />
            </agent_node>
        </nodes>
    </parallel_flow>
    """
    return BeautifulSoup(xml, "xml").find("parallel_flow")


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_parallel_flow_valid(mock_get_agent_tool_provider, parser, parallel_flow_xml, mock_agent):
    """Test parsing a valid parallel flow with a single agent node."""
    # Setup
    agents = {"agent1": mock_agent}
    output_schemas = {}
    tool_registry = Mock()

    # Mock the tool provider class
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Execute
    # We need to patch the parse_agent_node method since it's called by parse_nodes
    with patch.object(parser, "parse_agent_node") as mock_parse_agent_node:
        mock_agent_node = Mock(spec=AgentNode)
        mock_agent_node.id = "agent_node1"
        mock_parse_agent_node.return_value = mock_agent_node

        parallel_flow = parser.parse_parallel_flow(parallel_flow_xml, agents, output_schemas, tool_registry)

    # Verify
    assert isinstance(parallel_flow, ParallelFlow)
    assert parallel_flow.id == "parallel_flow_1"
    assert parallel_flow.name == "Parallel Flow"
    assert len(parallel_flow.nodes) == 1
    assert parallel_flow.nodes[0] == mock_agent_node
    assert parallel_flow.max_workers == 4  # Default value


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_parallel_flow_with_multiple_nodes(
    mock_get_agent_tool_provider, parser, parallel_flow_with_multiple_nodes_xml, mock_agent
):
    """Test parsing a parallel flow with multiple nodes."""
    # Setup
    agents = {"agent1": mock_agent}
    output_schemas = {}
    tool_registry = Mock()

    # Mock the tool provider class
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Create mocks for the nodes
    mock_agent_node = Mock(spec=AgentNode)
    mock_agent_node.id = "agent_node1"

    mock_tool_node = Mock(spec=ToolNodeWithParamsConversion)
    mock_tool_node.id = "tool_node1"

    # Execute
    with (
        patch.object(parser, "parse_agent_node") as mock_parse_agent_node,
        patch.object(parser, "parse_tool_node_with_params_conversion") as mock_parse_tool_node,
    ):
        mock_parse_agent_node.return_value = mock_agent_node
        mock_parse_tool_node.return_value = mock_tool_node

        parallel_flow = parser.parse_parallel_flow(
            parallel_flow_with_multiple_nodes_xml, agents, output_schemas, tool_registry
        )

    # Verify
    assert isinstance(parallel_flow, ParallelFlow)
    assert parallel_flow.id == "parallel_flow_1"
    assert len(parallel_flow.nodes) == 2
    assert parallel_flow.nodes[0] == mock_agent_node
    assert parallel_flow.nodes[1] == mock_tool_node


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_parallel_flow_with_custom_workers(
    mock_get_agent_tool_provider, parser, parallel_flow_with_custom_workers_xml, mock_agent
):
    """Test parsing a parallel flow with custom max_workers value."""
    # Setup
    agents = {"agent1": mock_agent}
    output_schemas = {}
    tool_registry = Mock()

    # Mock the tool provider class
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Execute
    with patch.object(parser, "parse_agent_node") as mock_parse_agent_node:
        mock_agent_node = Mock(spec=AgentNode)
        mock_agent_node.id = "agent_node1"
        mock_parse_agent_node.return_value = mock_agent_node

        parallel_flow = parser.parse_parallel_flow(
            parallel_flow_with_custom_workers_xml, agents, output_schemas, tool_registry
        )

    # Verify
    assert isinstance(parallel_flow, ParallelFlow)
    assert parallel_flow.max_workers == 8  # Custom value


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_parallel_flow_with_nested_flow(
    mock_get_agent_tool_provider, parser, parallel_flow_with_nested_flow_xml, mock_agent
):
    """Test parsing a parallel flow with a nested sequential flow."""
    # Setup
    agents = {"agent1": mock_agent}
    output_schemas = {}
    tool_registry = Mock()

    # Mock the tool provider class
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Create mocks for the nodes
    mock_agent_node = Mock(spec=AgentNode)
    mock_agent_node.id = "agent_node1"

    mock_inner_agent_node = Mock(spec=AgentNode)
    mock_inner_agent_node.id = "inner_agent_node"

    # The inner flow will be an actual SequentialFlow, not a mock
    inner_flow = SequentialFlow(id="inner_flow", name="Inner Flow", nodes=[mock_inner_agent_node])

    # Execute
    with patch.object(parser, "parse_nodes") as mock_parse_nodes:
        # When parse_nodes is called for the outer flow's nodes,
        # return a dict with the agent_node and inner_flow
        mock_parse_nodes.return_value = {"agent_node1": mock_agent_node, "inner_flow": inner_flow}

        parallel_flow = parser.parse_parallel_flow(
            parallel_flow_with_nested_flow_xml, agents, output_schemas, tool_registry
        )

    # Verify
    assert isinstance(parallel_flow, ParallelFlow)
    assert parallel_flow.id == "outer_flow"
    assert len(parallel_flow.nodes) == 2
    assert parallel_flow.nodes[0] == mock_agent_node
    assert parallel_flow.nodes[1].id == "inner_flow"
    assert isinstance(parallel_flow.nodes[1], SequentialFlow)
    assert len(parallel_flow.nodes[1].nodes) == 1
    assert parallel_flow.nodes[1].nodes[0] == mock_inner_agent_node


def test_parse_parallel_flow_without_id(parser, parallel_flow_without_id_xml):
    """Test parsing a parallel flow missing the id attribute."""
    agents = {"agent1": Mock()}
    output_schemas = {}
    tool_registry = Mock()

    with pytest.raises(ValueError, match="Missing required attribute 'id' in parallel_flow"):
        parser.parse_parallel_flow(parallel_flow_without_id_xml, agents, output_schemas, tool_registry)


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_nodes_with_parallel_flow(mock_get_agent_tool_provider, parser, mock_agent):
    """Test parsing nodes that include a parallel flow."""
    # Setup
    xml = """
    <nodes>
        <agent_node id="agent_node1" name="Agent Node">
            <agent name="agent1" />
        </agent_node>
        <parallel_flow id="parallel_flow" name="Parallel Flow">
            <nodes>
                <agent_node id="flow_agent_node" name="Flow Agent Node">
                    <agent name="agent1" />
                </agent_node>
            </nodes>
        </parallel_flow>
    </nodes>
    """
    nodes_bs = BeautifulSoup(xml, "xml").find("nodes")

    # Create mock agents
    agents = {"agent1": mock_agent}

    # Create mock output schemas
    output_schemas = {}

    # Mock the tool provider class
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Execute
    # We need to patch the necessary methods to avoid deep recursion
    with patch("nagatoai_core.graph.plan.xml_parser.XMLPlanParser.parse_agent_node") as mock_parse_agent_node:
        # Create mock agent nodes
        mock_agent_node1 = Mock(spec=AgentNode)
        mock_agent_node1.id = "agent_node1"

        mock_flow_agent_node = Mock(spec=AgentNode)
        mock_flow_agent_node.id = "flow_agent_node"

        # Configure the mock to return different values for different calls
        mock_parse_agent_node.side_effect = lambda *args, **kwargs: (
            mock_agent_node1 if args[0].get("id") == "agent_node1" else mock_flow_agent_node
        )

        nodes = parser.parse_nodes(nodes_bs, agents, output_schemas)

    # Verify
    # With the current implementation, we get 3 nodes:
    # 1. The top-level agent node
    # 2. The parallel flow node
    # 3. The agent node inside the parallel flow is also added to the top level
    assert len(nodes) == 3
    assert "agent_node1" in nodes
    assert "parallel_flow" in nodes
    assert "flow_agent_node" in nodes
    assert isinstance(nodes["agent_node1"], Mock)  # It's our mock AgentNode
    assert isinstance(nodes["parallel_flow"], ParallelFlow)
    assert len(nodes["parallel_flow"].nodes) == 1
    assert nodes["parallel_flow"].nodes[0].id == "flow_agent_node"


@pytest.fixture
def conditional_flow_xml():
    """Fixture for a basic conditional flow XML."""
    return BeautifulSoup(
        """
        <conditional_flow id="test_flow" name="Test Conditional Flow">
            <broadcast_comparison>false</broadcast_comparison>
            <input_index>0</input_index>
            <input_attribute>status</input_attribute>
            <comparison_value>active</comparison_value>
            <comparison_type>EQUAL</comparison_type>
            <predicate_join_type>AND</predicate_join_type>
            <positive_path>
                <agent_node id="positive_node" name="Positive Node">
                    <agent name="test_agent"/>
                    <temperature>0.7</temperature>
                    <max_tokens>150</max_tokens>
                </agent_node>
            </positive_path>
            <negative_path>
                <agent_node id="negative_node" name="Negative Node">
                    <agent name="test_agent"/>
                    <temperature>0.5</temperature>
                    <max_tokens>100</max_tokens>
                </agent_node>
            </negative_path>
        </conditional_flow>
        """,
        "xml",
    ).find("conditional_flow")


@pytest.fixture
def conditional_flow_custom_comparison_xml():
    """Fixture for a conditional flow XML with custom comparison function."""
    return BeautifulSoup(
        """
        <conditional_flow id="custom_flow" name="Custom Comparison Flow">
            <broadcast_comparison>true</broadcast_comparison>
            <input_index>0</input_index>
            <comparison_value>test</comparison_value>
            <custom_comparison_function>
                <module>nagatoai_core.graph.comparison_functions</module>
                <function>starts_with</function>
            </custom_comparison_function>
            <predicate_join_type>OR</predicate_join_type>
            <positive_path>
                <agent_node id="positive_node" name="Positive Node">
                    <agent name="test_agent"/>
                    <temperature>0.7</temperature>
                    <max_tokens>150</max_tokens>
                </agent_node>
            </positive_path>
        </conditional_flow>
        """,
        "xml",
    ).find("conditional_flow")


@pytest.fixture
def conditional_flow_nested_xml():
    """Fixture for a conditional flow XML with nested flows."""

    xml_string = """
    <conditional_flow id="nested_flow" name="Nested Flow">
        <comparison_value>10</comparison_value>
        <comparison_type>GREATER_THAN</comparison_type>
        <positive_path>
            <conditional_flow id="inner_flow" name="Inner Flow">
                <comparison_value>20</comparison_value>
                <comparison_type>LESS_THAN</comparison_type>
                <positive_path>
                    <agent_node id="inner_positive_node" name="Inner Positive">
                        <agent name="test_agent"/>
                    </agent_node>
                </positive_path>
                <negative_path>
                    <agent_node id="inner_negative_node" name="Inner Negative">
                        <agent name="test_agent"/>
                    </agent_node>
                </negative_path>
            </conditional_flow>
        </positive_path>
        <negative_path>
            <sequential_flow id="seq_flow" name="Sequential Flow">
                <nodes>
                    <agent_node id="seq_node1" name="Seq Node 1">
                        <agent name="test_agent"/>
                    </agent_node>
                    <agent_node id="seq_node2" name="Seq Node 2">
                        <agent name="test_agent"/>
                    </agent_node>
                </nodes>
            </sequential_flow>
        </negative_path>
    </conditional_flow>
    """
    logging.info(f"Original XML string: {xml_string}")

    # Try using lxml parser instead of xml parser
    soup = BeautifulSoup(xml_string, "lxml-xml")
    result = soup.find("conditional_flow")

    # Debug the structure of the negative path
    negative_path = result.find("negative_path", recursive=False)
    logging.info(f"negative_path_elem: {negative_path}")
    seq_flow = negative_path.find("sequential_flow", recursive=False) if negative_path else None
    logging.info(f"sequential_flow in negative path: {seq_flow}")

    if seq_flow:
        logging.info(f"sequential_flow id: {seq_flow.get('id')}")
        nodes_elem = seq_flow.find("nodes", recursive=False)
        logging.info(f"nodes element: {nodes_elem}")
        if nodes_elem:
            agent_nodes = nodes_elem.find_all("agent_node", recursive=False)
            logging.info(f"Number of agent nodes: {len(agent_nodes)}")
            for i, node in enumerate(agent_nodes):
                logging.info(f"Agent node {i} id: {node.get('id')}")

    # Create an XML string which is known to parse correctly
    logging.info(f"Result structure: {result}")

    return result


@pytest.fixture
def conditional_flow_without_id_xml():
    """Fixture for a conditional flow XML without an ID."""
    return BeautifulSoup(
        """
        <conditional_flow name="No ID Flow">
            <comparison_value>active</comparison_value>
            <comparison_type>EQUAL</comparison_type>
            <positive_path>
                <agent_node id="positive_node" name="Positive Node">
                    <agent name="test_agent"/>
                </agent_node>
            </positive_path>
        </conditional_flow>
        """,
        "xml",
    ).find("conditional_flow")


@pytest.fixture
def conditional_flow_without_paths_xml():
    """Fixture for a conditional flow XML without any paths."""
    return BeautifulSoup(
        """
        <conditional_flow id="no_paths_flow" name="No Paths Flow">
            <comparison_value>active</comparison_value>
            <comparison_type>EQUAL</comparison_type>
        </conditional_flow>
        """,
        "xml",
    ).find("conditional_flow")


@pytest.fixture
def mock_starts_with_function():
    """Mock function for testing custom comparison functions."""

    def starts_with(value, prefix):
        if isinstance(value, str):
            return value.startswith(prefix)
        return False

    return starts_with


@patch("nagatoai_core.graph.plan.xml_parser.importlib.import_module")
def test_parse_conditional_flow_with_custom_comparison(
    mock_import_module,
    parser,
    conditional_flow_custom_comparison_xml,
    mock_starts_with_function,
):
    """Test parsing a conditional flow XML with custom comparison function."""
    # Create a mock that will pass isinstance checks for OpenAIAgent
    mock_agent = Mock(spec=OpenAIAgent)
    # Make this mock pass isinstance checks with OpenAIAgent
    mock_agent.__class__ = OpenAIAgent

    agents = {"test_agent": mock_agent}
    output_schemas = {}

    # Mock the module import to return our mock function
    mock_module = type("module", (), {})()
    mock_module.starts_with = mock_starts_with_function
    mock_import_module.return_value = mock_module

    # Execute
    result = parser.parse_conditional_flow(conditional_flow_custom_comparison_xml, agents, output_schemas, None)

    # Verify
    assert isinstance(result, ConditionalFlow)
    assert result.id == "custom_flow"
    assert result.broadcast_comparison is True
    assert result.comparison_value == "test"
    assert result.custom_comparison_fn is not None
    assert result.predicate_join_type == PredicateJoinType.OR
    assert result.positive_path is not None
    assert result.positive_path.id == "positive_node"
    assert result.negative_path is None


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_conditional_flow_nested(mock_get_agent_tool_provider, parser, conditional_flow_nested_xml):
    """Test parsing a conditional flow XML with nested flows."""
    # Setup

    # Create a mock that will pass isinstance checks for OpenAIAgent
    mock_agent = Mock(spec=OpenAIAgent)
    mock_agent.__class__ = OpenAIAgent

    agents = {"test_agent": mock_agent}
    output_schemas = {}

    # Make sure the tool provider returns a proper OpenAIToolProvider
    mock_get_agent_tool_provider.return_value = OpenAIToolProvider

    # Debug the XML structure
    negative_path_elem = conditional_flow_nested_xml.find("negative_path", recursive=False)
    logging.info(f"negative_path_elem: {negative_path_elem}")
    seq_flow = negative_path_elem.find("sequential_flow", recursive=False) if negative_path_elem else None
    logging.info(f"sequential_flow in negative path: {seq_flow}")

    if seq_flow:
        logging.info(f"sequential_flow id: {seq_flow.get('id')}")
        nodes_elem = seq_flow.find("nodes", recursive=False)
        logging.info(f"nodes element: {nodes_elem}")
        if nodes_elem:
            agent_nodes = nodes_elem.find_all("agent_node", recursive=False)
            logging.info(f"Number of agent nodes: {len(agent_nodes)}")
            for i, node in enumerate(agent_nodes):
                logging.info(f"Agent node {i} id: {node.get('id')}")

    # Execute
    result = parser.parse_conditional_flow(conditional_flow_nested_xml, agents, output_schemas, None)

    # Verify
    assert isinstance(result, ConditionalFlow)
    assert result.id == "nested_flow"
    assert result.comparison_value == "10"
    assert result.comparison_type == ComparisonType.GREATER_THAN

    # Verify positive path is a nested conditional flow
    assert isinstance(result.positive_path, ConditionalFlow)
    assert result.positive_path.id == "inner_flow"
    assert result.positive_path.comparison_value == "20"
    assert result.positive_path.comparison_type == ComparisonType.LESS_THAN
    assert result.positive_path.positive_path.id == "inner_positive_node"
    assert result.positive_path.negative_path.id == "inner_negative_node"

    # Log what's actually in the negative path
    logging.info(f"Negative path type: {type(result.negative_path)}")
    logging.info(f"Negative path id: {result.negative_path.id if hasattr(result.negative_path, 'id') else None}")

    # Fix the test with proper assertion for what should be there
    assert isinstance(result.negative_path, SequentialFlow)
    assert result.negative_path.id == "seq_flow"


def test_parse_conditional_flow_without_id(parser, conditional_flow_without_id_xml):
    """Test parsing a conditional flow XML without an ID throws an error."""
    with pytest.raises(ValueError, match="Missing required attribute 'id' in conditional_flow"):
        parser.parse_conditional_flow(conditional_flow_without_id_xml, {}, {}, None)


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
def test_parse_nodes_with_conditional_flow(mock_get_agent_tool_provider, parser, mock_agent):
    """Test parsing nodes that include a conditional flow."""
    # Setup
    agents = {"test_agent": mock_agent}
    output_schemas = {}
    mock_get_agent_tool_provider.return_value = lambda **kwargs: None

    nodes_xml = BeautifulSoup(
        """
        <nodes>
            <conditional_flow id="test_flow" name="Test Flow">
                <comparison_value>active</comparison_value>
                <comparison_type>EQUAL</comparison_type>
                <positive_path>
                    <agent_node id="positive_node" name="Positive Node">
                        <agent name="test_agent"/>
                    </agent_node>
                </positive_path>
                <negative_path>
                    <agent_node id="negative_node" name="Negative Node">
                        <agent name="test_agent"/>
                    </agent_node>
                </negative_path>
            </conditional_flow>
        </nodes>
        """,
        "xml",
    )

    # Execute
    result = parser.parse_nodes(nodes_xml, agents, output_schemas)

    # Verify
    assert "test_flow" in result
    assert isinstance(result["test_flow"], ConditionalFlow)
    assert result["test_flow"].comparison_value == "active"
    assert result["test_flow"].comparison_type == ComparisonType.EQUAL
    assert result["test_flow"].positive_path.id == "positive_node"
    assert result["test_flow"].negative_path.id == "negative_node"


@patch("nagatoai_core.graph.plan.xml_parser.get_agent_tool_provider")
@patch("nagatoai_core.graph.plan.xml_parser.importlib.import_module")
def test_get_comparison_function(mock_import_module, mock_get_agent_tool_provider, parser, mock_starts_with_function):
    """Test the _get_comparison_function method."""
    # Setup
    mock_module = type("module", (), {})()
    mock_module.starts_with = mock_starts_with_function
    mock_import_module.return_value = mock_module

    # Execute
    func = parser._get_comparison_function("test_module", "starts_with")

    # Verify
    assert func is mock_starts_with_function
    mock_import_module.assert_any_call("test_module")


@patch("nagatoai_core.graph.plan.xml_parser.importlib.import_module")
def test_get_comparison_function_import_error(mock_import_module, parser):
    """Test the _get_comparison_function method when import fails."""
    # Setup
    mock_import_module.side_effect = ImportError("Module not found")

    # Execute and verify
    with pytest.raises(ValueError, match="Failed to import comparison function"):
        parser._get_comparison_function("non_existent_module", "some_function")


@patch("nagatoai_core.graph.plan.xml_parser.importlib.import_module")
def test_get_comparison_function_attribute_error(mock_import_module, parser):
    """Test the _get_comparison_function method when function is not found."""
    # Setup
    mock_module = type("module", (), {})()
    mock_import_module.return_value = mock_module

    # Execute and verify
    with pytest.raises(ValueError, match="Failed to import comparison function"):
        parser._get_comparison_function("test_module", "non_existent_function")
