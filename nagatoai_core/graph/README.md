# Graph Package

The graph package provides a flexible and powerful framework for building directed acyclic graphs (DAGs) of nodes that can be executed in a controlled sequence. It's designed to handle complex workflows involving AI agents, tools, and data transformations.

## Overview

The graph package implements a DAG-based execution model where:

1. Nodes represent individual units of work (agents, tools, or flows)
2. Edges define dependencies and execution order
3. Data flows between nodes through `NodeResult` objects
4. The graph ensures proper execution order and handles data passing

## Core Components

### Graph

The `Graph` class is the central component that:
- Manages node relationships and execution order
- Ensures the graph is a valid DAG (no cycles)
- Handles data flow between nodes
- Executes nodes in the correct order

### Node Types

1. **AbstractNode**: Base class for all nodes with common functionality
2. **AgentNode**: Executes AI agents with configurable parameters
3. **ToolNode**: Executes tools with parameter conversion capabilities
4. **Flow Nodes**: Special nodes that manage subgraphs:
   - `SequentialFlow`: Executes nodes in sequence
   - `ParallelFlow`: Executes nodes in parallel
   - `ConditionalFlow`: Executes nodes based on conditions
   - `TransformerFlow`: Transforms data using custom functions
   - `UnfoldFlow`: Expands lists into individual items

## XML Plan Parser

The package includes an XML parser that allows you to define graphs declaratively. Here's how to use it:

```python
from nagatoai_core.graph.plan.xml_parser import XMLPlanParser

# Define your plan in XML
xml_str = """
<plan>
    <agents>
        <agent name="agent1">
            <model>gpt-4</model>
            <role>Assistant</role>
            <role_description>Helpful AI assistant</role_description>
            <nickname>helper</nickname>
        </agent>
    </agents>

    <output_schemas>
        <output_schema name="Person">
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        </output_schema>
    </output_schemas>

    <nodes>
        <agent_node id="node1" name="First Node">
            <agent name="agent1"/>
            <temperature>0.7</temperature>
            <max_tokens>150</max_tokens>
        </agent_node>

        <sequential_flow id="flow1" name="Main Flow">
            <nodes>
                <agent_node id="node2" name="Second Node">
                    <agent name="agent1"/>
                </agent_node>
                <tool_node_with_params_conversion id="tool1" name="Tool Node">
                    <agent name="agent1"/>
                    <tool name="HumanInputTool"/>
                </tool_node_with_params_conversion>
            </nodes>
        </sequential_flow>
    </nodes>

    <graph>
        <edges>
            <edge from="node1" to="flow1"/>
        </edges>
    </graph>
</plan>
"""

# Parse the XML into a Plan
parser = XMLPlanParser()
plan = parser.parse(xml_str)

# The plan contains:
# - agents: Dictionary of configured agents
# - output_schemas: Dictionary of Pydantic models for data validation
# - graph: The execution graph with nodes and edges
```

## Flow Types

### Sequential Flow
Executes nodes in sequence, passing results from one node to the next.

### Parallel Flow
Executes multiple nodes in parallel with configurable worker count.

### Conditional Flow
Executes different paths based on conditions:
- Supports standard comparisons (EQUAL, GREATER_THAN, etc.)
- Can use custom comparison functions
- Has positive and negative paths
- Can broadcast comparison results

### Transformer Flow
Combines node results with another flow through a transformation function:
- Useful for injecting subgraphs
- Supports custom transformation logic
- Maintains original inputs alongside flow results

### Unfold Flow
Expands lists into individual items:
- Preserves metadata (optional)
- Handles empty lists
- Can wrap non-list inputs

## Best Practices

1. **Graph Structure**
   - Keep graphs acyclic (no cycles)
   - Avoid isolated nodes
   - Use meaningful node IDs and names

2. **Error Handling**
   - Nodes should handle their own errors
   - Use try-except blocks in node execution
   - Return error information in NodeResult

3. **Performance**
   - Use parallel flows for independent operations
   - Consider using transformer flows for complex transformations
   - Use unfold flows for batch processing

4. **XML Plans**
   - Validate XML before parsing
   - Use descriptive names and IDs
   - Document complex flows
   - Test plans with small examples first

## Example Usage

```python
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.types import NodeResult

# Create a graph
graph = Graph()

# Add nodes
node1 = AgentNode(id="node1", name="First Node")
node2 = ToolNode(id="node2", name="Tool Node")
node3 = AgentNode(id="node3", name="Final Node")

# Add edges
graph.add_edge(node1, node2)
graph.add_edge(node2, node3)

# Compile the graph
graph.compile()

# Run the graph with initial inputs
inputs = [NodeResult(node_id="node1", result="initial data")]
results = graph.run(inputs)
```

## Testing

The package includes comprehensive tests for:
- Graph compilation and cycle detection
- Node execution and data flow
- XML parsing and validation
- Flow types and their behaviors
- Error handling and edge cases

Run tests using pytest:
```bash
poetry run pytest tests/nagatoai_core/graph/
```
