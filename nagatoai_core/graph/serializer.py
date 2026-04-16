from __future__ import annotations

# Standard Library
import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import Optional

# Nagato AI
from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.agent_node import AgentNode
from nagatoai_core.graph.conditional_flow import ConditionalFlow
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.parallel_flow import ParallelFlow
from nagatoai_core.graph.sequential_flow import SequentialFlow
from nagatoai_core.graph.tool_node_with_params_conversion import ToolNodeWithParamsConversion
from nagatoai_core.graph.transformer_flow import TransformerFlow
from nagatoai_core.graph.unfold_flow import UnfoldFlow


def serialize_graph(graph: Graph) -> str:
    """
    Serialize a Graph to a human-readable XML string.

    :param graph: The graph to serialize.
    :returns: An XML string representation of the graph.
    """
    plan_elem = ET.Element("plan")
    nodes_elem = ET.SubElement(plan_elem, "nodes")

    for node_id in graph.execution_order:
        node = graph.node_map[node_id]
        _serialize_node(nodes_elem, node)

    for node_id, node in graph.node_map.items():
        if node_id not in graph.execution_order:
            _serialize_node(nodes_elem, node)

    graph_elem = ET.SubElement(plan_elem, "graph")
    edges_elem = ET.SubElement(graph_elem, "edges")
    for source_id, target_ids in graph.adjacency_list.items():
        for target_id in target_ids:
            edge_elem = ET.SubElement(edges_elem, "edge")
            edge_elem.set("from", source_id)
            edge_elem.set("to", target_id)

    rough_xml = ET.tostring(plan_elem, encoding="unicode")
    dom = xml.dom.minidom.parseString(rough_xml)
    return dom.toprettyxml(indent="  ", encoding=None)


def _serialize_node(parent: ET.Element, node: AbstractNode) -> None:
    """
    Serialize a single node into the parent XML element.

    :param parent: The parent XML element to append to.
    :param node: The node to serialize.
    """
    if isinstance(node, ToolNodeWithParamsConversion):
        _serialize_tool_node_with_params_conversion(parent, node)
    elif isinstance(node, AgentNode):
        _serialize_agent_node(parent, node)
    elif isinstance(node, SequentialFlow):
        _serialize_sequential_flow(parent, node)
    elif isinstance(node, ParallelFlow):
        _serialize_parallel_flow(parent, node)
    elif isinstance(node, ConditionalFlow):
        _serialize_conditional_flow(parent, node)
    elif isinstance(node, TransformerFlow):
        _serialize_transformer_flow(parent, node)
    elif isinstance(node, UnfoldFlow):
        _serialize_unfold_flow(parent, node)
    else:
        elem = ET.SubElement(parent, "node")
        elem.set("id", node.id)
        elem.set("type", type(node).__name__)
        if node.name:
            elem.set("name", node.name)


def _serialize_agent_node(parent: ET.Element, node: AgentNode) -> None:
    elem = ET.SubElement(parent, "agent_node")
    elem.set("id", node.id)
    if node.name:
        elem.set("name", node.name)

    model_elem = ET.SubElement(elem, "model")
    model_elem.text = node.agent.model

    if node.prompt_template:
        prompt_elem = ET.SubElement(elem, "prompt")
        prompt_elem.text = node.prompt_template.template

    temp_elem = ET.SubElement(elem, "temperature")
    temp_elem.text = str(node.temperature)

    max_tokens_elem = ET.SubElement(elem, "max_tokens")
    max_tokens_elem.text = str(node.max_tokens)


def _serialize_tool_node_with_params_conversion(parent: ET.Element, node: ToolNodeWithParamsConversion) -> None:
    elem = ET.SubElement(parent, "tool_node_with_params_conversion")
    elem.set("id", node.id)
    if node.name:
        elem.set("name", node.name)

    agent_elem = ET.SubElement(elem, "agent")
    agent_elem.set("name", node.agent.name)

    tool_elem = ET.SubElement(elem, "tool")
    tool_elem.set("name", node.tool_provider.tool.name)

    retries_elem = ET.SubElement(elem, "retries")
    retries_elem.text = str(node.retries)

    clear_elem = ET.SubElement(elem, "clear_memory_after_conversion")
    clear_elem.text = str(node.clear_memory_after_conversion).lower()


def _serialize_sequential_flow(parent: ET.Element, node: SequentialFlow) -> None:
    elem = ET.SubElement(parent, "sequential_flow")
    elem.set("id", node.id)
    if node.name:
        elem.set("name", node.name)
    nodes_elem = ET.SubElement(elem, "nodes")
    for child in node.nodes:
        _serialize_node(nodes_elem, child)


def _serialize_parallel_flow(parent: ET.Element, node: ParallelFlow) -> None:
    elem = ET.SubElement(parent, "parallel_flow")
    elem.set("id", node.id)
    if node.name:
        elem.set("name", node.name)
    workers_elem = ET.SubElement(elem, "max_workers")
    workers_elem.text = str(node.max_workers)
    nodes_elem = ET.SubElement(elem, "nodes")
    for child in node.nodes:
        _serialize_node(nodes_elem, child)


def _serialize_conditional_flow(parent: ET.Element, node: ConditionalFlow) -> None:
    elem = ET.SubElement(parent, "conditional_flow")
    elem.set("id", node.id)
    if node.name:
        elem.set("name", node.name)

    bc_elem = ET.SubElement(elem, "broadcast_comparison")
    bc_elem.text = str(node.broadcast_comparison).lower()

    idx_elem = ET.SubElement(elem, "input_index")
    idx_elem.text = str(node.input_index)

    if node.input_attribute is not None:
        attr_elem = ET.SubElement(elem, "input_attribute")
        attr_elem.text = node.input_attribute

    if node.comparison_value is not None:
        val_elem = ET.SubElement(elem, "comparison_value")
        val_elem.text = str(node.comparison_value)

    ct_elem = ET.SubElement(elem, "comparison_type")
    ct_elem.text = node.comparison_type.value

    pj_elem = ET.SubElement(elem, "predicate_join_type")
    pj_elem.text = node.predicate_join_type.value

    if node.positive_path is not None:
        pp_elem = ET.SubElement(elem, "positive_path")
        _serialize_node(pp_elem, node.positive_path)

    if node.negative_path is not None:
        np_elem = ET.SubElement(elem, "negative_path")
        _serialize_node(np_elem, node.negative_path)


def _serialize_transformer_flow(parent: ET.Element, node: TransformerFlow) -> None:
    elem = ET.SubElement(parent, "transformer_flow")
    elem.set("id", node.id)
    if node.name:
        elem.set("name", node.name)
    fp_elem = ET.SubElement(elem, "flow_param")
    _serialize_node(fp_elem, node.flow_param)


def _serialize_unfold_flow(parent: ET.Element, node: UnfoldFlow) -> None:
    elem = ET.SubElement(parent, "unfold_flow")
    elem.set("id", node.id)
    if node.name:
        elem.set("name", node.name)

    pm_elem = ET.SubElement(elem, "preserve_metadata")
    pm_elem.text = str(node.preserve_metadata).lower()

    sel_elem = ET.SubElement(elem, "skip_empty_lists")
    sel_elem.text = str(node.skip_empty_lists).lower()

    wnli_elem = ET.SubElement(elem, "wrap_non_list_inputs")
    wnli_elem.text = str(node.wrap_non_list_inputs).lower()
