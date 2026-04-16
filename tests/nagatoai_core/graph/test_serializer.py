# Standard Library
import xml.etree.ElementTree as ET
from typing import List

# Nagato AI
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.serializer import serialize_graph
from nagatoai_core.graph.types import NodeResult


class TestNode(AbstractNode):
    """A simple test node for serialization tests."""

    increment_by: int = 1

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        results = []
        for input_item in inputs:
            value = input_item.result
            if isinstance(value, (int, float)):
                result = value + self.increment_by
            else:
                result = value
            results.append(NodeResult(node_id=self.id, result=result, error=None))
        return results


def test_serialize_simple_graph():
    graph = Graph()
    a = TestNode(id="a", name="A")
    b = TestNode(id="b", name="B")
    graph.add_edge(a, b)
    graph.compile()

    xml_str = serialize_graph(graph)

    root = ET.fromstring(xml_str)
    assert root.tag == "plan"

    nodes_elem = root.find("nodes")
    assert nodes_elem is not None
    node_elems = list(nodes_elem)
    assert len(node_elems) == 2

    for elem in node_elems:
        assert elem.tag == "node"
        assert elem.get("id") in ("a", "b")
        assert elem.get("type") == "TestNode"

    graph_elem = root.find("graph")
    assert graph_elem is not None
    edges_elem = graph_elem.find("edges")
    assert edges_elem is not None


def test_serialize_graph_edges():
    graph = Graph()
    a = TestNode(id="a", name="A")
    b = TestNode(id="b", name="B")
    c = TestNode(id="c", name="C")
    graph.add_edge(a, b)
    graph.add_edge(a, c)
    graph.add_edge(b, c)
    graph.compile()

    xml_str = serialize_graph(graph)

    root = ET.fromstring(xml_str)
    edges_elem = root.find("graph/edges")
    edge_elems = edges_elem.findall("edge")

    assert len(edge_elems) == 3

    edge_pairs = {(e.get("from"), e.get("to")) for e in edge_elems}
    assert ("a", "b") in edge_pairs
    assert ("a", "c") in edge_pairs
    assert ("b", "c") in edge_pairs
