# Standard Library
from typing import List

# Third Party
import pytest

# Nagato AI
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.delta import AddEdge, AddNode, GraphDelta, RemoveEdge, RemoveNode, ReplaceNode, apply_delta
from nagatoai_core.graph.graph import Graph
from nagatoai_core.graph.types import NodeResult


class TestNode(AbstractNode):
    """A simple test node that increments input values by a specified amount."""

    increment_by: int = 1
    execution_count: int = 0

    def execute(self, inputs: List[NodeResult]) -> List[NodeResult]:
        self.execution_count += 1
        results = []
        for input_item in inputs:
            value = input_item.result
            if isinstance(value, (int, float)):
                result = value + self.increment_by
            else:
                result = value
            results.append(NodeResult(node_id=self.id, result=result, error=None))
        return results


def _build_simple_graph() -> Graph:
    """Build a simple A -> B -> C graph for testing."""
    graph = Graph()
    a = TestNode(id="a", name="A")
    b = TestNode(id="b", name="B")
    c = TestNode(id="c", name="C")
    graph.add_edge(a, b)
    graph.add_edge(b, c)
    graph.compile()
    return graph


def test_apply_delta_add_node():
    graph = _build_simple_graph()
    d = TestNode(id="d", name="D")
    delta = GraphDelta(ops=[AddNode(node=d), AddEdge(source_id="c", target_id="d")])

    new_graph = apply_delta(graph, delta)

    assert "d" in new_graph.node_map
    assert len(new_graph.node_map) == 4
    assert "d" in new_graph.adjacency_list["c"]


def test_apply_delta_remove_node():
    graph = _build_simple_graph()
    delta = GraphDelta(ops=[RemoveEdge(source_id="b", target_id="c"), RemoveNode(node_id="c")])

    new_graph = apply_delta(graph, delta)

    assert "c" not in new_graph.node_map
    assert len(new_graph.node_map) == 2
    for targets in new_graph.adjacency_list.values():
        assert "c" not in targets


def test_apply_delta_replace_node():
    graph = _build_simple_graph()
    b_new = TestNode(id="b", name="B-replaced", increment_by=5)
    delta = GraphDelta(ops=[ReplaceNode(node_id="b", new_node=b_new)])

    new_graph = apply_delta(graph, delta)

    assert "b" in new_graph.node_map
    assert new_graph.node_map["b"].name == "B-replaced"
    assert "b" in new_graph.adjacency_list["a"]
    assert "c" in new_graph.adjacency_list["b"]


def test_apply_delta_add_edge():
    graph = Graph()
    a = TestNode(id="a", name="A")
    b = TestNode(id="b", name="B")
    c = TestNode(id="c", name="C")
    graph.add_edge(a, b)
    graph.add_edge(a, c)
    graph.compile()

    delta = GraphDelta(ops=[AddEdge(source_id="b", target_id="c")])

    new_graph = apply_delta(graph, delta)

    assert "c" in new_graph.adjacency_list["b"]


def test_apply_delta_remove_edge():
    graph = Graph()
    a = TestNode(id="a", name="A")
    b = TestNode(id="b", name="B")
    c = TestNode(id="c", name="C")
    graph.add_edge(a, b)
    graph.add_edge(a, c)
    graph.add_edge(b, c)
    graph.compile()

    delta = GraphDelta(ops=[RemoveEdge(source_id="b", target_id="c")])

    new_graph = apply_delta(graph, delta)

    assert "c" not in new_graph.adjacency_list.get("b", [])
    assert "b" in new_graph.adjacency_list["a"]
    assert "c" in new_graph.adjacency_list["a"]


def test_apply_delta_cycle_raises():
    graph = _build_simple_graph()
    delta = GraphDelta(ops=[AddEdge(source_id="c", target_id="a")])

    with pytest.raises(ValueError, match="Cycle detected"):
        apply_delta(graph, delta)


def test_apply_delta_original_unchanged():
    graph = _build_simple_graph()
    original_node_ids = set(graph.node_map.keys())
    original_adj = {k: list(v) for k, v in graph.adjacency_list.items()}

    d = TestNode(id="d", name="D")
    delta = GraphDelta(ops=[AddNode(node=d), AddEdge(source_id="c", target_id="d")])

    apply_delta(graph, delta)

    assert set(graph.node_map.keys()) == original_node_ids
    for k, v in original_adj.items():
        assert list(graph.adjacency_list[k]) == v


def test_apply_delta_empty_ops():
    graph = _build_simple_graph()
    delta = GraphDelta(ops=[])

    new_graph = apply_delta(graph, delta)

    assert set(new_graph.node_map.keys()) == set(graph.node_map.keys())
    for source_id, targets in graph.adjacency_list.items():
        assert list(new_graph.adjacency_list[source_id]) == list(targets)
