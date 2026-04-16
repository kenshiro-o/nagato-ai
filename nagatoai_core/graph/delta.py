from __future__ import annotations

# Standard Library
from collections import defaultdict
from copy import deepcopy
from typing import Annotated, Dict, List, Literal, Union

# Third Party
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.graph import Graph


class DeltaOp(BaseModel):
    """Base class for graph delta operations."""

    class Config:
        arbitrary_types_allowed = True


class AddNode(DeltaOp):
    """Add a new node to the graph."""

    kind: Literal["add_node"] = "add_node"
    node: AbstractNode


class RemoveNode(DeltaOp):
    """Remove a node and all its edges from the graph."""

    kind: Literal["remove_node"] = "remove_node"
    node_id: str


class ReplaceNode(DeltaOp):
    """Replace an existing node, preserving its edges."""

    kind: Literal["replace_node"] = "replace_node"
    node_id: str
    new_node: AbstractNode


class AddEdge(DeltaOp):
    """Add a directed edge between two existing nodes."""

    kind: Literal["add_edge"] = "add_edge"
    source_id: str
    target_id: str


class RemoveEdge(DeltaOp):
    """Remove a directed edge between two nodes."""

    kind: Literal["remove_edge"] = "remove_edge"
    source_id: str
    target_id: str


DeltaOpUnion = Annotated[
    Union[AddNode, RemoveNode, ReplaceNode, AddEdge, RemoveEdge],
    Field(discriminator="kind"),
]


class GraphDelta(BaseModel):
    """
    Represents a set of ordered operations to transform a graph.

    :param ops: The ordered list of delta operations to apply.
    """

    ops: List[DeltaOpUnion] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


def apply_delta(graph: Graph, delta: GraphDelta) -> Graph:
    """
    Apply a GraphDelta to a Graph, returning a new Graph without mutating the original.

    :param graph: The source graph to apply the delta to.
    :param delta: The delta operations to apply.
    :returns: A new compiled Graph with the delta applied.
    :raises ValueError: If a delta operation is invalid or would create a cycle.
    """
    working_nodes: Dict[str, AbstractNode] = {nid: node for nid, node in graph.node_map.items()}
    working_adj: Dict[str, List[str]] = {nid: list(targets) for nid, targets in graph.adjacency_list.items()}

    for op in delta.ops:
        if isinstance(op, AddNode):
            if op.node.id in working_nodes:
                raise ValueError(f"Node '{op.node.id}' already exists in graph")
            working_nodes[op.node.id] = op.node

        elif isinstance(op, RemoveNode):
            if op.node_id not in working_nodes:
                raise ValueError(f"Node '{op.node_id}' not found in graph")
            del working_nodes[op.node_id]
            working_adj.pop(op.node_id, None)
            for source_id in list(working_adj.keys()):
                working_adj[source_id] = [t for t in working_adj[source_id] if t != op.node_id]

        elif isinstance(op, ReplaceNode):
            if op.node_id not in working_nodes:
                raise ValueError(f"Node '{op.node_id}' not found in graph")
            old_id = op.node_id
            new_id = op.new_node.id
            del working_nodes[old_id]
            working_nodes[new_id] = op.new_node
            if old_id != new_id:
                targets = working_adj.pop(old_id, [])
                working_adj[new_id] = targets
                for source_id in list(working_adj.keys()):
                    working_adj[source_id] = [new_id if t == old_id else t for t in working_adj[source_id]]
            else:
                pass

        elif isinstance(op, AddEdge):
            if op.source_id not in working_nodes:
                raise ValueError(f"Source node '{op.source_id}' not found in graph")
            if op.target_id not in working_nodes:
                raise ValueError(f"Target node '{op.target_id}' not found in graph")
            if op.source_id not in working_adj:
                working_adj[op.source_id] = []
            working_adj[op.source_id].append(op.target_id)

        elif isinstance(op, RemoveEdge):
            if op.source_id in working_adj and op.target_id in working_adj[op.source_id]:
                working_adj[op.source_id].remove(op.target_id)
            else:
                raise ValueError(f"Edge from '{op.source_id}' to '{op.target_id}' not found in graph")

    new_graph = Graph()

    for source_id, target_ids in working_adj.items():
        for target_id in target_ids:
            source_node = working_nodes[source_id]
            target_node = working_nodes[target_id]
            new_graph.add_edge(source_node, target_node)

    for nid, node in working_nodes.items():
        if nid not in new_graph.node_map:
            new_graph.node_map[nid] = node
            new_graph.nodes_set.add(node)

    new_graph.compile()
    return new_graph
