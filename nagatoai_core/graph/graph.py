# Standard Library
import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

# Third Party
from pydantic import BaseModel, Field

# Nagato AI
from nagatoai_core.graph.abstract_flow import AbstractFlow
from nagatoai_core.graph.abstract_node import AbstractNode
from nagatoai_core.graph.types import NodeResult


class Graph(BaseModel):
    """
    Graph represents a directed acyclic graph (DAG) of nodes that are executed based on their dependencies.

    The Graph class manages the execution flow between different AbstractNode implementations,
    maintaining the edges between nodes and ensuring proper execution order.
    """

    nodes_set: Set[AbstractNode] = Field(default_factory=set)
    adjacency_list: Dict[str, List[str]] = Field(default_factory=lambda: defaultdict(list))
    node_map: Dict[str, AbstractNode] = Field(default_factory=dict)
    current_node: Optional[AbstractNode] = None
    result_map: Dict[str, List[NodeResult]] = Field(default_factory=dict)
    compiled: bool = False
    execution_order: List[str] = Field(default_factory=list)

    def add_edge(self, source: AbstractNode, target: AbstractNode) -> None:
        """
        Add a directed edge from source node to target node.

        Args:
            source: The source node
            target: The target node
        """
        # Add nodes to the set and node map
        self.nodes_set.add(source)
        self.nodes_set.add(target)
        self.node_map[source.id] = source
        self.node_map[target.id] = target

        # Add the edge to the adjacency list
        self.adjacency_list[source.id].append(target.id)

        # Mark as not compiled whenever the graph structure changes
        self.compiled = False

    def compile(self) -> None:
        """
        Verify that the graph is a valid DAG with no cycles and compute the execution order.

        Raises:
            ValueError: If a cycle is detected in the graph
            ValueError: If isolated nodes are detected in the graph
        """
        if not self.nodes_set:
            self.compiled = True
            return

        # Reset execution order
        self.execution_order = []

        # Check for cycles using depth-first search
        visited = set()
        temp_visited = set()

        def dfs(node_id: str) -> None:
            """Helper function for depth-first traversal with cycle detection"""
            if node_id in temp_visited:
                raise ValueError(f"Cycle detected in graph involving node {node_id}")

            if node_id in visited:
                return

            temp_visited.add(node_id)

            for neighbor_id in self.adjacency_list[node_id]:
                dfs(neighbor_id)

            temp_visited.remove(node_id)
            visited.add(node_id)
            self.execution_order.insert(0, node_id)  # Prepend for correct order

        # Find nodes with no connections (truly isolated)
        # A truly isolated node has no incoming or outgoing edges
        outgoing_edges = {node_id: len(neighbors) > 0 for node_id, neighbors in self.adjacency_list.items()}
        incoming_edges = defaultdict(int)
        for node_id, neighbors in self.adjacency_list.items():
            for neighbor_id in neighbors:
                incoming_edges[neighbor_id] += 1

        # Check for completely isolated nodes (no incoming or outgoing edges)
        isolated_nodes = []
        for node in self.nodes_set:
            has_outgoing = outgoing_edges.get(node.id, False)
            has_incoming = incoming_edges.get(node.id, 0) > 0
            if not has_outgoing and not has_incoming:
                isolated_nodes.append(node.id)

        if isolated_nodes:
            raise ValueError(f"Isolated nodes detected in graph: {', '.join(isolated_nodes)}")

        # Find all root nodes (nodes with no incoming edges) for the execution order
        root_nodes = [node.id for node in self.nodes_set if node.id not in incoming_edges]

        # Run DFS from each node to detect cycles - start with root nodes first
        if root_nodes:
            for node_id in root_nodes:
                if node_id not in visited:
                    dfs(node_id)
        else:
            # If no root nodes, the graph might have a cycle
            # Start DFS from any node to find the cycle
            start_node = next(iter(self.nodes_set)).id
            dfs(start_node)

        # Verify all nodes were visited
        if len(visited) < len(self.nodes_set):
            unvisited = [node.id for node in self.nodes_set if node.id not in visited]
            # Check if any unvisited nodes form cycles
            for node_id in unvisited:
                if node_id not in visited:
                    dfs(node_id)

        # If we got here, no cycles were detected
        self.compiled = True

    def run(self, inputs: List[NodeResult]) -> List[NodeResult]:
        """
        Execute the graph by traversing nodes in topological order.

        Args:
            inputs: Initial input NodeResult objects

        Returns:
            List[NodeResult]: Results from terminal nodes in the graph
        """
        logging.info(f"**** Running graph with inputs {inputs}")
        if not self.nodes_set:
            return inputs

        if not self.compiled:
            self.compile()

        # Initialize result tracking
        node_results: Dict[str, List[NodeResult]] = {}

        # Initialize input nodes with provided inputs
        initial_node_ids = set()
        for result in inputs:
            initial_node_ids.add(result.node_id)

        # Start with the input results
        for node_id in initial_node_ids:
            node_results[node_id] = [r for r in inputs if r.node_id == node_id]

        # Identify nodes with no predecessors (root nodes)
        predecessors = set()
        for _, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                predecessors.add(neighbor)

        root_nodes = [node.id for node in self.nodes_set if node.id not in predecessors]

        # Process each node in topological order
        for node_id in self.execution_order:
            # Skip nodes that already have results (input nodes)
            if node_id in node_results:
                continue

            node = self.node_map[node_id]
            self.current_node = node

            # Gather inputs from all predecessor nodes
            current_inputs = []

            # If this is a root node with no predecessors, use the original inputs
            if node_id in root_nodes:
                current_inputs = inputs
            else:
                # Otherwise gather inputs from predecessors as usual
                for predecessor_id, neighbors in self.adjacency_list.items():
                    if node_id in neighbors and predecessor_id in node_results:
                        current_inputs.extend(node_results[predecessor_id])

            # Execute the node
            logging.info(f"**** Executing node {node_id} with inputs {current_inputs}")
            results = node.execute(current_inputs)

            # Store results
            node_results[node_id] = results

            # Update result map for reference
            for result in results:
                if result.node_id not in self.result_map:
                    self.result_map[result.node_id] = []
                self.result_map[result.node_id].append(result)

        # Return results from terminal nodes (nodes with no outgoing edges)
        terminal_nodes = [node_id for node_id in self.node_map if not self.adjacency_list[node_id]]

        # If there are no terminal nodes, return results from all nodes
        if not terminal_nodes:
            all_results = []
            for results in node_results.values():
                all_results.extend(results)
            return all_results

        # Return results from terminal nodes
        terminal_results = []
        for node_id in terminal_nodes:
            if node_id in node_results:
                terminal_results.extend(node_results[node_id])

        return terminal_results
