# T2 + T3 — GraphDelta, apply_delta, XML schema, graph serializer

Combined because T2 (the data model and in-memory application) and
T3 (XML ingest + graph serialization) are interdependent and should
land together.

## Goal of the task

Give the planner a way to express "change the current plan in these
specific ways" without re-emitting the whole graph, and give the
runner a way to apply those changes safely while preserving all the
properties that make `Graph` debuggable (acyclicity, topological
order, compile-time validation).

## Files to create

- `nagatoai_core/graph/delta.py`

## Files to modify

- `nagatoai_core/graph/plan/xml_parser.py` — add `parse_delta()`
  and `serialize(graph)`.

## Design

### `GraphDelta` (in `delta.py`)

Ordered list of typed operations. The order matters — a common case
is "remove an old node and add its replacement", and inverting the
order would fail the compile step.

```
class DeltaOp(BaseModel): ...  # discriminated union base

class AddNode(DeltaOp):
    kind: Literal["add_node"] = "add_node"
    node: AbstractNode         # reuses existing node types

class RemoveNode(DeltaOp):
    kind: Literal["remove_node"] = "remove_node"
    node_id: str

class ReplaceNode(DeltaOp):
    kind: Literal["replace_node"] = "replace_node"
    node_id: str
    new_node: AbstractNode

class AddEdge(DeltaOp):
    kind: Literal["add_edge"] = "add_edge"
    source_id: str
    target_id: str

class RemoveEdge(DeltaOp):
    kind: Literal["remove_edge"] = "remove_edge"
    source_id: str
    target_id: str

class GraphDelta(BaseModel):
    ops: List[DeltaOp]
```

Use a Pydantic discriminated union on `kind`.

### `apply_delta(graph, delta) -> Graph` (in `delta.py`)

Pure function. Returns a **new** `Graph` — does not mutate the input.

Rationale for returning a new graph rather than mutating in place:

- Preserves the old graph for trace / audit (the `GoalRunner` wants
  to show both "before" and "after" per iteration).
- Avoids aliasing bugs if any caller keeps a reference to the old
  `Graph`.
- `Graph` is small and copying it is cheap compared to the cost of
  an LLM call.

Implementation sketch:

```
def apply_delta(graph: Graph, delta: GraphDelta) -> Graph:
    new_graph = Graph()  # fresh, empty
    # copy over existing nodes and edges
    for source_id, targets in graph.adjacency_list.items():
        source = graph.node_map[source_id]
        for target_id in targets:
            target = graph.node_map[target_id]
            new_graph.add_edge(source, target)
    # also copy isolated nodes (nodes with neither in nor out edges)
    # — but note Graph.compile rejects these, so this path is only
    # relevant during intermediate construction, not steady state.

    for op in delta.ops:
        match op:
            case AddNode(node=n):
                new_graph.node_map[n.id] = n
                new_graph.nodes_set.add(n)
            case RemoveNode(node_id=nid):
                new_graph.nodes_set.discard(new_graph.node_map.get(nid))
                new_graph.node_map.pop(nid, None)
                new_graph.adjacency_list.pop(nid, None)
                for src in list(new_graph.adjacency_list.keys()):
                    new_graph.adjacency_list[src] = [
                        t for t in new_graph.adjacency_list[src] if t != nid
                    ]
            case ReplaceNode(node_id=nid, new_node=nn):
                # in-place swap; edges are preserved
                new_graph.nodes_set.discard(new_graph.node_map.get(nid))
                new_graph.node_map[nn.id] = nn
                new_graph.nodes_set.add(nn)
                if nn.id != nid:
                    # rewire adjacency list if id changed
                    new_graph.adjacency_list[nn.id] = new_graph.adjacency_list.pop(nid, [])
                    for src in list(new_graph.adjacency_list.keys()):
                        new_graph.adjacency_list[src] = [
                            nn.id if t == nid else t
                            for t in new_graph.adjacency_list[src]
                        ]
            case AddEdge(source_id=s, target_id=t):
                new_graph.add_edge(new_graph.node_map[s], new_graph.node_map[t])
            case RemoveEdge(source_id=s, target_id=t):
                new_graph.adjacency_list[s] = [
                    nid for nid in new_graph.adjacency_list[s] if nid != t
                ]

    new_graph.compiled = False
    new_graph.compile()  # raises on cycle / isolated node
    return new_graph
```

The caller (the `GoalRunner`) is responsible for catching the
`ValueError` that `compile` raises, recording the failure, and
giving the planner another attempt.

### XML `<delta>` schema (in `xml_parser.py`)

```xml
<delta>
  <add_node>
    <!-- existing node/flow XML — reuses parse_nodes -->
  </add_node>
  <remove_node id="old_node_id"/>
  <replace_node id="node_id">
    <!-- existing node/flow XML -->
  </replace_node>
  <add_edge source="a" target="b"/>
  <remove_edge source="a" target="c"/>
</delta>
```

Implementation:

```
class XMLPlanParser:
    def parse_delta(self, xml: str, current_graph: Graph) -> GraphDelta:
        # parse root element, iterate children in order,
        # dispatch each tag to the existing parse_nodes / parse_flow
        # logic for the body of add_node / replace_node,
        # build a list[DeltaOp], return GraphDelta(ops=...).
```

No new parsing primitives — this is a thin wrapper over the
existing node/flow parsing. The `current_graph` parameter is passed
so the parser can validate that `remove_node` / `remove_edge` /
`replace_node` refer to existing IDs (fail fast at parse time so
the planner sees the error immediately).

### Graph serializer: `XMLPlanParser.serialize(graph) -> str`

The planner needs to see the current graph as XML so it can reason
about what to change. Today there is no graph → XML round-tripper;
we need to add one.

- Must cover every node and flow type the parser can read. Keeping
  `parse_nodes` and `serialize` in lockstep is an invariant worth
  calling out in a comment on both methods.
- Output should be human-readable (indented) for debugging.
- Serialize the graph topologically so the XML reads top-to-bottom
  in execution order.

## Pre-flight audit

Before implementing, confirm the following by reading the code:

- **Does anything outside `Graph.run` keep live references to
  `Graph.nodes_set` / `adjacency_list`?** If yes, the
  "return-a-new-graph" approach may leave dangling references.
  `grep` the `nagatoai_core` tree for `.nodes_set` and
  `.adjacency_list` accesses from outside the `Graph` class. The
  one-shot execution model makes this unlikely, but verify.
- **Is `AbstractNode` hashable?** `Graph.nodes_set` is a set, so
  yes — but the Pydantic BaseModel default may not give identity
  semantics. Check how nodes currently get added/compared. If
  nodes are hashed by id, deltas are safer to implement; if by
  deep equality, `ReplaceNode` has subtle corners.

## Done when

- `GraphDelta`, `DeltaOp`, `apply_delta` all importable.
- `XMLPlanParser.parse_delta` and `XMLPlanParser.serialize` both
  implemented and round-trip on a basic example.
- `apply_delta` returns a fresh `Graph` for every delta, leaving
  the input untouched (covered by a unit test).
- Cycle-inducing delta is rejected with a clear `ValueError`.
- `make fmt && make lint` pass.
- Unit tests in `tests/nagatoai_core/graph/test_delta.py` cover
  add / remove / replace / add-edge / remove-edge / compile
  failure. See `task-verification.md`.
