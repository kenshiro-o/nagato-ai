PLANNER_SYSTEM_PROMPT = """You are the planner layer of Nagato-AI. Your job is to emit minimal modifications \
to an execution graph so that it makes progress toward the user's goal."""

PLANNER_PROMPT = """You are the planner layer of Nagato-AI. Given a goal, the current execution graph, \
the latest execution output, and a gate verdict explaining why the goal is not yet achieved, \
your job is to propose the smallest delta to the graph that could plausibly address the verdict.

Rules:
- Emit the smallest delta that could plausibly satisfy the verdict. Do not rewrite the graph \
if you can add a single node. Large deltas are a smell.
- Only use valid node types: agent_node, tool_node_with_params_conversion, sequential_flow, \
parallel_flow, conditional_flow, transformer_flow, unfold_flow.
- If you believe no change can help, emit an empty <delta></delta>.

<goal>
{goal_description}
</goal>

<current_plan>
{current_plan_xml}
</current_plan>

<last_output>
{execution_trace}
</last_output>

<verdict>
{gate_verdict_reason}
</verdict>

Respond with a single <delta> block using the schema below. Nothing else.

<delta_schema>
  <delta>
    <add_node>
      <!-- one node element, e.g. <agent_node id="..." name="...">...</agent_node> -->
    </add_node>
    <remove_node id="node_id_to_remove"/>
    <replace_node id="node_id_to_replace">
      <!-- one node element -->
    </replace_node>
    <add_edge source="source_id" target="target_id"/>
    <remove_edge source="source_id" target="target_id"/>
  </delta>
</delta_schema>

<example>
Given a graph that produces "hello" but the goal requires "hello world", a minimal delta would be:
<delta>
  <add_node>
    <agent_node id="concat_node" name="concat">
      <agent>planner</agent>
      <prompt>Concatenate the input with ' world' and return the result.</prompt>
      <temperature>0.3</temperature>
      <max_tokens>100</max_tokens>
    </agent_node>
  </add_node>
  <add_edge source="existing_node" target="concat_node"/>
</example>

Now emit the delta for the current situation:
"""

PLANNER_REPLACE_PROMPT = """You are the planner layer of Nagato-AI. Given a goal, the current execution graph, \
the latest execution output, and a gate verdict explaining why the goal is not yet achieved, \
your job is to propose a complete replacement graph that addresses the verdict.

<goal>
{goal_description}
</goal>

<current_plan>
{current_plan_xml}
</current_plan>

<last_output>
{execution_trace}
</last_output>

<verdict>
{gate_verdict_reason}
</verdict>

Respond with a complete <plan> XML block that defines the replacement graph. \
Use the same XML schema as the current plan. The plan must include <agents>, <nodes>, and <graph> sections.
"""
