LLM_JUDGE_PROMPT = """You are a goal-verification judge. Your task is to determine whether a goal has been achieved based on the output produced by a graph execution.

<goal>
{goal_description}
</goal>

<graph_output>
{graph_output}
</graph_output>

Analyze the graph output above and determine whether it satisfies the stated goal.
Consider completeness, correctness, and relevance of the output with respect to the goal.

Respond with the following XML format:
<verdict>
    <achieved>true/false</achieved>
    <reason>Your detailed explanation of why the goal was or was not achieved.</reason>
</verdict>

Ensure to output correctly formed XML tags (e.g. opening <verdict> and closing </verdict> tags).
"""
