[![X (formerly Twitter)](https://img.shields.io/twitter/follow/Ed_Forson?style=social)](https://twitter.com/Ed_Forson)
[![GitHub](https://img.shields.io/github/followers/kenshiro-o?label=Follow&style=social)](https://github.com/kenshiro-o)

# Nagato-AI

![Official Nagato AI Poster](docs/assets/Official_Nagato_AI_Poster.webp)


Nagato-AI is an intuitive AI Agent library that works across multiple LLMs.

Currently it supports OpenAI's GPT, Anthpropic's Claude, Google's Gemini, Groq (e.g. Llama 3) , And DeepSeek LLMs. You can create agents from any of the aforementioned family of models and combine them together to build the most effective AI Agent system you desire.

The name _Nagato_ is inspired from the popular anime Naruto. In Naruto, Nagato is a very powerful ninja who possesses special eyes (Rinnegan) that gives him immense powers.
Nagato's powers enable him to control multiple bodies endowed with different abilities. Nagato is also able to see through the eyes of all the bodies which he controls, thereby minimising blindspots that opponents may want to exploit.

Therefore, you can think of Nagato as the linchpin that summons and coordinates AI Agents which have a specific _mission_ to complete.

Note that from now on I will use the terms _Nagato_ and *Nagato-AI* interchangibly to refer to this library.

# How to use Nagato-AI

## Installation

### Working directly with source reposirory

If you're working on the source repository (either via a fork or the original repository), you must ensure that you have [poetry](https://python-poetry.org/) packaging/dependency management installed on your machine.
Once poetry is installed, then simply run the following command in your termninal (from the root folder of nagato code base) to install all required dependencies:

```
poetry install
```

### Installing via pip

Simply run the command:

```
pip install nagatoai_core
```

That's it! Nagato AI available to use in your code.

### Configuring environment variables

By default, Nagato will look for environment variables to create the AI Agents and tools.
First, make sure to create a `.env` file. Then add those variables to the `.env` file you just created.

You only need to add some of the below environment variables for the model and the tools you plan to use. The current list of environment variables is the following:
```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GROQ_API_KEY=
GOOGLE_API_KEY=
DEEPSEEK_API_KEY=

READWISE_API_KEY=

SERPER_API_KEY=

ELEVENLABS_API_KEY=
```

For instance if you only plan to use GPT-based agents and Readwise tools you should only set the `OPENAI_API_KEY` and `READWISE_API_KEY` environment variables.

### Running Python code

Assuming your program's entrypoint is defined in a file called `main.py`, you can run it by typing the following command:

```
poetry run python main.py
```


## LLMs supported

Nagato currently supports the following LLMs
* Claude 3 (Anthropic)
* GPT-3 to GPT-4 (OpenAI)
* Groq (which gives you access to Llama 3.1)
* Google Gemini

## LLMOps (Instrumentation)

Currently Nagato AI uses [Langfuse](https://langfuse.com/) for tracing LLM calls.
Set the environment variables below to be able to send traces:

```
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
```

You can see how Langfuse is being used in the `SingleAgentTaskRunner` class.


## Examples of AI Agent configuration

Nagato is built with flexibility at its core, so you could program it using your paradigm of choice. However these are some of the ways I've seen people use Nagato so far.

By default Nagato expects all LLM API keys to be set as environment variables. Nagato may load the keys from the following variables:

```
OPENAI_API_KEY=<api-key>
ANTHROPIC_API_KEY=<api-key>
READWISE_API_KEY=<api-key>
```

### Coordinator, worker, and critic agents

In this configuration we have the following:

* 🎯 Coordinator: breaks down a problem statement (from stdin) into an objective and suggests tasks
* 📚 Researcher: works on a task by performing research
* ✅ Critic: evaluates whether the task was completed


#### Example using models from different providers


```python
# Using Claude-3 Opus as the coordinator agent
coordinator_agent: Agent = create_agent(
    anthropic_api_key,
    "claude-3-5-sonnet-20241022",
    "Coordinator",
    COORDINATOR_SYSTEM_PROMPT,
    "Coordinator Agent",
)

# Using GPT-4 turbo as the researcher agent
researcher_agent = create_agent(
    anthropic_api_key,
    "gpt-4-turbo-2024-04-09",
    "Researcher",
    RESEARCHER_SYSTEM_PROMPT,
    "Researcher Agent",
)

# Use Google Genini 1.5 Flash as the critic ahemt
critic_agent = create_agent(
    google_api_key,
    "gemini-1.5-flash",
    "Critic",
    CRITIC_SYSTEM_PROMPT,
    "Critic Agent",
)
...
```


The full blow example is available [here](docs/examples/coordinator_researcher_critic.py)

### Worker and critic agents

In this configuration we directly submit as input an objective and a set of tasks needed to complete the objective.
Therefore we can skip the coordinator agent and have the worker agent(s) work on the tasks, while the critic agent evaluates whether the task carried out meets the requirements originally specified.


```python
task_list: List[Task] = [
    Task(
        goal="Fetch last 100 user tweets",
        description="Fetch the tweets from the user using the Twitter API. Limit the number of tweets fetched to 100 only."),
    Task(
        goal="Perform sentiment analysis on the tweets",
        description="Feed the tweets to the AI Agent to analyze sentiment per overall sentiment acoss tweets. Range of values for sentiment can be: Positive, Negative, or Neutral"
    )]

coordinator_agent: Agent = create_agent(
    anthropic_api_key,
    "claude-3-5-sonnet-20241022",
    "Coordinator",
    COORDINATOR_SYSTEM_PROMPT,
    "Coordinator Agent",
)

critic_agent = create_agent(
    anthropic_api_key,
    "claude-3-haiku-20240307",
    "Critic",
    CRITIC_SYSTEM_PROMPT,
    "Critic Agent",
)

for task in task_list:
    # Insert the task into the prompt
    worker_prompt = ...

    worker_exchange = researcher_agent.chat(worker_prompt, task, 0.7, 2000)

    # insert the response from the agent into prompt for the critic
    critic_prompt = ...

    critic_exchange = critic_agent(critic_prompt, task, 0.7, 2000)

    # Evaluate whether the task was completed based on the answer from the critic agent
    ...
```

## Tool calling

Check the full example [here](docs/examples/coordinator_researcher_critic.py) to see how tool calling works.
We now support tool calling for  GPT, Claude 3, and Llama 3 (via Groq) models.


### Tool creation

Creating a tool is straightforward. You must create have these two elements in place for a tool to be usable:

1. A config class that contains the parameters that your tool will be called with
2. A tool class that inherits from `AbstractTool`, and contains the main logic for your tool.

For instance the below shows how we've created a tool to get the user to confirm yes/no in the terminal

```python
from typing import Any, Type

from pydantic import BaseModel, Field
from rich.prompt import Confirm

from nagatoai_core.tool.abstract_tool import AbstractTool


class HumanConfirmInputConfig(BaseModel):
    """
    HumanConfirmInputConfig represents the configuration for the HumanConfirmInputTool.
    """

    message: str = Field(
        ...,
        description="The message to display to the user to confirm whether to proceed or not",
    )


class HumanConfirmInputTool(AbstractTool):
    """
    HumanConfirmInputTool represents a tool that prompts the user to confirm whether to proceed or not.
    """

    name: str = "human_confirm_input"
    description: str = (
        """Prompts the user to confirm whether to proceed or not. Returns a boolean value indicating the user's choice."""
    )
    args_schema: Type[BaseModel] = HumanConfirmInputConfig

    def _run(self, config: HumanConfirmInputConfig) -> Any:
        """
        Prompts the user to confirm whether to proceed or not.
        :param message: The message to display to the user to confirm whether to proceed or not.
        :return: A boolean value indicating the user's choice.
        """
        confirm = Confirm.ask("[bold yellow]" + config.message + "[/bold yellow]")

        return confirm
```

## Graph-based Agent Systems

Nagato supports creating complex agent systems using a directed acyclic graph (DAG) approach. This allows you to create powerful workflows by connecting different tool nodes and agent nodes.

### Graph Core Concepts

- **Graph**: The main structure that connects all nodes and manages execution flow
- **Nodes**: Processing units that perform specific tasks:
  - **Tool Nodes**: Execute specific tools (e.g., YouTube downloader, transcription)
  - **Agent Nodes**: LLM-based agents that process information
- **Flows**: Control structures for organizing execution. Some examples:
  - **SequentialFlow**: Executes nodes in sequence
  - **TransformerFlow**: Allows manipulation of data between flows
  - **ConditionalFlow**: Allows branching to a specific flow depending on predictate being true or false
- **Edges**: Connections between nodes that define the execution path

### Example: YouTube Video Processing Graph

Below is an example of creating a graph that:
1. Downloads a YouTube video
2. Transcribes it using Groq's Whisper API
3. Gets human input for focus areas
4. Generates a summary using a Gemini agent

```python
# Initialize tools
youtube_download_tool = YouTubeVideoDownloadTool()
groq_whisper_tool = GroqWhisperTool()
human_input_tool = HumanInputTool()

# Create tool providers
youtube_download_tool_provider = OpenAIToolProvider(
    name=youtube_download_tool.name,
    description=youtube_download_tool.description,
    args_schema=youtube_download_tool.args_schema,
    tool=youtube_download_tool,
)
# ... similar setup for other tools

# Create agents
conversion_agent = create_agent(
    api_key=google_api_key,
    model="gemini-1.5-flash",
    role="Parameter Converter",
    role_description="You convert parameters between different formats.",
    nickname="Converter",
)

summary_agent = create_agent(
    api_key=google_api_key,
    model="gemini-2.0-flash",
    role="Video Summarizer",
    role_description="You summarize video transcripts concisely.",
    nickname="Summarizer",
)

# Create nodes
youtube_download_node = ToolNodeWithParamsConversion(
    id="youtube_download_node",
    tool_provider=youtube_download_tool_provider,
    agent=conversion_agent,
    retries=2,
)

# ... create other nodes

# Create flows to organize nodes
sequential_yt_dl_transcribe_flow = SequentialFlow(
    id="sequential_yt_dl_transcribe_flow",
    nodes=[youtube_download_node, groq_whisper_node],
)

# ... create other flows

# Create and configure the graph
graph = Graph()

# Add edges to create the workflow
graph.add_edge(sequential_yt_dl_transcribe_flow, transformer_flow)
graph.add_edge(transformer_flow, sequential_summary_flow)

# Compile and validate the graph
graph.compile()

# Run the graph with initial input
initial_input = [NodeResult(node_id="input", result=input_data, step=0)]
results = graph.run(initial_input)
```

In this example, the graph processes a YouTube video through multiple steps. Each node in the graph performs a specific function, and the edges determine how data flows between nodes. The power of this approach is that you can create complex workflows by combining different tools and agents in flexible ways.

Check out the full example in [docs/examples/youtube_transcription_summary_graph.py](docs/examples/youtube_transcription_summary_graph.py).


# What's next

Nagato is still in its very early development phase. This means that I am likely to introduce breaking changes over the next iterations of the library.

Moreover, there is a lot of functionality currently missing from Nagato. I will remedy this over time. There is no official roadmap per se but I plan to add the following capabilities to Nagato:

* ✅ implement function calling (complement to adding tools)
* ✅ introduce basic tools (e.g. surfing the web)
* ✅ implement agent based on Llama 3 model (via Groq)
* ✅ implement agent based on Google Gemini models (without function calling)
* ✅ cache results from function calling
* ✅ implement v1 of self-reflection and re-planning for agents
* ✅ Implement audio/text-to-speech tools
* ✅ implement function calling for Google Gemini agent
* ✅ LLMOps instrumentation (via Langfuse)
* ✅ Add DeepSeek Agent
* 🎯 Build DAG/Graph based agentic capability
* 🎯 Extract Chain of Thought Reasoning
* 🎯 implement short/long-term memory for agents (with RAG and memory synthesis)
* 🎯 implement additional modalities (e.g. image, sound, etc.)
* 🎯 Support for local LLMs (e.g. via Ollama)

# How can you support

I'd be grateful if you could do some of the following to support this project:

* star this repository on Github
* follow me on [X/Twitter](https://twitter.com/Ed_Forson)
* raise Github issues if you've come across any bug using Nagato or would like a feature to be added to Nagato
