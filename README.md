[![X (formerly Twitter)](https://img.shields.io/twitter/follow/Ed_Forson?style=social)](https://twitter.com/Ed_Forson)
[![GitHub](https://img.shields.io/github/followers/kenshiro-o?label=Follow&style=social)](https://github.com/kenshiro-o)

# Nagato-AI

![Official Nagato AI Poster](docs/assets/Official_Nagato_AI_Poster.webp)


Nagato-AI is an intuitive AI Agent library that works across multiple LLMs.

Currently it supports OpenAI's GPT, Anthpropic's Claude, Google's Gemini, and Groq (e.g. Llama 3) LLMs. You can create agents from any of the aforementioned family of models and combine them together to build the most effective AI Agent system you desire.

The name _Nagato_ is inspired from the popular anime Naruto. In Naruto, Nagato is a very powerful ninja who possesses special eyes (Rinnegan) that gives him immense powers.
Nagato's powers enable him to control multiple bodies endowed with different abilities. Nagato is also able to see through the eyes of all the bodies which he controls, thereby minimising blindspots that opponents may want to exploit.

Therefore, you can think of Nagato as the linchpin that summons and coordinates AI Agents which have a specific _mission_ to complete.

Note that from now on I will use the terms _Nagato_ and *Nagato-AI* interchangibly to refer to this library.

# How to use Nagato-AI

## Installing Poetry dependency management

First, ensure that you have [poetry](https://python-poetry.org/) packaging/dependency management installed on your machine.
Once poetry is installed, then simply run the following command in your termninal (from the root folder of nagato code base) to install all required dependencies:

```
poetry install
```

### Configuring environment variables

By default, Nagato will look for environment variables to create the AI Agents and tools.
First, make sure to create a `.env` file. Then add those variables to the `.env` file you just created.

You only need to add some of the below environment variables for the model and the tools you plan to use. The current list of environment variables is the following:
```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GROQ_API_KEY=
GOOGLE_API_KEY=

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
* Groq (which gives you access to Llama 3)
* Google Gemini

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
    "claude-3-opus-20240229",
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
    "claude-3-sonnet-20240229",
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
* 🎯 implement function calling for Google Gemini agent
* 🎯 implement short/long-term memory for agents (with RAG and memory synthesis)
* 🎯 implement additional modalities (e.g. image, sound, etc.)
* 🎯 Support for local LLMs (e.g. via Ollama)
* 🎯 LLMOps instrumentation

# How can you support

I'd be grateful if you could do some of the following to support this project:

* star this repository on Github
* follow me on [X/Twitter](https://twitter.com/Ed_Forson)
* raise Github issues if you've come across any bug using Nagato or would like a feature to be added to Nagato