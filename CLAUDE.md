# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nagato-AI is a Python library for building multi-LLM AI agent systems. It supports OpenAI, Anthropic, Google Gemini, Groq, and DeepSeek models through a unified agent interface. Agents can be orchestrated via graphs (DAGs), chains (sequential pipelines), or mission/task runners.

## Commands

```bash
# Install
uv sync

# Format (isort + Black, line-length 120)
make fmt

# Lint (pylint, threshold 6)
make lint

# Run all tests
make test

# Run tests with debug logging
make test-debug

# Run a single test file
uv run pytest tests/nagatoai_core/graph/test_graph.py

# Run a single test
uv run pytest tests/nagatoai_core/graph/test_graph.py::test_function_name
```

## Architecture

### Agents (`nagatoai_core/agent/`)
Abstract `Agent` base class with provider-specific implementations (OpenAI, Anthropic, Google, Groq, DeepSeek). Use `create_agent()` and `get_agent_type()` from `factory.py` to instantiate agents by model name. Each agent maintains conversation history and supports tool calling. `message.py` defines `Exchange`, `Message`, `ToolCall`, `ToolResult`.

### Tools (`nagatoai_core/tool/`)
`AbstractTool` base class with a `_run()` method. Tools are registered in `ToolRegistry` which provides fuzzy name matching via Levenshtein distance. Tool providers (`tool/provider/`) generate LLM-specific schemas for each tool. Built-in tools live in `tool/lib/` (audio, filesystem, human input, web scraping, video, time, Readwise).

### Graphs (`nagatoai_core/graph/`)
DAG-based workflow execution. `Graph` manages nodes and edges with cycle detection. Node types: `AgentNode`, `ToolNode`, `ToolNodeWithParamsConversion`. Flow types compose nodes: `SequentialFlow`, `ParallelFlow`, `ConditionalFlow`, `TransformerFlow`, `UnfoldFlow`. Workflows can also be defined declaratively in XML via `graph/plan/xml_parser.py`. See `graph/README.md` for detailed graph documentation.

### Chains (`nagatoai_core/chain/`)
Sequential processing pipelines. `Chain` executes a list of links (`AgentLink`, `ToolLink`, `AdapterLink`, `AccumulatorLink`) with retry logic. `AgentParamConverter` handles parameter transformation between chain steps.

### Runners (`nagatoai_core/runner/`)
`SingleAgentTaskRunner` executes a `Task` with a single agent, supporting tool calling with result caching (`memory/tool_run_cache.py`) and optional Langfuse tracing. `SingleCriticEvaluator` uses a critic agent to evaluate task outcomes.

### Missions & Tasks (`nagatoai_core/mission/`)
`Mission` groups related `Task` objects. Tasks track status (PENDING/IN_PROGRESS/COMPLETED/FAILED) and outcomes via `TaskResult` with evaluation ratings.

## Code Conventions

- All non-test classes must inherit from Pydantic `BaseModel`
- Use reStructuredText docstring style
- Use pytest for tests (functions, not classes), placed in `tests/nagatoai_core/` mirroring source structure
- Minimize comments — only add when necessary
- Python 3.12+, managed with uv

## Environment Variables

API keys are loaded from `.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`. Optional: `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` (for tracing), `SERPER_API_KEY`, `READWISE_API_KEY`, `ELEVENLABS_API_KEY`.
