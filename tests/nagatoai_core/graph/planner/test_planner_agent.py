# Standard Library
import os
from unittest.mock import MagicMock, patch
import logging
from typing import List

# Third Party
import pytest
from bs4 import BeautifulSoup

# Nagato AI
from nagatoai_core.agent.factory import create_agent, get_agent_tool_provider
from nagatoai_core.graph.planner.planner_agent import PlannerAgent
from nagatoai_core.graph.planner.objective import Objective
from nagatoai_core.mission.task import Task
from nagatoai_core.graph.plan.validator import XMLPlanValidator
from nagatoai_core.tool.provider.abstract_tool_provider import AbstractToolProvider
from nagatoai_core.tool.provider.openai import OpenAIToolProvider
from nagatoai_core.tool.lib.readwise.book_finder import ReadwiseDocumentFinderTool
from nagatoai_core.tool.lib.readwise.highlights_lister import ReadwiseHighightsListerTool
from nagatoai_core.tool.lib.readwise.book_highlights_lister import ReadwiseBookHighlightsListerTool
from nagatoai_core.tool.lib.human.confirm import HumanConfirmInputTool
from nagatoai_core.tool.lib.human.input import HumanInputTool
from nagatoai_core.tool.lib.web.page_scraper import WebPageScraperTool
from nagatoai_core.tool.lib.web.serper_search import SerperSearchTool
from nagatoai_core.tool.lib.video.youtube.video_download import YouTubeVideoDownloadTool


@pytest.fixture
def api_key():
    """Get API key from environment or use a default test key."""
    # Check if OpenAI API key is set, otherwise skip the test
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ.get("OPENAI_API_KEY")
    else:
        pytest.skip("No OpenAI API key found, skipping test")


@pytest.fixture
def tools():
    """Create a list of tool providers."""
    # Return an empty list for simplicity
    tools_available = [
        YouTubeVideoDownloadTool(),
        WebPageScraperTool(),
        SerperSearchTool(),
        ReadwiseDocumentFinderTool(),
        ReadwiseHighightsListerTool(),
        ReadwiseBookHighlightsListerTool(),
        HumanConfirmInputTool(),
        HumanInputTool(),
    ]

    tool_providers = [
        OpenAIToolProvider(name=tool.name, description=tool.description, tool=tool) for tool in tools_available
    ]

    return tool_providers


@pytest.fixture
def planner_agent(api_key):
    """Create a planner agent with a real underlying agent."""
    model = "o3-mini"
    role = "planner"
    role_description = "You are a planning assistant that helps break down tasks. You take the user's request, and generate an objective and a list of milestones."
    nickname = "Planner"

    # Create a real agent
    agent = create_agent(api_key, model, role, role_description, nickname)

    # Create planner agent and add xml validator
    planner = PlannerAgent(agent=agent)

    return planner


def test_generate_objective_basic_prompt(planner_agent: PlannerAgent):
    """Test generating an objective from a basic prompt."""
    # Test
    prompt = "Download the video with id 'oFfVt3S51T4' from YouTube and write a 500-word summary about it"
    objective = planner_agent.generate_objective(None, prompt, [])

    # Assertions
    assert isinstance(objective, Objective)
    logging.info(f"*** Objective: {objective}")
    assert objective.objective  # Verify objective is not empty
    assert isinstance(objective.milestones, list)
    assert objective.user_request == prompt


def test_generate_plan_xml_youtube_video_download(planner_agent: PlannerAgent, tools: List[AbstractToolProvider]):
    """Test generating an XML plan from an objective."""
    # Test
    prompt = "Download the video with id 'oFfVt3S51T4' from YouTube and write a 500-word summary about it"
    objective = Objective(
        user_request=prompt,
        objective="Download the specified YouTube video and produce a concise 500-word summary capturing its content.",
        milestones=[
            "Locate and verify the YouTube video using its provided ID.",
            "Download the video from YouTube.",
            "Review the video content to understand its key elements.",
            "Draft a 500-word summary highlighting the main points.",
            "Review and edit the summary for clarity and accuracy.",
        ],
    )
    xml_plan = planner_agent.generate_plan_xml(prompt, objective, tools)
    logging.info(f"*** XML Plan: {xml_plan}")

    assert isinstance(xml_plan, str)
    assert len(xml_plan) > 0

    # Ensure that the plan contains a tool_node_with_params_conversion node
    plan_soup = BeautifulSoup(xml_plan, "xml").find("plan")
    assert plan_soup is not None

    # Ensure that the plan contains a tool_node_with_params_conversion node
    tool_node_with_params_conversion = plan_soup.find("tool_node_with_params_conversion", recursive=True)
    assert tool_node_with_params_conversion is not None

    # Ensure that the node has the tool youtube_video_download
    tool = tool_node_with_params_conversion.find("tool", recursive=True)
    assert tool is not None
    assert tool.get("name") == "youtube_video_download"


def test_generate_plan_xml_readwise_book_highlights_summeriser(
    planner_agent: PlannerAgent, tools: List[AbstractToolProvider]
):
    """Test generating an XML plan from an objective."""
    # Test
    prompt = (
        "Write me a 500-word summary of the article 'Building Effective Agents' that I have highlighted on Readwise"
    )
    objective = Objective(
        user_request=prompt,
        objective="Summarise the highlights in Readwise for the article Building Effective Agents into a 500-word summary",
        milestones=[
            "Locate and verify the Readwise article using its provided ID.",
            "Download the article from Readwise.",
            "Review the article content to understand its key elements.",
            "Draft a 500-word summary highlighting the main points.",
            "Review and edit the summary for clarity and accuracy.",
        ],
    )

    xml_plan = planner_agent.generate_plan_xml(prompt, objective, tools)
    logging.info(f"*** XML Plan: {xml_plan}")

    assert isinstance(xml_plan, str)
    assert len(xml_plan) > 0
