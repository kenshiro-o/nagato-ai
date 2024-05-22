from typing import Type, Dict

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
import requests

from nagatoai_core.tool.abstract_tool import AbstractTool


class SerperSearchConfig(BaseSettings, BaseModel):
    api_key: str = Field(
        ...,
        description="The Serper API key to use for authentication.",
        alias="SERPER_API_KEY",
        exclude_from_schema=True,
    )

    search_query: str = Field(
        ...,
        description="The search query to use for the keyword search.",
    )


class SerperSearchTool(AbstractTool):
    """
    SerperSearchTool is a tool that performs keyword search on the Google search engine
    """

    name: str = "serper_search"
    description: str = (
        """Performs keyword search on the Google search engine. Returns a JSON object that contains the search results.
        """
    )
    args_schema: Type[BaseModel] = SerperSearchConfig

    def _run(self, config: SerperSearchConfig) -> Dict:
        """
        Performs keyword search on the Google search engine.
        :param config: The configuration for the SerperSearchTool.
        :return: The search results as a JSON object.
        """
        headers = {
            "X-API-KEY": config.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "q": config.search_query,
        }

        url = "https://google.serper.dev/search"

        response = requests.post(url, json=payload, headers=headers)
        data = response.json()

        return data
