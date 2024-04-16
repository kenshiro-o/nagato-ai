from typing import Any, Type

from pydantic import BaseModel, Field
import requests

from nagatoai_core.tool.abstract_tool import AbstractTool
from .base_config import BaseReadwiseConfig, READWISE_API_URL


class ReadwiseBookHighlightsListerConfig(BaseReadwiseConfig, BaseModel):
    """
    ReadwiseBookHighlightsListerConfig represents the configuration for the ReadwiseBookHighlightsLister tool.
    """

    book_id: int = Field(
        ...,
        description="The ID of the book to list the highlights for in Readwise.",
    )


class ReadwiseBookHighlightsListerTool(AbstractTool):
    """
    ReadwiseBookHighlightsListerTool represents a tool that lists the highlights for a book in Readwise.
    """

    name: str = "readwise_book_highlights_lister"
    description: str = (
        """Lists the highlights for a book in Readwise given its ID. Returns a JSON object that contains the highlights of the book stored in Readwise. If no book with this ID is found, the response will be and empty array."""
    )
    args_schema: Type[BaseModel] = ReadwiseBookHighlightsListerConfig

    def get_highlights(
        self, config: ReadwiseBookHighlightsListerConfig, url: str, page_size: int
    ) -> Any:
        """
        Get all highlights from Readwise
        :return: The result of the list operation.
        """
        headers = {
            "Authorization": f"Token {config.api_key}",
        }
        params = {
            "page_size": page_size,
            "book_id": config.book_id,
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        response_json = response.json()

        return response_json

    def _run(self, config: ReadwiseBookHighlightsListerConfig) -> Any:
        """
        Lists the highlights for a book in Readwise given its ID.
        :param book_id: The ID of the book to list the highlights for.
        :return: The result of the list operation.
        """
        page_size = 50
        response = self.get_highlights(
            config, f"{READWISE_API_URL}/highlights/", page_size
        )
        current_count = 0
        total_highlights = response["count"]

        highlights = []
        while True:
            results = response["results"]
            highlights.extend(results)

            current_count += len(results)
            if current_count >= total_highlights:
                break

            next_page = response["next"]
            if not next_page:
                break

        return highlights
