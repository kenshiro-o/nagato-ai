from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, Field
import requests

from nagatoai_core.tool.abstract_tool import AbstractTool
from .base_config import BaseReadwiseConfig, READWISE_API_URL


class ReadwiseBookFinderConfig(BaseReadwiseConfig, BaseModel):
    """
    ReadwiseBookFinderConfig represents the configuration for the ReadwiseBookFinder tool.
    """

    book_name: str = Field(
        ...,
        description="The name of the book to search for in Readwise.",
    )


class ReadwiseBookFinderTool(AbstractTool):
    """
    ReadwiseBookFinderTool represents a tool that searches for a book in Readwise.
    """

    name: str = "readwise_book_finder"
    description: str = (
        """Searches for a book in Readwise given its name. Returns a JSON object that contains the details of the book stored in Readwise. If no book with this name is found, the response will be null."""
    )
    args_schema: Type[BaseModel] = ReadwiseBookFinderConfig

    def get_books(
        self, config: ReadwiseBookFinderConfig, url: str, page_size: int
    ) -> Any:
        """
        Get all books from Readwise
        :return: The result of the search.
        """
        headers = {
            "Authorization": f"Token {config.api_key}",
        }
        params = {
            "page_size": page_size,
            "category": "books",
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        response_json = response.json()

        return response_json

    def _run(self, config: ReadwiseBookFinderConfig) -> Any:
        """
        Searches for a book in Readwise given its name.
        :param book_name: The name of the book to search for.
        :return: The result of the search.
        """
        page_size = 50
        response = self.get_books(config, f"{READWISE_API_URL}/books/", page_size)
        current_count = 0
        total_books = response["count"]

        while True:
            results = response["results"]
            current_count += len(results)

            for result in results:
                if config.book_name.lower() == result["title"].lower():
                    return result

            if current_count >= total_books:
                break

            next_page = response["next"]
            if not next_page:
                break

            response = self.get_books(config, next_page, page_size)

        # No book found - maybe we should return an error in the future
        return None
