# Standard Library
import time
from typing import Any, Type

# Third Party
import requests
from pydantic import BaseModel, Field

# Nagato AI
# Company Libraries
from nagatoai_core.tool.abstract_tool import AbstractTool

from .base_config import READWISE_API_URL, BaseReadwiseConfig

SLEEP_TIME_BETWEEN_TOOL_CALLS_IN_SECONDS = 3


class ReadwiseDocumentFinderConfig(BaseReadwiseConfig, BaseModel):
    """
    ReadwiseDocumentFinderConfig represents the configuration for the ReadwiseDocumentFinderTool tool.
    """

    document_name: str = Field(
        ...,
        description="The name of the document to search for in Readwise.",
    )
    document_category: str = Field(
        default="",
        description="The category of the document to search for in Readwise. Allowed values are 'books', 'articles', 'tweets', 'authors', 'supplementals', 'podcasts', or simply empty string '' if no category is specified. This is an optional field, so the default is empty string",
    )


class ReadwiseDocumentFinderTool(AbstractTool):
    """
    ReadwiseDocumentFinderTool represents a tool that searches for a document in Readwise.
    """

    name: str = "readwise_document_finder"
    description: str = (
        """readwise_document_finder searches for a document in Readwise given its name. Returns a JSON object that contains the details of the document stored in Readwise.
        If no document with this name is found, the response will be null.
        """
    )
    args_schema: Type[BaseModel] = ReadwiseDocumentFinderConfig

    def get_documents(self, config: ReadwiseDocumentFinderConfig, url: str, page_size: int) -> Any:
        """
        Get all documents from Readwise
        :config: The configuration for the ReadwiseDocumentFinder tool.
        :param url: The URL to get the documents from.
        :page_size: The number of documents to get per page.
        :return: The result of the search.
        """
        headers = {
            "Authorization": f"Token {config.api_key}",
        }
        # Only add params if we're not using a next_page URL (which already includes params)
        if "page=" not in url:
            params = {
                "page_size": page_size,
            }
            if config.document_category:
                params["category"] = config.document_category
        else:
            params = {}

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        response_json = response.json()

        return response_json

    def _run(self, config: ReadwiseDocumentFinderConfig) -> Any:
        """
        Searches for a document in Readwise given its name.
        :param config: The configuration for the ReadwiseDocumentFinder tool.
        :return: The result of the search.
        """
        page_size = 50
        response = self.get_documents(config, f"{READWISE_API_URL}/books/", page_size)
        current_count = 0
        total_documents = response["count"]

        config_doc_name_sanitised = config.document_name.lower().strip().replace("  ", " ")
        while True:
            results = response["results"]
            current_count += len(results)

            for result in results:
                # Remove leading/trailing spaces and make sure we never have double spaces either
                result_title_sanitised = result["title"].lower().strip().replace("  ", " ")
                if config_doc_name_sanitised == result_title_sanitised:
                    return result

            if current_count >= total_documents:
                break

            next_page = response["next"]
            if not next_page:
                break

            # Sleep for a bit to prevent Readwise from returning TOO MANY REQUESTS error
            time.sleep(SLEEP_TIME_BETWEEN_TOOL_CALLS_IN_SECONDS)

            response = self.get_documents(config, next_page, page_size)

        # No document found - maybe we should return an error in the future
        return None
