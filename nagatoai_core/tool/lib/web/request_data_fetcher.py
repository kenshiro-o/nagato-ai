# Standard Library
import json
from typing import Any, Dict, Optional, Type, Union

# Third Party
import requests
from pydantic import BaseModel, Field

# Nagato AI
# Company Libraries
from nagatoai_core.tool.abstract_tool import AbstractTool


class RequestDataFetcherConfig(BaseModel):
    """
    Configuration for the RequestDataFetcher tool.
    """

    url: str = Field(
        ...,
        description="The URL to send the request to.",
    )
    headers: Dict[str, str] = Field(
        default={},
        description="Headers to send with the request. Default is an empty dictionary.",
    )
    method: str = Field(
        default="GET",
        description="HTTP method for the request. Either 'GET' or 'POST'. Default is 'GET'.",
    )
    body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The request body for POST requests. Default is None.",
    )
    json_output: bool = Field(
        default=False,
        description="Whether to return the response content as JSON. Default is False.",
    )


class RequestDataFetcherTool(AbstractTool):
    """
    RequestDataFetcherTool represents a tool that fetches data from a given URL using the requests library.
    """

    name: str = "request_data_fetcher"
    description: str = (
        """Fetches data from a specified URL using either GET or POST method.
        Allows customization of headers and request body. Can return response as JSON if specified."""
    )
    args_schema: Type[BaseModel] = RequestDataFetcherConfig

    def _run(self, config: RequestDataFetcherConfig) -> Dict[str, Any]:
        """
        Sends a request to the specified URL and returns the response.

        :param config: The configuration containing URL, headers, method, body, and json_output option.
        :return: A dictionary containing the response status code, headers, and content (as text or JSON).
        """
        try:
            if config.method.upper() not in ["GET", "POST"]:
                raise ValueError("Method must be either 'GET' or 'POST'")

            if config.method.upper() == "GET":
                response = requests.get(config.url, headers=config.headers)
            else:  # POST
                response = requests.post(config.url, headers=config.headers, json=config.body)

            response.raise_for_status()  # Raises an HTTPError for bad responses

            content: Union[str, Dict[str, Any]]
            if config.json_output:
                try:
                    content = response.json()
                except json.JSONDecodeError:
                    raise ValueError("Response content is not valid JSON")
            else:
                content = response.text

            return {
                "url": config.url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": content,
            }

        except requests.RequestException as e:
            raise Exception(f"Error fetching data: {str(e)}")
