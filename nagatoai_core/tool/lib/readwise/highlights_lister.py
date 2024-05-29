from typing import Any, Type, List, Optional

from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from dateutil.parser import parse
import requests

from nagatoai_core.tool.abstract_tool import AbstractTool
from .base_config import BaseReadwiseConfig, READWISE_API_URL


class ReadwiseHighightsListerConfig(BaseReadwiseConfig, BaseModel):
    """
    ReadwiseHighightsListerConfig represents the configuration for the ReadwiseHighightsLister tool.
    """

    tags: Optional[List[str]] = Field(
        default=None,
        description="The array of tags to filter the highlights by. This field is optional.",
    )

    max_count: Optional[int] = Field(
        default=None,
        description="The maximum number of highlights to fetch. This field is optional.",
    )

    from_datetime: Optional[str] = Field(
        default=None,
        description="The datetime string from which to fetch the highlights. Values should be of the format YYYY-MM-DDThh:mm:ssZ (e.g. 2020-02-01T21:35:53Z). This field is optional.",
    )

    to_datetime: Optional[str] = Field(
        default=None,
        description="The datetime string to which to fetch the highlights. Values should be of the format YYYY-MM-DDThh:mm:ssZ (e.g. 2020-02-01T21:35:53Z). This field is optional.",
    )


class ReadwiseHighightsListerTool(AbstractTool):
    """
    ReadwiseHighightsListerTool represents a tool that lists the highlights from Readwise.
    You can filter the highlights using optional filters like tags and date range.
    """

    name: str = "readwise_highlights_lister"
    description: str = (
        """Lists the highlights from Readwise. You can filter the highlights using optional filters like tags array and datetime range."""
    )
    args_schema: Type[BaseModel] = ReadwiseHighightsListerConfig

    def get_highlights(
        self, config: ReadwiseHighightsListerConfig, url: str, page_size: int
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
        }

        if config.from_datetime:
            from_datetime_dt = parse(config.from_datetime)
            from_datetime_dt_str = from_datetime_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            params["updated__gt"] = from_datetime_dt_str

        if config.to_datetime:
            to_datetime_dt = parse(config.to_datetime)
            to_datetime_dt_str = to_datetime_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            params["updated__lt"] = to_datetime_dt_str

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        response_json = response.json()

        return response_json

    def _run(self, config: ReadwiseHighightsListerConfig) -> Any:
        """
        Lists the highlights from Readwise given the optional filters.
        :return: The result of the list operation.
        """
        page_size = 50
        response = self.get_highlights(
            config, f"{READWISE_API_URL}/highlights/", page_size
        )
        current_count = 0

        highlights = []
        while True:
            results = response["results"]
            next_page = response["next"]

            for highlight in results:
                highlight_tag_names = [tag["name"] for tag in highlight["tags"]]
                if config.tags and not any(
                    tag in highlight_tag_names for tag in config.tags
                ):
                    continue

                highlights.append(highlight)
                current_count += 1
                if config.max_count and current_count >= config.max_count:
                    return highlights

            if not next_page:
                break

            response = self.get_highlights(config, next_page, page_size)

        return highlights
