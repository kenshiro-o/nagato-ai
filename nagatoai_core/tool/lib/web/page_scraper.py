from typing import Any, Type

from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup

from nagatoai_core.tool.abstract_tool import AbstractTool


class WebPageScraperConfig(BaseModel):
    """
    WebPageScraperConfig represents the configuration for the WebPageScraper tool.
    """

    url: str = Field(
        ...,
        description="The URL of the web page to scrape.",
    )

    simplified_output: bool = Field(
        default=True,
        description="Whether to simplify the output by removing unnecessary script and style tags for instance. The default value is True.",
    )


class WebPageScraperTool(AbstractTool):
    """
    WebPageScraperTool represents a tool that scrapes a specific web page using BeautifulSoup.
    """

    name: str = "web_page_scraper"
    description: str = (
        """Scrapes the content of a specific web page given its URL. Returns the HTML content of the page as a string. If the URL is invalid or the page cannot be accessed, an error will be raised."""
    )
    args_schema: Type[BaseModel] = WebPageScraperConfig

    def _run(self, config: WebPageScraperConfig) -> Any:
        """
        Scrapes the content of a specific web page using BeautifulSoup.
        :param config: The configuration containing the URL of the web page to scrape.
        :return: The HTML content of the web page as a string.
        """
        url = config.url
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36"
        }

        try:
            # Send a GET request to the URL
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Create a BeautifulSoup object to parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style tags to reduce size of output
            for script in soup(["script", "style"]):
                script.decompose()

            # Get the text content of the page
            text_content = soup.get_text(separator="\n", strip=True)

            return text_content

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error occurred while scraping the web page: {str(e)}")
