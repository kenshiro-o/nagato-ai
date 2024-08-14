from typing import Any, Type
import gzip
import chardet

from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup, Comment

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
        default=False,
        description="Whether to simplify the output by removing unnecessary script and style tags for instance. The default value is False.",
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

        try:
            # Send a GET request to the URL
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Check if the content is gzipped
            if response.headers.get("Content-Encoding") == "gzip":
                try:
                    content = gzip.decompress(response.content)
                except gzip.BadGzipFile:
                    # If decompression fails, use the original content
                    content = response.content
            else:
                content = response.content

            # Detect the encoding
            encoding = chardet.detect(content)["encoding"]

            # Decode the content
            html_content = content.decode(encoding or "utf-8", errors="replace")

            # Create a BeautifulSoup object to parse the HTML content
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script, style and link tags to reduce size of output
            for script in soup(["script", "style", "link"]):
                script.decompose()

            # Remove comments
            for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
                comment.extract()

            if not config.simplified_output:
                clean_html = soup.prettify()
                return {
                    "page_url": config.url,
                    "html_content": clean_html,
                }

            # Get the text content of the page
            text_content = soup.get_text(separator="\n", strip=True)
            return {
                "page_url": config.url,
                "html_content": text_content,
            }

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error occurred while scraping the web page: {str(e)}")
