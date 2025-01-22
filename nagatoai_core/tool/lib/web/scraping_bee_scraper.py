# Standard Library
from typing import Dict, Type

# Third Party
from bs4 import BeautifulSoup, Comment
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from scrapingbee import ScrapingBeeClient

# Nagato AI
# Company Libraries
from nagatoai_core.tool.abstract_tool import AbstractTool


class ScrapingBeeConfig(BaseSettings, BaseModel):
    api_key: str = Field(
        ...,
        description="The ScrapingBee API key to use for authentication.",
        alias="SCRAPINGBEE_API_KEY",
        exclude_from_schema=True,
    )

    url: str = Field(
        ...,
        description="The URL of the webpage to scrape.",
    )

    # params: Dict = Field(
    #     default={},
    #     description="Additional parameters for the ScrapingBee API request. The default value is just an empty dictionary.",
    # )


class ScrapingBeeTool(AbstractTool):
    """
    ScrapingBeeTool is a tool that scrapes the contents of a website using the ScrapingBee API.
    """

    name: str = "scrapingbee_web_page_scraper"
    description: str = (
        """Scrapes the contents of a specified webpage using the ScrapingBee API.
        Returns the scraped content and additional metadata.
        """
    )
    args_schema: Type[BaseModel] = ScrapingBeeConfig

    def _run(self, config: ScrapingBeeConfig) -> Dict:
        """
        Scrapes the contents of a webpage using the ScrapingBee API.
        :param config: The configuration for the ScrapingBeeTool.
        :return: A dictionary containing the scraped content and metadata.
        """
        try:
            client = ScrapingBeeClient(api_key=config.api_key)
            # response = client.get(config.url, params=config.params)
            response = client.get(config.url)

            html_content = response.content.decode("utf-8")

            soup = BeautifulSoup(html_content, "html.parser")
            # Remove script, style and link tags to reduce size of output
            for script in soup(["script", "style", "link"]):
                script.decompose()

            # Remove comments
            for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
                comment.extract()

            clean_html = soup.prettify()

            return {
                "status_code": response.status_code,
                "html_content": clean_html,
                "headers": dict(response.headers),
                "page_url": config.url,
            }

        except Exception as e:
            raise RuntimeError(f"Error scraping webpage: {str(e)}")
