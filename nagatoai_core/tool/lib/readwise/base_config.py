from pydantic import Field
from pydantic_settings import BaseSettings

READWISE_API_URL = "https://readwise.io/api/v2"


class BaseReadwiseConfig(BaseSettings):
    """
    BaseReadwiseConfig represents the base configuration for all Readwise tools.
    """

    api_key: str = Field(
        ...,
        description="The Readwise API key to use for authentication.",
        alias="READWISE_API_KEY",
        exclude_from_schema=True,
    )
