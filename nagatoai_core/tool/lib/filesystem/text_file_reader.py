from typing import Any, Type

from pydantic import BaseModel, Field

from nagatoai_core.tool.abstract_tool import AbstractTool


class TextFileReaderConfig(BaseModel):
    """
    TextFileReaderConfig represents the configuration for the TextFileReaderTool.
    """

    file_path: str = Field(
        ...,
        description="The path to the text file to read",
    )


class TextFileReaderTool(AbstractTool):
    """
    TextFileReaderTool represents a tool that reads a text file from the specified path
    and returns its contents.
    """

    name: str = "text_file_reader"
    description: str = (
        """Reads a text file from the specified path and returns its contents."""
    )
    args_schema: Type[BaseModel] = TextFileReaderConfig

    def _run(self, config: TextFileReaderConfig) -> Any:
        """
        Reads a text file from the specified path and returns its contents.
        :param config: The configuration for the tool.
        :return: The contents of the text file as a string.
        """
        try:
            with open(config.file_path, "r") as file:
                content = file.read()
            return content
        except FileNotFoundError:
            return f"File not found: {config.file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
