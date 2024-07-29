import os
from typing import Type
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings

from nagatoai_core.tool.abstract_tool import AbstractTool


class TextFileWriterConfig(BaseSettings, BaseModel):
    full_path: str = Field(
        ...,
        description="The full file path for the target text file we are writing to.",
    )

    content: str = Field(
        ...,
        description="The content we are writing into the text file.",
    )


class TextFileWriterTool(AbstractTool):
    """
    TextFileWriterTool is a tool that writes content to a text file.
    """

    name: str = "text_file_writer"
    description: str = (
        """Writes the provided content to a text file at the specified file path.
        Returns information about the writing process and the output file.
        """
    )
    args_schema: Type[BaseModel] = TextFileWriterConfig

    def _run(self, config: TextFileWriterConfig) -> dict:
        """
        Writes content to a text file.
        :param config: The configuration for the TextFileWriterTool.
        :return: A dictionary containing information about the writing process and the output file.
        """
        try:
            # Check if the file path is just a file name
            if not os.path.isabs(config.full_path):
                # If the path is not absolute, use the current working directory
                config.full_path = os.path.join(os.getcwd(), config.full_path)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(config.full_path), exist_ok=True)

            # Write the content to the file
            with open(config.full_path, "w", encoding="utf-8") as file:
                file.write(config.content)

            # Get file information
            file_size = os.path.getsize(config.full_path)
            file_name = os.path.basename(config.full_path)

            return {
                "status": "success",
                "full_path": config.full_path,
                "file_name": file_name,
                "file_size_bytes": file_size,
                "characters_written": len(config.content),
            }

        except Exception as e:
            raise RuntimeError(f"Error writing to text file: {str(e)}")
