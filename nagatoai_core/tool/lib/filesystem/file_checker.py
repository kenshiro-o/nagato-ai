import os
from typing import Type
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings

from nagatoai_core.tool.abstract_tool import AbstractTool


class FileCheckerConfig(BaseSettings, BaseModel):
    full_path: str = Field(
        ...,
        description="The full path of the file to check for existence.",
    )


class FileCheckerTool(AbstractTool):
    """
    FileCheckerTool is a tool that checks whether a file exists at a given path.
    """

    name: str = "file_checker"
    description: str = (
        """Checks whether a file exists at the specified full path.
        Returns a boolean indicating if the file exists and additional information about the file if it does exist.
        """
    )
    args_schema: Type[BaseModel] = FileCheckerConfig

    def _run(self, config: FileCheckerConfig) -> dict:
        """
        Checks whether a file exists at the given path and provides information about the file if it exists.
        :param config: The configuration for the FileCheckerTool.
        :return: A dictionary containing information about the file existence and properties.
        """
        try:
            if os.path.exists(config.full_path):
                file_stats = os.stat(config.full_path)
                return {
                    "exists": True,
                    "is_file": os.path.isfile(config.full_path),
                    "is_directory": os.path.isdir(config.full_path),
                    "size_bytes": file_stats.st_size,
                    "last_modified": file_stats.st_mtime,
                    "permissions": oct(file_stats.st_mode)[-3:],
                }
            else:
                return {"exists": False}
        except Exception as e:
            raise RuntimeError(f"Error checking file: {str(e)}")
