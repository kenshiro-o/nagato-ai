# Standard Library
import csv
from typing import Any, Dict, List, Type, Union

# Third Party
from pydantic import BaseModel, Field

# Nagato AI
# Company Libraries
from nagatoai_core.tool.abstract_tool import AbstractTool


class CSVReaderConfig(BaseModel):
    """
    CSVReaderConfig represents the configuration for the CSVReader tool.
    """

    full_path: str = Field(
        ...,
        description="The full path to the CSV file to be read.",
    )

    read_as_dict: bool = Field(
        default=False,
        description="Whether to read the CSV as a dictionary using csv.DictReader. The default value is False (read the CSV file as a list).",
    )

    skip_header: bool = Field(
        default=False,
        description="Whether to skip the header row when reading the CSV file. The default value is False (i.e. - we keep the header).",
    )


class CSVReaderTool(AbstractTool):
    """
    CSVReaderTool represents a tool that reads a CSV file and returns its contents.
    """

    name: str = "csv_reader"
    description: str = (
        """Reads a CSV file from the given full path and returns its contents.
        Can read the CSV as a list of lists or a list of dictionaries."""
    )
    args_schema: Type[BaseModel] = CSVReaderConfig

    def _run(self, config: CSVReaderConfig) -> Union[List[List[str]], List[Dict[str, str]]]:
        """
        Reads the CSV file based on the provided configuration.

        :param config: The configuration containing the full path to the CSV file
                       and whether to read it as a dictionary.
        :return: The contents of the CSV file as either a list of lists or a list of dictionaries.
        """
        try:
            ret = {"full_path": config.full_path, "rows": []}

            with open(config.full_path, "r", newline="", encoding="utf-8") as csvfile:
                if config.read_as_dict:
                    reader = csv.DictReader(csvfile)
                    for i, row in enumerate(reader):
                        if i == 0:
                            if not config.skip_header:
                                ret["header"] = list(row.keys())
                            continue

                        ret["rows"].append(row)

                    return ret

                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    if i == 0:
                        if not config.skip_header:
                            ret["header"] = row
                        continue

                    ret["rows"].append(row)

                return ret

        except FileNotFoundError as fe:
            raise Exception(f"CSV file not found at path: {config.full_path}") from fe
        except csv.Error as e:
            raise Exception(f"Error reading CSV file at path: {config.full_path}") from e
        except Exception as e:
            raise Exception(
                f"An unexpected error occurred while reading the CSV file at path: {config.full_path}"
            ) from e
