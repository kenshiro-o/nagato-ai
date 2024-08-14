from typing import List, Dict, Any

from nagatoai_core.chain.chain import Link


class AccumulatorLink(Link):
    """
    AccumulatorLink represents a link that accumulates the data it is given.
    It then forward this data without modifying it.
    The accumulated data can be accessed later.
    """

    accumulated_data: List[Any] = []

    class Config:
        extra = "ignore"

    def forward(self, input_data: Dict) -> Any:
        """
        Forward the data through the link
        :param input_data: The data that will be accumulated
        :return: The original input data
        """
        self.accumulated_data.append(input_data)
        return input_data

    def category(self) -> str:
        """
        Get the category of the link
        :return: The category of the link
        """
        return "ACCUMULATOR_LINK"
