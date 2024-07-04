from typing import Dict, Any, Callable

from nagatoai_core.chain.chain import Link


class AdaptorLink(Link):
    """
    AdaptorLink takes input params and transforms it into another format
    """

    adapter_fn: Callable[[Any], Any]

    def forward(self, input_data: Dict) -> Any:
        """
        Forward the data through the link
        :param input_data: input data that will be converted via the adapter
        :return: The output data that will be converted via the adapter
        """
        return self.adapter_fn(input_data)

    def category(self) -> str:
        """
        Get the category of the link
        :return: The category of the link
        """
        return "ADAPTOR_LINK"
