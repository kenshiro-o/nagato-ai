from typing import Dict
import traceback

from pydantic import BaseModel


class PromptTemplate(BaseModel):
    """
    PromptTemplate represents a prompt template with placeholder variables that can be used to generate prompts.
    """

    template: str
    data_light_schema: Dict[str, str]

    def generate_prompt(self, data: Dict) -> str:
        """
        Generate a prompt using the template and the placeholder variables.
        :data: The data that will be used to replace the placeholder variables in the template
        :return: The generated prompt
        """
        try:
            return self.template.format(**data)
        except Exception as e:
            print(traceback.format_exc())
            raise e
