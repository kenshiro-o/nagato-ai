# Standard Library
import json
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints

# Third Party
from pydantic import BaseModel, Field, create_model


class BaseModelGenerator:
    """
    A utility class that generates Pydantic BaseModel classes from JSON schemas.
    """

    @staticmethod
    def _get_field_type(schema_property: Dict[str, Any]) -> tuple:
        """
        Convert JSON schema type to a Python/Pydantic type.

        Returns:
            tuple: (type, field_params) where field_params contains additional validators
        """
        property_type = schema_property.get("type")
        field_params = {}

        # Add description if available
        if "description" in schema_property:
            field_params["description"] = schema_property["description"]

        # Add default if available
        if "default" in schema_property:
            field_params["default"] = schema_property["default"]

        if property_type == "string":
            if schema_property.get("format") == "email":
                return (str, field_params)
            return (str, field_params)
        elif property_type == "integer":
            if "minimum" in schema_property:
                field_params["ge"] = schema_property["minimum"]
            if "maximum" in schema_property:
                field_params["le"] = schema_property["maximum"]
            return (int, field_params)
        elif property_type == "number":
            return (float, field_params)
        elif property_type == "boolean":
            return (bool, field_params)
        elif property_type == "array":
            items = schema_property.get("items", {})
            item_type, _ = BaseModelGenerator._get_field_type(items)
            return (List[item_type], field_params)
        elif property_type == "object":
            # Create a nested model for objects
            nested_model = BaseModelGenerator._create_model_from_properties(
                schema_property.get("properties", {}), schema_property.get("required", [])
            )
            return (nested_model, field_params)
        else:
            # Default to Any if type is not specified or not recognized
            return (Any, field_params)

    @staticmethod
    def _create_model_from_properties(properties: Dict[str, Any], required: List[str]) -> Type[BaseModel]:
        """
        Create a Pydantic model from schema properties.
        """
        field_definitions = {}

        for field_name, field_schema in properties.items():
            field_type, field_params = BaseModelGenerator._get_field_type(field_schema)

            # Make field optional if not required
            if field_name not in required:
                # For non-required fields, add default=None
                field_params["default"] = field_params.get("default", None)

            field_definitions[field_name] = (field_type, Field(**field_params))

        # Create a new model with the field definitions
        return create_model("DynamicModel", **field_definitions)

    @classmethod
    def generate(cls, json_schema: str) -> Type[BaseModel]:
        """
        Generate a Pydantic BaseModel from a JSON schema string.

        Args:
            json_schema: JSON schema as a string

        Returns:
            Type[BaseModel]: A dynamically generated Pydantic BaseModel class
        """
        # Parse the JSON schema
        schema = json.loads(json_schema)

        # Get properties and required fields
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Create and return the model
        return cls._create_model_from_properties(properties, required)
