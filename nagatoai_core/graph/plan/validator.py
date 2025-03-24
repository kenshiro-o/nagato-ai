"""XML Plan validator utility."""

# Standard Library
import json
import os
from typing import Optional

# Third Party
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from jsonschema.validators import validator_for
from lxml import etree


class XMLPlanValidator:
    """Validator for XML plan files."""

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the validator.

        Args:
            schema_path: Path to the XSD schema file. If None, will look for plan.xsd in the same directory.
        """
        if schema_path is None:
            # Get the directory of this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            schema_path = os.path.join(current_dir, "plan.xsd")

        # Load and parse the schema
        schema_doc = etree.parse(schema_path)
        self.schema = etree.XMLSchema(schema_doc)

    def validate_file(self, xml_file_path: str) -> tuple[bool, list[str]]:
        """
        Validate an XML file against the schema.

        Args:
            xml_file_path: Path to the XML file to validate.

        Returns:
            Tuple of (is_valid, error_messages). If valid, error_messages will be empty.
        """
        try:
            # Parse the XML file
            doc = etree.parse(xml_file_path)
            # Validate against schema
            self.schema.assertValid(doc)
            return True, []
        except etree.DocumentInvalid as e:
            return False, [str(e)]

    def validate_string(self, xml_string: str) -> tuple[bool, list[str]]:
        """
        Validate an XML string against the schema.

        Args:
            xml_string: The XML string to validate.

        Returns:
            Tuple of (is_valid, error_messages). If valid, error_messages will be empty.
        """
        try:
            # Parse the XML string
            doc = etree.fromstring(xml_string.encode())
            # Validate against schema
            self.schema.assertValid(doc)
            return True, []
        except etree.DocumentInvalid as e:
            return False, [str(e)]

    def validate_plan(self, xml_string: str) -> tuple[bool, list[str]]:
        """
        Validate a plan XML string against the schema.

        This is a convenience method that wraps validate_string() and adds plan-specific validation.

        Args:
            xml_string: The plan XML string to validate.

        Returns:
            Tuple of (is_valid, error_messages). If valid, error_messages will be empty.
        """
        # First validate against schema
        is_valid, error_messages = self.validate_string(xml_string)
        if not is_valid:
            return False, error_messages

        # Additional plan-specific validation
        try:
            doc = etree.fromstring(xml_string.encode())
            additional_errors = self._validate_plan_specific(doc)
            return len(additional_errors) == 0, additional_errors
        except Exception as e:
            return False, [f"Error during plan-specific validation: {str(e)}"]

    def _validate_plan_specific(self, doc: etree._Element) -> list[str]:
        """
        Perform additional plan-specific validation.

        This method checks for additional constraints that can't be expressed in the XSD,
        such as:
        - Node IDs referenced in edges must exist
        - Agent names referenced in nodes must exist
        - Tool names referenced in nodes must exist
        - Output schema names referenced in nodes must exist
        - JSONSchema validity in output_schema elements

        Args:
            doc: The parsed XML document.

        Returns:
            List of error messages. Empty list means no errors.
        """
        errors = []

        # Define namespace for XPath queries
        nsmap = {"ns": "http://nagatoai.com/schema/plan"}

        # Get all node IDs
        node_ids = set()
        for node in doc.xpath(".//*[@id]", namespaces=nsmap):
            node_ids.add(node.get("id"))

        # Get all agent names
        agent_names = set()
        for agent in doc.xpath(".//ns:agents/ns:agent", namespaces=nsmap):
            agent_names.add(agent.get("name"))

        # Get all output schema names
        schema_names = set()
        for schema in doc.xpath(".//ns:output_schemas/ns:output_schema", namespaces=nsmap):
            schema_names.add(schema.get("name"))

        # Validate edges
        for edge in doc.xpath(".//ns:graph/ns:edges/ns:edge", namespaces=nsmap):
            from_id = edge.get("from")
            to_id = edge.get("to")
            if from_id not in node_ids:
                errors.append(f"Edge references non-existent node ID: {from_id}")
            if to_id not in node_ids:
                errors.append(f"Edge references non-existent node ID: {to_id}")

        # Validate agent references
        for agent_ref in doc.xpath(".//ns:agent", namespaces=nsmap):
            agent_name = agent_ref.get("name")
            parent_tag = agent_ref.getparent().tag if agent_ref.getparent() is not None else None
            if agent_name not in agent_names and not parent_tag.endswith("agents"):
                errors.append(f"Node references non-existent agent: {agent_name}")

        # Validate output schema references
        for schema_ref in doc.xpath(".//ns:output_schema", namespaces=nsmap):
            schema_name = schema_ref.get("name")
            parent_tag = schema_ref.getparent().tag if schema_ref.getparent() is not None else None
            # Skip the definition of output_schema elements themselves
            if schema_name not in schema_names and not parent_tag.endswith("output_schemas"):
                errors.append(f"Node references non-existent output schema: {schema_name}")

        # Validate JSONSchema content
        for schema_elem in doc.xpath(".//ns:output_schemas/ns:output_schema/ns:schema", namespaces=nsmap):
            json_str = schema_elem.text
            if json_str:
                try:
                    json_str = json_str.strip()
                    json_obj = json.loads(json_str)
                    # Check if it's a valid JSONSchema
                    validator_class = validator_for(json_obj)
                    validator_class.check_schema(json_obj)
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid JSON in output schema: {e}")
                except JsonSchemaValidationError as e:
                    errors.append(f"Invalid JSONSchema in output schema: {e}")
                except Exception as e:
                    errors.append(f"Error validating JSONSchema: {e}")
            else:
                errors.append("Empty schema content in output_schema element")

        return errors
