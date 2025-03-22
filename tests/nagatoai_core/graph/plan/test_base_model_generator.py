import json
import pytest
from pydantic import BaseModel
from nagatoai_core.graph.plan.base_model_generator import BaseModelGenerator


def test_simple_schema():
    """Test generation of a model from a simple JSON schema."""
    schema = """
    {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "The person's name"
        },
        "age": {
          "type": "integer",
          "minimum": 0,
          "description": "The person's age"
        }
      },
      "required": ["name", "age"]
    }
    """

    # Generate the model
    model_class = BaseModelGenerator.generate(schema)

    # Verify it's a subclass of BaseModel
    assert issubclass(model_class, BaseModel)

    # Verify required fields
    model_fields = model_class.__annotations__
    assert "name" in model_fields
    assert "age" in model_fields

    # Create an instance and verify validation
    instance = model_class(name="Test User", age=25)
    assert instance.name == "Test User"
    assert instance.age == 25

    # Test validation error for negative age
    with pytest.raises(ValueError):
        model_class(name="Test User", age=-1)

    # Test validation error for missing required field
    with pytest.raises(ValueError):
        model_class(name="Test User")


def test_complex_schema():
    """Test generation of a model from a complex JSON schema with nested objects and arrays."""
    schema = """
    {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "The person's name"
        },
        "age": {
          "type": "integer",
          "minimum": 0,
          "description": "The person's age"
        },
        "email": {
          "type": "string",
          "format": "email",
          "description": "The person's email address"
        },
        "is_active": {
          "type": "boolean",
          "default": true
        },
        "tags": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "address": {
          "type": "object",
          "properties": {
            "street": {"type": "string"},
            "city": {"type": "string"},
            "country": {"type": "string"}
          },
          "required": ["street", "city"]
        }
      },
      "required": ["name", "age"]
    }
    """

    # Generate the model
    model_class = BaseModelGenerator.generate(schema)

    # Create a valid instance
    instance = model_class(
        name="Test User",
        age=30,
        email="test@example.com",
        tags=["tag1", "tag2"],
        address={"street": "123 Main St", "city": "Test City"},
    )

    # Verify field values
    assert instance.name == "Test User"
    assert instance.age == 30
    assert instance.email == "test@example.com"
    assert instance.is_active is True  # Default value
    assert instance.tags == ["tag1", "tag2"]
    assert instance.address.street == "123 Main St"
    assert instance.address.city == "Test City"

    # Verify nested validation works
    with pytest.raises(ValueError):
        model_class(name="Test User", age=30, address={"street": "123 Main St"})  # Missing required 'city'


def test_optional_fields():
    """Test handling of optional fields."""
    schema = """
    {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "bio": {"type": "string"}
      },
      "required": ["name"]
    }
    """

    model_class = BaseModelGenerator.generate(schema)

    # Test with only required field
    instance = model_class(name="Test User")
    assert instance.name == "Test User"
    assert not hasattr(instance, "age") or instance.age is None

    # Test with optional fields
    instance = model_class(name="Test User", age=25, bio="Test bio")
    assert instance.name == "Test User"
    assert instance.age == 25
    assert instance.bio == "Test bio"


def test_defaults():
    """Test default values in schema."""
    schema = """
    {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "is_active": {"type": "boolean", "default": true},
        "count": {"type": "integer", "default": 0}
      },
      "required": ["name"]
    }
    """

    model_class = BaseModelGenerator.generate(schema)
    instance = model_class(name="Test User")

    assert instance.name == "Test User"
    assert instance.is_active is True
    assert instance.count == 0


def test_nested_array_models():
    """Test generation of models with nested arrays of objects like Highlight/Highlights."""
    schema = """
    {
      "type": "object",
      "properties": {
        "highlights": {
          "type": "array",
          "description": "A list of highlights.",
          "items": {
            "type": "object",
            "properties": {
              "start_offset_seconds": {
                "type": "number",
                "description": "The start offset of the highlight in the transcript in seconds."
              },
              "end_offset_seconds": {
                "type": "number",
                "description": "The end offset of the highlight in the transcript in seconds."
              },
              "transcript_text": {
                "type": "string",
                "description": "The transcript text of the highlight."
              },
              "reason": {
                "type": "string",
                "description": "The reason for selecting the highlight."
              }
            },
            "required": ["start_offset_seconds", "end_offset_seconds", "transcript_text", "reason"]
          }
        }
      },
      "required": ["highlights"]
    }
    """

    # Generate the model
    model_class = BaseModelGenerator.generate(schema)

    # Verify the model structure
    model_fields = model_class.__annotations__
    assert "highlights" in model_fields

    # Create a valid instance
    instance = model_class(
        highlights=[
            {
                "start_offset_seconds": 10.5,
                "end_offset_seconds": 20.75,
                "transcript_text": "This is an important part of the video.",
                "reason": "Key information about the topic",
            },
            {
                "start_offset_seconds": 45.0,
                "end_offset_seconds": 55.25,
                "transcript_text": "Another significant section worth noting.",
                "reason": "Contains critical data",
            },
        ]
    )

    # Verify values in the nested structure
    assert len(instance.highlights) == 2
    assert instance.highlights[0].start_offset_seconds == 10.5
    assert instance.highlights[0].end_offset_seconds == 20.75
    assert instance.highlights[0].transcript_text == "This is an important part of the video."
    assert instance.highlights[0].reason == "Key information about the topic"

    # Test validation errors in nested objects
    with pytest.raises(ValueError):
        model_class(
            highlights=[
                {
                    "start_offset_seconds": 10.5,
                    # Missing end_offset_seconds
                    "transcript_text": "This is an important part of the video.",
                    "reason": "Key information about the topic",
                }
            ]
        )

    # Test with empty list
    empty_instance = model_class(highlights=[])
    assert len(empty_instance.highlights) == 0
