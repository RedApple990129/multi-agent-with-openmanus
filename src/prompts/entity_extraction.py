"""Entity extraction prompt for OpenManus.

This module provides a prompt template for extracting entities from conversations.
"""

ENTITY_EXTRACTION_TEMPLATE = """
You are an AI assistant tasked with extracting entities from a conversation.
Entities are objects, people, organizations, concepts, or other named things that are mentioned in the text.

For each entity, identify:
1. The entity name
2. The entity type (Person, Organization, Project, Concept, Technology, Location, etc.)
3. Any properties or attributes mentioned about the entity

Output the entities in a JSON array format like this:
[
  {
    "name": "Entity name",
    "type": "Entity type",
    "properties": {
      "property1": "value1",
      "property2": "value2"
    }
  }
]

Conversation:
{conversation}

Extracted entities (JSON format):
"""

# Register the template
def register_templates(template_registry):
    """Register entity extraction template with the template registry."""
    template_registry["entity_extraction"] = ENTITY_EXTRACTION_TEMPLATE