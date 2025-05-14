"""Knowledge extraction prompt template for OpenManus.

This module provides a prompt template for extracting factual knowledge
from conversations to store in the knowledge graph and vector database.
"""

KNOWLEDGE_EXTRACTION_TEMPLATE = """
You are a knowledge extraction system. Your task is to extract factual statements from the conversation.
Extract only clear, factual information that would be useful to remember for future reference.
Do not include opinions, speculations, or uncertain information.

For each fact, provide:
1. The factual statement in a clear, concise format
2. Any entities mentioned (people, organizations, concepts, etc.)
3. The source of the information if mentioned

Conversation:
{conversation}

Extract the facts in the following JSON format:
[
  {{
    "content": "Factual statement 1",
    "entities": [{{"type": "Person", "name": "John Doe"}}, {{"type": "Organization", "name": "Acme Corp"}}],
    "source": "mentioned in conversation"
  }},
  ...
]

If no clear facts are present, return an empty array: []
"""

# Register the template
def register_templates(template_registry):
    """Register knowledge extraction template with the template registry."""
    template_registry["knowledge_extraction"] = KNOWLEDGE_EXTRACTION_TEMPLATE