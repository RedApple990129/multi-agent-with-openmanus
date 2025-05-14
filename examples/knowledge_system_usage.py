"""Example usage of the OpenManus knowledge system.

This example demonstrates how to use the knowledge system components
including entity extraction, memory management, and knowledge retrieval.
"""

import json
from typing import Dict, List

# Import knowledge components
from src.knowledge.memory import MemoryManager
from src.prompts.template import OpenManusPromptTemplate
from src.llms.provider import get_llm

# Initialize components
def initialize_knowledge_system():
    """Initialize the knowledge system components."""
    print("Initializing knowledge system...")
    
    # Initialize memory manager
    memory_manager = MemoryManager()
    success = memory_manager.initialize()
    print(f"Memory manager initialized: {success}")
    
    # Initialize prompt templates
    OpenManusPromptTemplate.templates = {}
    OpenManusPromptTemplate.initialize()
    print("Prompt templates initialized")
    
    return memory_manager

# Entity extraction example
def extract_entities(conversation: str, memory_manager: MemoryManager):
    """Extract entities from a conversation and store them in memory."""
    print("\nExtracting entities from conversation...")
    
    # Get LLM
    llm = get_llm()
    
    # Apply entity extraction prompt template
    entity_extraction_prompt = OpenManusPromptTemplate.apply_prompt_template(
        "entity_extraction", {"conversation": conversation}
    )
    
    # Extract entities using LLM
    entity_extraction_response = llm.invoke(entity_extraction_prompt)
    
    try:
        # Parse entities from the response
        extracted_entities = json.loads(entity_extraction_response.content)
        print(f"Extracted {len(extracted_entities)} entities:")
        for entity in extracted_entities:
            print(f"  - {entity['name']} ({entity['type']})")
            
        # Store entities in memory
        for entity in extracted_entities:
            # Create a fact about this entity
            fact = f"{entity['name']} is a {entity['type']}"
            if entity.get('properties'):
                properties_str = ", ".join([f"{k}: {v}" for k, v in entity['properties'].items()])
                fact += f" with properties: {properties_str}"
            
            # Store the fact with the entity
            memory_manager.store_fact(
                fact,
                source="entity_extraction",
                related_entities=[entity]
            )
            
        return extracted_entities
    except Exception as e:
        print(f"Error parsing entities: {e}")
        return []

# Knowledge retrieval example
def retrieve_knowledge(query: str, memory_manager: MemoryManager):
    """Retrieve knowledge from memory based on a query."""
    print(f"\nRetrieving knowledge for query: '{query}'")
    
    # Retrieve relevant memories
    memories = memory_manager.retrieve_relevant_memories(query, limit=3)
    print(f"Found {len(memories)} relevant memories:")
    for i, memory in enumerate(memories):
        print(f"  {i+1}. {memory['document']}")
    
    # Retrieve entities by type
    entities = memory_manager.retrieve_by_entity_type("Person", limit=3)
    if entities:
        print(f"\nFound {len(entities)} Person entities:")
        for i, entity in enumerate(entities):
            print(f"  {i+1}. {entity['e'].get('name')}")
    
    # Retrieve recent memories
    recent = memory_manager.retrieve_recent_memories(days=1, limit=3)
    print(f"\nFound {len(recent)} recent memories")
    
    return memories

# Main example
def main():
    """Run the knowledge system example."""
    # Initialize the knowledge system
    memory_manager = initialize_knowledge_system()
    
    # Example conversation
    conversation = """
    User: I'm working with John Smith on the OpenManus project. It's an open-source 
    agent framework built with Python and LangGraph. We're planning to present it 
    at the AI Conference in San Francisco next month.
    
    Assistant: That sounds exciting! How long have you been working on the OpenManus project?
    
    User: We've been developing it for about 6 months. The main goal is to create a 
    flexible framework for building multi-agent systems with long-term memory.
    """
    
    # Extract entities
    entities = extract_entities(conversation, memory_manager)
    
    # Store the conversation
    memory_manager.store_conversation(
        [
            {"role": "user", "content": "I'm working with John Smith on the OpenManus project. It's an open-source agent framework built with Python and LangGraph. We're planning to present it at the AI Conference in San Francisco next month."},
            {"role": "assistant", "content": "That sounds exciting! How long have you been working on the OpenManus project?"},
            {"role": "user", "content": "We've been developing it for about 6 months. The main goal is to create a flexible framework for building multi-agent systems with long-term memory."}
        ],
        context="Project discussion",
        metadata={"source": "example"}
    )
    
    # Retrieve knowledge
    retrieve_knowledge("Tell me about the OpenManus project", memory_manager)
    
    print("\nKnowledge system example completed")

if __name__ == "__main__":
    main()