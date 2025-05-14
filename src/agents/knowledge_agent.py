"""Knowledge agent for OpenManus.

This module provides a knowledge agent that integrates with the memory manager
to provide long-term memory and knowledge retrieval capabilities.
"""

import json
from typing import Dict, List, Any

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from src.knowledge.memory import MemoryManager
from src.graph.types import State
from src.prompts.template import OpenManusPromptTemplate
from src.llms.provider import get_llm

# Initialize memory manager
memory_manager = MemoryManager()

def knowledge_agent(state: State) -> Dict[str, Any]:
    """Knowledge agent for retrieving and storing information in long-term memory.
    
    Args:
        state: Current state of the workflow
        
    Returns:
        Updated state with knowledge context
    """
    # Initialize memory manager if not already initialized
    if not memory_manager.initialized:
        memory_manager.initialize()
    
    # Extract the latest message and conversation context
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    conversation_context = state.get("context", "")
    
    # Retrieve relevant memories based on the latest message
    relevant_memories = memory_manager.retrieve_relevant_memories(latest_message)
    
    # Categorize memories for better context organization
    categorized_memories = memory_manager.categorize_memories(latest_message)
    
    # Store the conversation in memory with context
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "role": "user" if isinstance(msg, HumanMessage) else "assistant" if isinstance(msg, AIMessage) else "system",
            "content": msg.content
        })
    
    memory_manager.store_conversation(
        formatted_messages,
        context=conversation_context,
        metadata={"source": "agent_workflow"}
    )
    
    # Extract facts from the conversation to store in memory
    llm = get_llm()
    fact_extraction_prompt = OpenManusPromptTemplate.apply_prompt_template(
        "knowledge_extraction", {"conversation": latest_message}
    )
    
    fact_extraction_response = llm.invoke(fact_extraction_prompt)
    
    # Extract entities from the conversation
    entities = []
    entity_extraction_prompt = OpenManusPromptTemplate.apply_prompt_template(
        "entity_extraction", {"conversation": latest_message}
    )
    
    entity_extraction_response = llm.invoke(entity_extraction_prompt)
    
    try:
        # Parse entities from the response
        extracted_entities = json.loads(entity_extraction_response.content)
        entities = extracted_entities
    except Exception as e:
        print(f"Error parsing entities: {e}")
        entities = []
    
    try:
        # Parse facts from the response
        facts = json.loads(fact_extraction_response.content)
        
        # Store facts in memory with related entities
        for fact in facts:
            related_entities = []
            # Link facts to relevant entities
            for entity in entities:
                if entity["name"].lower() in fact["content"].lower():
                    related_entities.append(entity)
            
            memory_manager.store_fact(
                fact["content"], 
                source="conversation",
                related_entities=related_entities
            )
    except Exception as e:
        print(f"Error storing facts: {e}")
    
    # Get entity information for key entities in the conversation
    entity_info = {}
    for entity in entities:
        entity_name = entity.get("name")
        if entity_name:
            entity_info[entity_name] = memory_manager.get_entity_information(entity_name)
    
    # Update state with memory context
    return {
        "memory_context": relevant_memories,
        "categorized_memories": categorized_memories,
        "entity_information": entity_info,
        "messages": messages
    }

# Register the knowledge agent
knowledge_agent_node = knowledge_agent