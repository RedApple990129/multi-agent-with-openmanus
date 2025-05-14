# OpenManus Knowledge System

## Overview

The OpenManus Knowledge System provides long-term memory and knowledge management capabilities for the agent workflow. It combines graph and vector databases to store and retrieve information efficiently.

## Components

### Memory Manager

The Memory Manager is the central component that provides an interface for storing and retrieving memories. It uses both graph and vector databases for different types of knowledge representation.

**Key Features:**
- Store factual statements with related entities
- Store conversation history with context
- Retrieve relevant memories based on semantic similarity
- Categorize memories by type
- Query the knowledge graph directly
- Get information about specific entities

### Entity Extraction

The Entity Extraction component identifies named entities from conversations, including:
- People
- Organizations
- Projects
- Concepts
- Technologies
- Locations

Each entity is stored with its type and properties for future reference.

### Knowledge Extraction

The Knowledge Extraction component extracts factual statements from conversations to store in the knowledge graph and vector database.

## Usage Examples

### Storing Facts

```python
from src.knowledge.memory import MemoryManager

# Initialize memory manager
memory_manager = MemoryManager()
memory_manager.initialize()

# Store a fact with related entities
memory_manager.store_fact(
    "OpenManus is an open-source agent framework built on LangGraph.",
    source="documentation",
    related_entities=[
        {"name": "OpenManus", "type": "Project", "properties": {"domain": "AI"}},
        {"name": "LangGraph", "type": "Technology", "properties": {"type": "framework"}}
    ]
)
```

### Retrieving Relevant Memories

```python
# Retrieve memories relevant to a query
memories = memory_manager.retrieve_relevant_memories(
    "Tell me about the OpenManus project architecture",
    limit=5
)

# Categorize memories by type
categorized = memory_manager.categorize_memories(
    "What technologies does OpenManus use?"
)

# Get information about a specific entity
entity_info = memory_manager.get_entity_information("OpenManus")
```

### Integration with Workflow

The Knowledge Agent is integrated into the workflow graph and provides memory context to other agents:

```python
# In workflow graph
builder.add_node("knowledge", knowledge_agent_node)
builder.add_edge("coordinator", "knowledge") # Coordinator -> Knowledge
builder.add_edge("knowledge", "planner") # Knowledge -> Planner (with memory context)
```

## Configuration

The Knowledge System is initialized when the server starts:

```python
# Initialize knowledge system
memory_manager = MemoryManager()
memory_manager.initialize()

# Initialize prompt templates including entity and knowledge extraction
OpenManusPromptTemplate.templates = {}
OpenManusPromptTemplate.initialize()
```