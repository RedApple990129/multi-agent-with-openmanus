# OpenManus Knowledge Management

This directory contains the knowledge management components for the OpenManus agent platform, providing long-term memory and knowledge storage capabilities.

## Components

### Graph Database (Neo4j)

The `graph_db.py` module provides integration with Neo4j for storing structured knowledge in a graph format. This allows for complex relationships between entities and facts to be represented and queried efficiently.

### Vector Database (ChromaDB)

The `vector_db.py` module provides integration with ChromaDB for semantic search capabilities. This allows for retrieval of information based on meaning rather than exact keyword matching.

### Memory Manager

The `memory.py` module provides a unified interface for managing agent memory, combining both graph and vector databases. It handles:

- Storing factual information extracted from conversations
- Maintaining conversation history
- Retrieving relevant memories based on context
- Querying the knowledge graph

## Integration with Agent Workflow

The knowledge management system is integrated into the agent workflow through the knowledge agent (`src/agents/knowledge_agent.py`), which:

1. Extracts factual information from conversations
2. Stores this information in both graph and vector databases
3. Retrieves relevant memories based on the current conversation context
4. Provides this context to other agents in the workflow

## Configuration

The knowledge management system can be configured through environment variables:

- `NEO4J_URI`: URI for the Neo4j database (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME`: Username for Neo4j authentication (default: `neo4j`)
- `NEO4J_PASSWORD`: Password for Neo4j authentication (default: `password`)
- `CHROMA_PERSIST_DIR`: Directory for ChromaDB persistence (default: `./chroma_db`)

## Usage

The memory manager can be used directly in agent implementations:

```python
from src.knowledge.memory import MemoryManager

# Initialize memory manager
memory_manager = MemoryManager()
memory_manager.initialize()

# Store a fact
memory_manager.store_fact(
    "OpenManus is a knowledge-enhanced AI agent platform",
    source="documentation",
    related_entities=[
        {"type": "Project", "name": "OpenManus"}
    ]
)

# Retrieve relevant memories
memories = memory_manager.retrieve_relevant_memories(
    "Tell me about OpenManus capabilities"
)
```