"""Memory management for OpenManus.

This module provides a MemoryManager class for managing long-term memory
for agents, combining graph and vector databases for knowledge storage.
"""

from typing import Dict, List, Optional, Union
import datetime
import json

from .graph_db import GraphDatabase
from .vector_db import VectorDatabase
from .memory_retrieval import MemoryRetrieval

class MemoryManager:
    """Memory manager for OpenManus agents.
    
    This class provides methods for storing and retrieving memories,
    using both graph and vector databases for different types of knowledge.
    """
    
    def __init__(self, graph_db: GraphDatabase = None, vector_db: VectorDatabase = None):
        """Initialize the memory manager.
        
        Args:
            graph_db: GraphDatabase instance
            vector_db: VectorDatabase instance
        """
        self.graph_db = graph_db or GraphDatabase()
        self.vector_db = vector_db or VectorDatabase()
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize connections to databases.
        
        Returns:
            True if initialization successful, False otherwise
        """
        graph_connected = self.graph_db.connect()
        vector_connected = self.vector_db.connect()
        self.initialized = graph_connected and vector_connected
        return self.initialized
    
    def store_fact(self, fact: str, source: str = None, 
                  related_entities: List[Dict] = None) -> Optional[int]:
        """Store a factual statement in memory.
        
        Args:
            fact: The factual statement to store
            source: Source of the information
            related_entities: List of related entities with their types
            
        Returns:
            Node ID if successful, None otherwise
        """
        if not self.initialized and not self.initialize():
            return None
        
        # Store in graph database
        properties = {
            "content": fact,
            "timestamp": datetime.datetime.now().isoformat(),
            "source": source or "agent"
        }
        
        fact_node_id = self.graph_db.create_node(["Fact"], properties)
        
        # Create relationships to related entities if provided
        if related_entities and fact_node_id:
            for entity in related_entities:
                entity_node_id = self.graph_db.create_node(
                    [entity.get("type", "Entity")], 
                    {"name": entity.get("name"), "properties": json.dumps(entity.get("properties", {}))}
                )
                if entity_node_id:
                    self.graph_db.create_relationship(
                        fact_node_id, entity_node_id, "MENTIONS"
                    )
        
        # Store in vector database for semantic search
        self.vector_db.add_documents(
            documents=[fact],
            metadatas=[{
                "type": "fact",
                "source": source or "agent",
                "timestamp": datetime.datetime.now().isoformat(),
                "graph_id": str(fact_node_id) if fact_node_id else None
            }],
            ids=[f"fact_{fact_node_id}" if fact_node_id else None]
        )
        
        return fact_node_id
    
    def store_conversation(self, messages: List[Dict], 
                         context: str = None, metadata: Dict = None) -> bool:
        """Store conversation history in memory.
        
        Args:
            messages: List of message dictionaries with role and content
            context: Context of the conversation
            metadata: Additional metadata about the conversation
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized and not self.initialize():
            return False
        
        # Create conversation node in graph
        metadata = metadata or {}
        properties = {
            "timestamp": datetime.datetime.now().isoformat(),
            "context": context or "",
            **metadata
        }
        
        conversation_node_id = self.graph_db.create_node(["Conversation"], properties)
        
        # Add message nodes and link to conversation
        for i, message in enumerate(messages):
            message_properties = {
                "role": message.get("role", "unknown"),
                "content": message.get("content", ""),
                "timestamp": message.get("timestamp", datetime.datetime.now().isoformat()),
                "order": i
            }
            
            message_node_id = self.graph_db.create_node(["Message"], message_properties)
            
            if message_node_id and conversation_node_id:
                self.graph_db.create_relationship(
                    conversation_node_id, message_node_id, "CONTAINS"
                )
        
        # Store in vector database for semantic search
        # Combine messages into a single document for context
        conversation_text = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages])
        
        self.vector_db.add_documents(
            documents=[conversation_text],
            metadatas=[{
                "type": "conversation",
                "timestamp": datetime.datetime.now().isoformat(),
                "context": context or "",
                "graph_id": str(conversation_node_id) if conversation_node_id else None,
                **metadata
            }],
            ids=[f"conversation_{conversation_node_id}" if conversation_node_id else None]
        )
        
        return True
    
    def retrieve_relevant_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve memories relevant to a query.
        
        Args:
            query: Query text to find relevant memories
            limit: Maximum number of results to return
            
        Returns:
            List of relevant memories with their metadata
        """
        if not self.initialized and not self.initialize():
            return []
        
        # Search vector database for semantic similarity
        results = self.vector_db.query(query, n_results=limit)
        
        # Enhance results with graph information if available
        enhanced_results = []
        for result in results:
            graph_id = result.get("metadata", {}).get("graph_id")
            if graph_id:
                # Get additional context from graph database
                if result.get("metadata", {}).get("type") == "fact":
                    # For facts, get related entities
                    related = self.graph_db.query(
                        "MATCH (f:Fact)-[:MENTIONS]->(e) WHERE id(f) = $id RETURN e",
                        {"id": int(graph_id)}
                    )
                    result["related_entities"] = related
                elif result.get("metadata", {}).get("type") == "conversation":
                    # For conversations, get messages in order
                    messages = self.graph_db.query(
                        "MATCH (c:Conversation)-[:CONTAINS]->(m:Message) WHERE id(c) = $id RETURN m ORDER BY m.order",
                        {"id": int(graph_id)}
                    )
                    result["messages"] = messages
            
            enhanced_results.append(result)
        
        return enhanced_results
    
    def query_knowledge_graph(self, cypher_query: str, params: Dict = None) -> List[Dict]:
        """Query the knowledge graph directly with Cypher.
        
        Args:
            cypher_query: Cypher query string
            params: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        if not self.initialized and not self.initialize():
            return []
        
        return self.graph_db.query(cypher_query, params)
    
    def categorize_memories(self, query: str) -> Dict[str, List[Dict]]:
        """Categorize memories by type based on a query.
        
        Args:
            query: Query text to find and categorize memories
            
        Returns:
            Dictionary of memory categories with their results
        """
        if not self.initialized and not self.initialize():
            return {"facts": [], "conversations": []}
        
        # Get all relevant memories
        memories = self.retrieve_relevant_memories(query, limit=10)
        
        # Categorize by type
        categorized = {
            "facts": [],
            "conversations": []
        }
        
        for memory in memories:
            memory_type = memory.get("metadata", {}).get("type")
            if memory_type == "fact":
                categorized["facts"].append(memory)
            elif memory_type == "conversation":
                categorized["conversations"].append(memory)
        
        return categorized
    
    def get_entity_information(self, entity_name: str) -> Dict:
        """Get information about a specific entity from the knowledge graph.
        
        Args:
            entity_name: Name of the entity to retrieve
            
        Returns:
            Dictionary with entity information and related facts
        """
        if not self.initialized and not self.initialize():
            return {"entity": None, "related_facts": []}
        
        # Find the entity in the graph database
        entity_query = """
        MATCH (e) 
        WHERE e.name = $name
        RETURN e
        """
        
        entity_result = self.graph_db.query(entity_query, {"name": entity_name})
        
        if not entity_result:
            return {"entity": None, "related_facts": []}
        
        entity = entity_result[0]["e"]
        
        # Find facts related to this entity
        facts_query = """
        MATCH (f:Fact)-[:MENTIONS]->(e)
        WHERE e.name = $name
        RETURN f
        """
        
        facts_result = self.graph_db.query(facts_query, {"name": entity_name})
        related_facts = [record["f"] for record in facts_result]
        
        return {
            "entity": entity,
            "related_facts": related_facts
        }
    
    def clear_memory(self, memory_type: str = None) -> bool:
        """Clear memories of a specific type or all memories.
        
        Args:
            memory_type: Type of memory to clear ("fact", "conversation", or None for all)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized and not self.initialize():
            return False
        
        try:
            # Clear from vector database
            if memory_type:
                # Delete only specific type
                self.vector_db.delete(where={"type": memory_type})
            else:
                # Delete all
                self.vector_db.delete(where={})
            
            # Clear from graph database
            if memory_type == "fact":
                self.graph_db.query("MATCH (f:Fact) DETACH DELETE f")
            elif memory_type == "conversation":
                self.graph_db.query("MATCH (c:Conversation) DETACH DELETE c")
            elif memory_type is None:
                self.graph_db.query("MATCH (n) DETACH DELETE n")
            
            return True
        except Exception as e:
            print(f"Error clearing memory: {e}")
            return False
            
    # Additional retrieval methods
    
    def retrieve_by_entity_type(self, entity_type: str, limit: int = 10) -> List[Dict]:
        """Retrieve entities of a specific type from the knowledge graph.
        
        Args:
            entity_type: Type of entity to retrieve (Person, Organization, etc.)
            limit: Maximum number of results to return
            
        Returns:
            List of entities with the specified type
        """
        return MemoryRetrieval.retrieve_by_entity_type(self, entity_type, limit)
    
    def retrieve_entity_relationships(self, entity_name: str, relationship_type: Optional[str] = None) -> List[Dict]:
        """Retrieve relationships for a specific entity.
        
        Args:
            entity_name: Name of the entity
            relationship_type: Optional type of relationship to filter by
            
        Returns:
            List of relationships with connected entities
        """
        return MemoryRetrieval.retrieve_entity_relationships(self, entity_name, relationship_type)
    
    def retrieve_recent_memories(self, memory_type: Optional[str] = None, days: int = 7, limit: int = 10) -> List[Dict]:
        """Retrieve recent memories of a specific type.
        
        Args:
            memory_type: Type of memory to retrieve ("fact", "conversation", or None for all)
            days: Number of days to look back
            limit: Maximum number of results to return
            
        Returns:
            List of recent memories
        """
        return MemoryRetrieval.retrieve_recent_memories(self, memory_type, days, limit)
    
    def retrieve_memories_by_source(self, source: str, limit: int = 10) -> List[Dict]:
        """Retrieve memories from a specific source.
        
        Args:
            source: Source of the memories to retrieve
            limit: Maximum number of results to return
            
        Returns:
            List of memories from the specified source
        """
        return MemoryRetrieval.retrieve_memories_by_source(self, source, limit)
    
    def retrieve_related_facts(self, fact_id: Union[str, int], limit: int = 10) -> List[Dict]:
        """Retrieve facts related to a specific fact.
        
        Args:
            fact_id: ID of the fact to find related facts for
            limit: Maximum number of results to return
            
        Returns:
            List of related facts
        """
        return MemoryRetrieval.retrieve_related_facts(self, fact_id, limit)