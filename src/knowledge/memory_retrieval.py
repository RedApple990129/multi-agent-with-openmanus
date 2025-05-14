"""Memory retrieval utilities for OpenManus.

This module provides additional retrieval methods for the memory manager
to enhance knowledge management capabilities.
"""

from typing import Dict, List, Optional, Union
import datetime


class MemoryRetrieval:
    """Memory retrieval utilities for the MemoryManager.
    
    This class provides additional retrieval methods that can be used
    by the memory manager to enhance knowledge management capabilities.
    """
    
    @staticmethod
    def retrieve_by_entity_type(memory_manager, entity_type: str, limit: int = 10) -> List[Dict]:
        """Retrieve entities of a specific type from the knowledge graph.
        
        Args:
            memory_manager: The memory manager instance
            entity_type: Type of entity to retrieve (Person, Organization, etc.)
            limit: Maximum number of results to return
            
        Returns:
            List of entities with the specified type
        """
        if not memory_manager.initialized and not memory_manager.initialize():
            return []
        
        query = f"""
        MATCH (e:{entity_type})
        RETURN e
        LIMIT {limit}
        """
        
        return memory_manager.graph_db.query(query, {})
    
    @staticmethod
    def retrieve_entity_relationships(memory_manager, entity_name: str, relationship_type: Optional[str] = None) -> List[Dict]:
        """Retrieve relationships for a specific entity.
        
        Args:
            memory_manager: The memory manager instance
            entity_name: Name of the entity
            relationship_type: Optional type of relationship to filter by
            
        Returns:
            List of relationships with connected entities
        """
        if not memory_manager.initialized and not memory_manager.initialize():
            return []
        
        if relationship_type:
            query = f"""
            MATCH (e)-[r:{relationship_type}]-(connected)
            WHERE e.name = $name
            RETURN e, r, connected
            """
        else:
            query = """
            MATCH (e)-[r]-(connected)
            WHERE e.name = $name
            RETURN e, r, connected
            """
        
        return memory_manager.graph_db.query(query, {"name": entity_name})
    
    @staticmethod
    def retrieve_recent_memories(memory_manager, memory_type: Optional[str] = None, days: int = 7, limit: int = 10) -> List[Dict]:
        """Retrieve recent memories of a specific type.
        
        Args:
            memory_manager: The memory manager instance
            memory_type: Type of memory to retrieve ("fact", "conversation", or None for all)
            days: Number of days to look back
            limit: Maximum number of results to return
            
        Returns:
            List of recent memories
        """
        if not memory_manager.initialized and not memory_manager.initialize():
            return []
        
        # Calculate date threshold
        threshold_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
        
        # Query vector database for recent memories
        where_clause = {"timestamp": {"$gt": threshold_date}}
        if memory_type:
            where_clause["type"] = memory_type
        
        return memory_manager.vector_db.query(
            query="",  # Empty query to match based on metadata only
            where=where_clause,
            n_results=limit
        )
    
    @staticmethod
    def retrieve_memories_by_source(memory_manager, source: str, limit: int = 10) -> List[Dict]:
        """Retrieve memories from a specific source.
        
        Args:
            memory_manager: The memory manager instance
            source: Source of the memories to retrieve
            limit: Maximum number of results to return
            
        Returns:
            List of memories from the specified source
        """
        if not memory_manager.initialized and not memory_manager.initialize():
            return []
        
        return memory_manager.vector_db.query(
            query="",  # Empty query to match based on metadata only
            where={"source": source},
            n_results=limit
        )
    
    @staticmethod
    def retrieve_related_facts(memory_manager, fact_id: Union[str, int], limit: int = 10) -> List[Dict]:
        """Retrieve facts related to a specific fact.
        
        Args:
            memory_manager: The memory manager instance
            fact_id: ID of the fact to find related facts for
            limit: Maximum number of results to return
            
        Returns:
            List of related facts
        """
        if not memory_manager.initialized and not memory_manager.initialize():
            return []
        
        # Get the fact content
        fact_node = memory_manager.graph_db.query(
            "MATCH (f:Fact) WHERE id(f) = $id RETURN f",
            {"id": int(fact_id) if isinstance(fact_id, str) else fact_id}
        )
        
        if not fact_node:
            return []
        
        fact_content = fact_node[0]["f"].get("content", "")
        
        # Find semantically similar facts
        return memory_manager.vector_db.query(
            query=fact_content,
            where={"type": "fact"},
            n_results=limit + 1  # Add 1 to account for the fact itself
        )[1:]  # Exclude the first result (the fact itself)