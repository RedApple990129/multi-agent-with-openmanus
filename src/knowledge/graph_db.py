"""Neo4j graph database integration for OpenManus.

This module provides a GraphDatabase class for interacting with Neo4j,
allowing the agent to store and retrieve structured knowledge.
"""

import os
from typing import Dict, List, Optional, Union

from neo4j import GraphDatabase as Neo4jDriver
from neo4j.exceptions import ServiceUnavailable

class GraphDatabase:
    """Neo4j graph database integration for OpenManus.
    
    This class provides methods for connecting to Neo4j and performing
    common operations like creating nodes, relationships, and querying.
    """
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """Initialize the Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (defaults to environment variable NEO4J_URI)
            username: Neo4j username (defaults to environment variable NEO4J_USERNAME)
            password: Neo4j password (defaults to environment variable NEO4J_PASSWORD)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        
    def connect(self) -> bool:
        """Connect to the Neo4j database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.driver = Neo4jDriver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except ServiceUnavailable as e:
            print(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def create_node(self, labels: List[str], properties: Dict) -> Optional[int]:
        """Create a node in the graph database.
        
        Args:
            labels: List of node labels
            properties: Dictionary of node properties
            
        Returns:
            Node ID if successful, None otherwise
        """
        if not self.driver:
            if not self.connect():
                return None
                
        labels_str = ':'.join(labels)
        properties_str = ', '.join([f"{k}: ${k}" for k in properties.keys()])
        
        query = f"CREATE (n:{labels_str} {{{properties_str}}}) RETURN id(n) as node_id"
        
        try:
            with self.driver.session() as session:
                result = session.run(query, **properties)
                record = result.single()
                return record["node_id"] if record else None
        except Exception as e:
            print(f"Error creating node: {e}")
            return None
    
    def create_relationship(self, start_node_id: int, end_node_id: int, 
                           relationship_type: str, properties: Dict = None) -> bool:
        """Create a relationship between two nodes.
        
        Args:
            start_node_id: ID of the start node
            end_node_id: ID of the end node
            relationship_type: Type of relationship
            properties: Dictionary of relationship properties
            
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            if not self.connect():
                return False
                
        properties = properties or {}
        properties_str = ', '.join([f"{k}: ${k}" for k in properties.keys()])
        
        query = f"""MATCH (a), (b) 
                  WHERE id(a) = $start_id AND id(b) = $end_id 
                  CREATE (a)-[r:{relationship_type} {{{properties_str}}}]->(b) 
                  RETURN type(r)"""
        
        try:
            with self.driver.session() as session:
                params = {"start_id": start_node_id, "end_id": end_node_id, **properties}
                result = session.run(query, **params)
                return result.single() is not None
        except Exception as e:
            print(f"Error creating relationship: {e}")
            return False
    
    def query(self, cypher_query: str, params: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results.
        
        Args:
            cypher_query: Cypher query string
            params: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        if not self.driver:
            if not self.connect():
                return []
                
        params = params or {}
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, **params)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error executing query: {e}")
            return []