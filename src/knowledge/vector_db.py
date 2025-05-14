"""ChromaDB vector database integration for OpenManus.

This module provides a VectorDatabase class for interacting with ChromaDB,
allowing the agent to store and retrieve embeddings for semantic search.
"""

import os
from typing import Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

class VectorDatabase:
    """ChromaDB vector database integration for OpenManus.
    
    This class provides methods for connecting to ChromaDB and performing
    vector operations like storing, retrieving, and searching embeddings.
    """
    
    def __init__(self, persist_directory: str = None, collection_name: str = "openmanus"):
        """Initialize the ChromaDB connection.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_function = None
        
    def connect(self) -> bool:
        """Connect to the ChromaDB database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Initialize client with persistence
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory
            ))
            
            # Initialize sentence transformer embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"  # Default model, can be configured
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            return True
        except Exception as e:
            print(f"Failed to connect to ChromaDB: {e}")
            return False
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None) -> bool:
        """Add documents to the vector database.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries for each document
            ids: List of IDs for each document
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client or not self.collection:
            if not self.connect():
                return False
        
        try:
            # Generate IDs if not provided
            if not ids:
                import uuid
                ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            
            # Ensure metadatas is provided for each document
            if not metadatas:
                metadatas = [{} for _ in range(len(documents))]
            
            # Add documents to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def query(self, query_text: str, n_results: int = 5, 
              filter_metadata: Dict = None) -> List[Dict]:
        """Query the vector database for similar documents.
        
        Args:
            query_text: Text to query
            n_results: Number of results to return
            filter_metadata: Filter results by metadata
            
        Returns:
            List of results with documents, distances, and metadatas
        """
        if not self.client or not self.collection:
            if not self.connect():
                return []
        
        try:
            # Query collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error querying documents: {e}")
            return []
    
    def delete(self, ids: List[str] = None, filter_metadata: Dict = None) -> bool:
        """Delete documents from the vector database.
        
        Args:
            ids: List of document IDs to delete
            filter_metadata: Delete documents matching this metadata filter
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client or not self.collection:
            if not self.connect():
                return False
        
        try:
            if ids:
                self.collection.delete(ids=ids)
            elif filter_metadata:
                self.collection.delete(where=filter_metadata)
            else:
                return False  # Must provide either ids or filter_metadata
            
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False