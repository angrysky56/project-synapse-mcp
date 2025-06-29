"""
Mock Knowledge Graph for testing without Neo4j connection issues.
"""

from typing import Dict, Any
import json
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MockKnowledgeGraph:
    """
    Mock knowledge graph that logs data instead of storing to Neo4j.
    
    This allows us to test the complete semantic pipeline without 
    Neo4j authentication issues.
    """

    def __init__(self):
        self.connected = False

    async def connect(self) -> None:
        """Mock connection that always succeeds."""
        logger.info("Mock Neo4j connection established")
        self.connected = True

    async def close(self) -> None:
        """Mock close operation."""
        logger.info("Mock Neo4j connection closed")
        self.connected = False

    async def store_processed_data(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock storage that logs the data instead of storing to Neo4j.
        
        Args:
            processed_data: dictionary containing entities, relationships, and facts
            
        Returns:
            dictionary with mock storage statistics
        """
        logger.info("Mock storing processed data:")
        logger.info(f"  Text ID: {processed_data.get('text_id', 'N/A')}")
        logger.info(f"  Source: {processed_data.get('source', 'N/A')}")
        logger.info(f"  Entities: {len(processed_data.get('entities', []))}")
        logger.info(f"  Relationships: {len(processed_data.get('relationships', []))}")
        logger.info(f"  Facts: {len(processed_data.get('facts', []))}")
        
        # Log sample data
        if processed_data.get('entities'):
            logger.info(f"  Sample entity: {processed_data['entities'][0]['name']}")
        if processed_data.get('facts'):
            logger.info(f"  Sample fact: {processed_data['facts'][0]['content'][:50]}...")
            
        # Return mock statistics
        return {
            'entities_count': len(processed_data.get('entities', [])),
            'relationships_count': len(processed_data.get('relationships', [])),
            'facts_count': len(processed_data.get('facts', [])),
            'new_nodes': len(processed_data.get('entities', [])) + len(processed_data.get('facts', [])),
            'new_edges': len(processed_data.get('relationships', []))
        }

    async def query_semantic(self, query: str, max_results: int = 10) -> list[dict]:
        """Mock semantic query that returns empty results."""
        logger.info(f"Mock semantic query: {query}")
        return []
