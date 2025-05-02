"""
Unit tests for the vector store module.
Tests the vector store adapters and factory implementation.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
import shutil

# Mock the required modules before importing
# Create a more complete mock for chromadb
chromadb_mock = MagicMock()
chromadb_mock.PersistentClient = MagicMock()
chromadb_mock.api = MagicMock()
chromadb_mock.api.models = MagicMock()
chromadb_mock.api.models.Collection = MagicMock()
sys.modules['chromadb'] = chromadb_mock
sys.modules['chromadb.api'] = chromadb_mock.api
sys.modules['chromadb.api.models'] = chromadb_mock.api.models
sys.modules['chromadb.api.models.Collection'] = chromadb_mock.api.models.Collection

# Mock qdrant_client
qdrant_mock = MagicMock()
qdrant_mock.http = MagicMock()
qdrant_mock.http.models = MagicMock()
sys.modules['qdrant_client'] = qdrant_mock
sys.modules['qdrant_client.http'] = qdrant_mock.http
sys.modules['qdrant_client.http.models'] = qdrant_mock.http.models

from core.config import VectorStoreConfig
from core.interfaces import IVectorStore
from indexing.vector_store import (
    ChromaVectorStoreAdapter,
    QdrantVectorStoreAdapter,
    VectorStoreFactory,
    QDRANT_AVAILABLE
)
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.embeddings.base import BaseEmbedding


# Create a mock embedding model for testing
class MockEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> list:
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def _get_text_embedding(self, text: str) -> list:
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def _get_text_embeddings(self, texts: list) -> list:
        return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]


class TestChromaVectorStoreAdapter(unittest.TestCase):
    """Test the ChromaVectorStoreAdapter class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment for all tests in this class."""
        # Set up a mock embedding model to avoid OpenAI API calls
        Settings.embed_model = MockEmbedding()
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VectorStoreConfig(
            engine="chroma",
            vector_db_path=self.temp_dir,
            collection_name="test_collection"
        )
        
        # Mock ChromaDB components
        self.mock_collection = MagicMock()
        self.mock_db = MagicMock()
        self.mock_db.get_or_create_collection.return_value = self.mock_collection
        
        # Mock VectorStoreIndex
        self.mock_index = MagicMock(spec=VectorStoreIndex)
        
        # Patch the necessary components
        self.chroma_client_patcher = patch('chromadb.PersistentClient')
        self.mock_chroma_client = self.chroma_client_patcher.start()
        self.mock_chroma_client.return_value = self.mock_db
        
        self.vector_store_patcher = patch('llama_index.vector_stores.chroma.ChromaVectorStore')
        self.mock_vector_store = self.vector_store_patcher.start()
        
        self.storage_context_patcher = patch('llama_index.core.StorageContext.from_defaults')
        self.mock_storage_context = self.storage_context_patcher.start()
        
        self.vector_store_index_patcher = patch('llama_index.core.VectorStoreIndex')
        self.mock_vector_store_index = self.vector_store_index_patcher.start()
        self.mock_vector_store_index.return_value = self.mock_index
        
        self.vector_store_index_from_patcher = patch('llama_index.core.VectorStoreIndex.from_vector_store')
        self.mock_vector_store_index_from = self.vector_store_index_from_patcher.start()
        self.mock_vector_store_index_from.return_value = self.mock_index
    
    def tearDown(self):
        """Clean up after the test."""
        self.chroma_client_patcher.stop()
        self.vector_store_patcher.stop()
        self.storage_context_patcher.stop()
        self.vector_store_index_patcher.stop()
        self.vector_store_index_from_patcher.stop()
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization of ChromaVectorStoreAdapter."""
        # Act
        adapter = ChromaVectorStoreAdapter(self.config)
        
        # Assert
        self.mock_chroma_client.assert_called_once_with(path=self.temp_dir)
        self.mock_db.get_or_create_collection.assert_called_once()
        self.assertEqual(adapter.config, self.config)
        self.assertIsNone(adapter.index)
    
    def test_create_index(self):
        """Test creating an index from nodes."""
        # Arrange
        adapter = ChromaVectorStoreAdapter(self.config)
        nodes = [
            TextNode(text="Test node 1"),
            TextNode(text="Test node 2")
        ]
        
        # Act
        result = adapter.create_index(nodes)
        
        # Assert
        self.mock_vector_store_index.assert_called_once()
        self.assertEqual(result, self.mock_index)
        self.assertEqual(adapter.index, self.mock_index)
    
    def test_persist(self):
        """Test persisting the index."""
        # Arrange
        adapter = ChromaVectorStoreAdapter(self.config)
        
        # Act
        adapter.persist()
        
        # Assert - ChromaDB auto-persists, so no additional calls should be made
        # This is just testing that the method doesn't raise exceptions
    
    def test_load(self):
        """Test loading the index from storage."""
        # Arrange
        adapter = ChromaVectorStoreAdapter(self.config)
        
        # Act
        result = adapter.load()
        
        # Assert
        self.mock_chroma_client.assert_called()
        self.mock_vector_store_index_from.assert_called_once()
        self.assertEqual(result, self.mock_index)
        self.assertEqual(adapter.index, self.mock_index)
    
    def test_load_with_custom_path(self):
        """Test loading the index from a custom path."""
        # Arrange
        adapter = ChromaVectorStoreAdapter(self.config)
        custom_path = os.path.join(self.temp_dir, "custom")
        
        # Act
        result = adapter.load(custom_path)
        
        # Assert
        self.mock_chroma_client.assert_called_with(path=custom_path)
        self.mock_vector_store_index_from.assert_called_once()
        self.assertEqual(result, self.mock_index)
        self.assertEqual(adapter.index, self.mock_index)


@unittest.skipIf(not QDRANT_AVAILABLE, "Qdrant client not available")
class TestQdrantVectorStoreAdapter(unittest.TestCase):
    """Test the QdrantVectorStoreAdapter class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment for all tests in this class."""
        # Set up a mock embedding model to avoid OpenAI API calls
        Settings.embed_model = MockEmbedding()
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VectorStoreConfig(
            engine="qdrant",
            collection_name="test_collection",
            qdrant_location="local",
            qdrant_local_path=self.temp_dir
        )
        
        # Mock Qdrant components
        self.mock_client = MagicMock()
        self.mock_collections = MagicMock()
        self.mock_collections.collections = []
        self.mock_client.get_collections.return_value = self.mock_collections
        
        # Mock VectorStoreIndex
        self.mock_index = MagicMock(spec=VectorStoreIndex)
        
        # Patch the necessary components
        self.qdrant_client_patcher = patch('qdrant_client.QdrantClient')
        self.mock_qdrant_client = self.qdrant_client_patcher.start()
        self.mock_qdrant_client.return_value = self.mock_client
        
        self.vector_store_patcher = patch('llama_index.vector_stores.qdrant.QdrantVectorStore')
        self.mock_vector_store = self.vector_store_patcher.start()
        
        self.storage_context_patcher = patch('llama_index.core.StorageContext.from_defaults')
        self.mock_storage_context = self.storage_context_patcher.start()
        
        self.vector_store_index_patcher = patch('llama_index.core.VectorStoreIndex')
        self.mock_vector_store_index = self.vector_store_index_patcher.start()
        self.mock_vector_store_index.return_value = self.mock_index
        
        self.vector_store_index_from_patcher = patch('llama_index.core.VectorStoreIndex.from_vector_store')
        self.mock_vector_store_index_from = self.vector_store_index_from_patcher.start()
        self.mock_vector_store_index_from.return_value = self.mock_index
    
    def tearDown(self):
        """Clean up after the test."""
        self.qdrant_client_patcher.stop()
        self.vector_store_patcher.stop()
        self.storage_context_patcher.stop()
        self.vector_store_index_patcher.stop()
        self.vector_store_index_from_patcher.stop()
        shutil.rmtree(self.temp_dir)
    
    def test_init_local(self):
        """Test initialization of QdrantVectorStoreAdapter with local storage."""
        # Act
        adapter = QdrantVectorStoreAdapter(self.config)
        
        # Assert
        self.mock_qdrant_client.assert_called_once_with(
            path=self.temp_dir,
            timeout=self.config.qdrant_timeout,
            prefer_grpc=self.config.qdrant_prefer_grpc
        )
        self.mock_client.get_collections.assert_called_once()
        self.assertEqual(adapter.config, self.config)
        self.assertIsNone(adapter.index)
    
    def test_init_cloud(self):
        """Test initialization of QdrantVectorStoreAdapter with cloud storage."""
        # Arrange
        cloud_config = VectorStoreConfig(
            engine="qdrant",
            collection_name="test_collection",
            qdrant_location="cloud",
            qdrant_url="https://test-url.qdrant.io",
            qdrant_api_key="test_api_key"
        )
        
        # Act
        adapter = QdrantVectorStoreAdapter(cloud_config)
        
        # Assert
        self.mock_qdrant_client.assert_called_once_with(
            url=cloud_config.qdrant_url,
            api_key=cloud_config.qdrant_api_key,
            timeout=cloud_config.qdrant_timeout,
            prefer_grpc=cloud_config.qdrant_prefer_grpc
        )
        self.mock_client.get_collections.assert_called_once()
        self.assertEqual(adapter.config, cloud_config)
        self.assertIsNone(adapter.index)
    
    def test_create_index(self):
        """Test creating an index from nodes."""
        # Arrange
        adapter = QdrantVectorStoreAdapter(self.config)
        nodes = [
            TextNode(text="Test node 1"),
            TextNode(text="Test node 2")
        ]
        
        # Act
        result = adapter.create_index(nodes)
        
        # Assert
        self.mock_vector_store_index.assert_called_once()
        self.assertEqual(result, self.mock_index)
        self.assertEqual(adapter.index, self.mock_index)
    
    def test_persist(self):
        """Test persisting the index."""
        # Arrange
        adapter = QdrantVectorStoreAdapter(self.config)
        
        # Act
        adapter.persist()
        
        # Assert - Qdrant auto-persists, so no additional calls should be made
        # This is just testing that the method doesn't raise exceptions
    
    def test_load(self):
        """Test loading the index from storage."""
        # Arrange
        adapter = QdrantVectorStoreAdapter(self.config)
        
        # Act
        result = adapter.load()
        
        # Assert
        self.mock_vector_store_index_from.assert_called_once()
        self.assertEqual(result, self.mock_index)
        self.assertEqual(adapter.index, self.mock_index)
    
    def test_load_with_custom_path(self):
        """Test loading the index from a custom path."""
        # Arrange
        adapter = QdrantVectorStoreAdapter(self.config)
        custom_path = os.path.join(self.temp_dir, "custom")
        
        # Act
        result = adapter.load(custom_path)
        
        # Assert
        self.mock_qdrant_client.assert_called_with(
            path=custom_path,
            timeout=self.config.qdrant_timeout,
            prefer_grpc=self.config.qdrant_prefer_grpc
        )
        self.mock_vector_store_index_from.assert_called_once()
        self.assertEqual(result, self.mock_index)
        self.assertEqual(adapter.index, self.mock_index)


class TestVectorStoreFactory(unittest.TestCase):
    """Test the VectorStoreFactory class."""
    
    def test_create_vector_store_chroma(self):
        """Test creating a ChromaVectorStoreAdapter."""
        # Arrange
        config = VectorStoreConfig(engine="chroma")
        
        # Act
        vector_store = VectorStoreFactory.create_vector_store(config)
        
        # Assert
        self.assertIsInstance(vector_store, ChromaVectorStoreAdapter)
        self.assertIsInstance(vector_store, IVectorStore)
    
    @unittest.skipIf(not QDRANT_AVAILABLE, "Qdrant client not available")
    def test_create_vector_store_qdrant(self):
        """Test creating a QdrantVectorStoreAdapter."""
        # Arrange
        config = VectorStoreConfig(engine="qdrant")
        
        # Act
        vector_store = VectorStoreFactory.create_vector_store(config)
        
        # Assert
        self.assertIsInstance(vector_store, QdrantVectorStoreAdapter)
        self.assertIsInstance(vector_store, IVectorStore)
    
    def test_create_vector_store_unknown(self):
        """Test handling of unknown vector store engines."""
        # Arrange
        config = VectorStoreConfig()
        config.engine = "unknown"  # Override the default
        
        # Act
        vector_store = VectorStoreFactory.create_vector_store(config)
        
        # Assert - should fall back to ChromaDB
        self.assertIsInstance(vector_store, ChromaVectorStoreAdapter)
        self.assertIsInstance(vector_store, IVectorStore)
    
    @patch('indexing.vector_store.QDRANT_AVAILABLE', False)
    def test_create_vector_store_qdrant_not_available(self):
        """Test fallback when Qdrant is not available."""
        # Arrange
        config = VectorStoreConfig(engine="qdrant")
        
        # Act
        vector_store = VectorStoreFactory.create_vector_store(config)
        
        # Assert - should fall back to ChromaDB
        self.assertIsInstance(vector_store, ChromaVectorStoreAdapter)
        self.assertIsInstance(vector_store, IVectorStore)


if __name__ == '__main__':
    unittest.main()
