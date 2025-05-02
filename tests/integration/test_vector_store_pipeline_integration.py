"""
Integration tests for the vector store module with the pipeline orchestrator.
Tests how the vector store integrates with the pipeline for document indexing.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import os
import tempfile
import shutil
import sys

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

# Mock magic
sys.modules['magic'] = MagicMock()

from core.config import UnifiedConfig, VectorStoreConfig, EmbedderConfig
from indexing.vector_store import ChromaVectorStoreAdapter, QdrantVectorStoreAdapter, VectorStoreFactory, QDRANT_AVAILABLE
from pipeline.orchestrator import PipelineOrchestrator
from llama_index.core.schema import Document, TextNode
from llama_index.core import Settings
from llama_index.core.embeddings.base import BaseEmbedding

# Create a mock embedding model for testing
class MockEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> list:
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def _get_text_embedding(self, text: str) -> list:
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def _get_text_embeddings(self, texts: list) -> list:
        return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]


class TestVectorStorePipelineIntegration(unittest.TestCase):
    """Test the integration between the vector store and pipeline orchestrator."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment for all tests in this class."""
        # Set up a mock embedding model to avoid OpenAI API calls
        Settings.embed_model = MockEmbedding()
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_db_path = os.path.join(self.temp_dir, "vector_db")
        self.qdrant_path = os.path.join(self.temp_dir, "qdrant_db")
        
        # Create configuration
        self.config = UnifiedConfig(
            input_directory=self.temp_dir,
            vector_store=VectorStoreConfig(
                engine="chroma",
                vector_db_path=self.vector_db_path,
                collection_name="test_collection"
            ),
            embedder=EmbedderConfig(
                provider="huggingface",
                model_name="test-model",
                use_cache=False
            )
        )
        
        # Mock components
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.create_index.return_value = MagicMock()
        
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed_nodes.return_value = [
            TextNode(text="Embedded node 1", embedding=[0.1, 0.2, 0.3]),
            TextNode(text="Embedded node 2", embedding=[0.4, 0.5, 0.6])
        ]
        
        # Create patch for factory
        self.vector_store_factory_patcher = patch('indexing.vector_store.VectorStoreFactory.create_vector_store')
        self.mock_vector_store_factory = self.vector_store_factory_patcher.start()
        self.mock_vector_store_factory.return_value = self.mock_vector_store
        
        # Create patch for embedder factory
        self.embedder_factory_patcher = patch('embedders.embedder_factory.EmbedderFactory.create_embedder')
        self.mock_embedder_factory = self.embedder_factory_patcher.start()
        self.mock_embedder_factory.return_value = self.mock_embedder
        
        # Create patch for document loading
        self.load_documents_patcher = patch.object(PipelineOrchestrator, '_load_documents')
        self.mock_load_documents = self.load_documents_patcher.start()
        self.mock_load_documents.return_value = (
            [Document(text="Test document", metadata={"source": "test.txt"})],
            0
        )
        
        # Create patch for document processing
        self.process_document_groups_patcher = patch.object(PipelineOrchestrator, '_process_document_groups')
        self.mock_process_document_groups = self.process_document_groups_patcher.start()
        self.mock_process_document_groups.return_value = [
            TextNode(text="Processed node 1"),
            TextNode(text="Processed node 2")
        ]
    
    def tearDown(self):
        """Clean up after the test."""
        self.vector_store_factory_patcher.stop()
        self.embedder_factory_patcher.stop()
        self.load_documents_patcher.stop()
        self.process_document_groups_patcher.stop()
        shutil.rmtree(self.temp_dir)
    
    def test_orchestrator_initializes_vector_store(self):
        """Test that the orchestrator initializes the vector store."""
        # Act
        orchestrator = PipelineOrchestrator(self.config)
        
        # Assert
        self.mock_vector_store_factory.assert_called_once_with(self.config.vector_store)
        self.assertEqual(orchestrator.vector_store, self.mock_vector_store)
    
    def test_orchestrator_indexes_nodes_after_embedding(self):
        """Test that the orchestrator indexes nodes after embedding."""
        # Arrange
        orchestrator = PipelineOrchestrator(self.config)
        
        # Act
        nodes = orchestrator.run()
        
        # Assert
        self.mock_embedder.embed_nodes.assert_called_once()
        self.mock_vector_store.create_index.assert_called_once()
        self.mock_vector_store.persist.assert_called_once()
        
        # Verify the nodes passed to create_index are the embedded nodes
        create_index_args = self.mock_vector_store.create_index.call_args[0]
        self.assertEqual(len(create_index_args[0]), 2)
        self.assertEqual(create_index_args[0][0].text, "Embedded node 1")
        self.assertEqual(create_index_args[0][1].text, "Embedded node 2")
    
    def test_orchestrator_skips_indexing_without_embedder(self):
        """Test that the orchestrator skips indexing when embedder is not available."""
        # Arrange
        self.mock_embedder_factory.side_effect = ImportError("Embedder not available")
        
        # Act
        orchestrator = PipelineOrchestrator(self.config)
        nodes = orchestrator.run()
        
        # Assert
        self.assertIsNone(orchestrator.embedder)
        self.assertIsNone(orchestrator.vector_store)
        self.mock_vector_store.create_index.assert_not_called()
        self.mock_vector_store.persist.assert_not_called()
    
    def test_orchestrator_handles_indexing_errors(self):
        """Test that the orchestrator handles errors during indexing."""
        # Arrange
        orchestrator = PipelineOrchestrator(self.config)
        self.mock_vector_store.create_index.side_effect = Exception("Indexing error")
        
        # Act
        nodes = orchestrator.run()
        
        # Assert
        self.mock_embedder.embed_nodes.assert_called_once()
        self.mock_vector_store.create_index.assert_called_once()
        self.mock_vector_store.persist.assert_not_called()
        
        # Verify that we still get the processed nodes despite the indexing error
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].text, "Embedded node 1")
        self.assertEqual(nodes[1].text, "Embedded node 2")
    
    def test_orchestrator_skips_indexing_without_embeddings(self):
        """Test that the orchestrator skips indexing when nodes have no embeddings."""
        # Arrange
        orchestrator = PipelineOrchestrator(self.config)
        self.mock_embedder.embed_nodes.return_value = [
            TextNode(text="Node without embedding 1"),
            TextNode(text="Node without embedding 2")
        ]
        
        # Act
        nodes = orchestrator.run()
        
        # Assert
        self.mock_embedder.embed_nodes.assert_called_once()
        self.mock_vector_store.create_index.assert_not_called()
        self.mock_vector_store.persist.assert_not_called()


class TestVectorStoreEndToEndIntegration(unittest.TestCase):
    """End-to-end tests for the vector store integration (optional)."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment for all tests in this class."""
        # Set up a mock embedding model to avoid OpenAI API calls
        Settings.embed_model = MockEmbedding()
    
    def setUp(self):
        """Set up the test environment with real temporary files."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.vector_db_path = os.path.join(self.test_dir, "vector_db")
        
        # Create test files
        self.text_file = os.path.join(self.test_dir, "test.txt")
        with open(self.text_file, "w") as f:
            f.write("This is a test document.")
        
        # Create a configuration for the orchestrator
        self.config = UnifiedConfig(
            input_directory=self.test_dir,
            vector_store=VectorStoreConfig(
                engine="chroma",
                vector_db_path=self.vector_db_path,
                collection_name="test_collection"
            )
        )
        
        # Mock the embedder to avoid actual embedding
        self.embedder_patcher = patch('embedders.embedder_factory.EmbedderFactory.create_embedder')
        self.mock_embedder_factory = self.embedder_patcher.start()
        self.mock_embedder = MagicMock()
        
        # Make the mock embedder return nodes with embeddings
        def embed_nodes(nodes):
            for i, node in enumerate(nodes):
                node.embedding = [float(i) / 10] * 10
            return nodes
            
        self.mock_embedder.embed_nodes.side_effect = embed_nodes
        self.mock_embedder_factory.return_value = self.mock_embedder
    
    def tearDown(self):
        """Clean up the test environment."""
        self.embedder_patcher.stop()
        shutil.rmtree(self.test_dir)
    
    @unittest.skip("This is an end-to-end test that should be run manually")
    def test_end_to_end_vector_store(self):
        """Test the end-to-end vector store integration with real files."""
        # Create an orchestrator with the test configuration
        orchestrator = PipelineOrchestrator(self.config)
        
        # Run the pipeline
        nodes = orchestrator.run()
        
        # Verify that we got nodes
        self.assertGreater(len(nodes), 0)
        
        # Verify that the vector store was created
        self.assertTrue(os.path.exists(self.vector_db_path))
        
        # Verify that the nodes have embeddings
        for node in nodes:
            self.assertIsNotNone(node.embedding)
            self.assertGreater(len(node.embedding), 0)


if __name__ == '__main__':
    unittest.main()
