"""
Unit tests for the configuration system.

These tests verify that the configuration system works correctly, including:
- Loading configuration from multiple sources
- Validation of configuration values
- Ensuring consistency between indexing and querying embedders
- Environment-specific configuration
"""

import os
import sys
import json
import yaml
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import configuration modules
from core.config_manager import ConfigManager
from core.config import (
    UnifiedConfig, EmbedderConfig, VectorStoreConfig,
    LLMConfig, ApplicationEnvironment
)


class TestConfigModels(unittest.TestCase):
    """Test the Pydantic configuration models."""
    
    def test_unified_config_validation(self):
        """Test validation of UnifiedConfig."""
        # Missing required parameter
        with self.assertRaises(ValueError):
            UnifiedConfig()
        
        # Valid configuration
        config = UnifiedConfig(input_directory="./data")
        self.assertEqual(config.input_directory, "./data")
        self.assertEqual(config.project_name, "unified-parser")
        self.assertEqual(config.environment, ApplicationEnvironment.DEVELOPMENT)
    
    def test_embedder_config_validation(self):
        """Test validation of EmbedderConfig."""
        # Default configuration
        config = EmbedderConfig()
        self.assertEqual(config.provider, "huggingface")
        self.assertEqual(config.model_name, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embed_batch_size, 10)
        
        # Invalid batch size
        with self.assertRaises(ValueError):
            EmbedderConfig(embed_batch_size=0)
    
    def test_vector_store_config_validation(self):
        """Test validation of VectorStoreConfig."""
        # Default configuration
        config = VectorStoreConfig()
        self.assertEqual(config.engine, "chroma")
        self.assertEqual(config.collection_name, "unified_knowledge")
        
        # Invalid engine
        with self.assertRaises(ValueError):
            VectorStoreConfig(engine="invalid_engine")
        
        # Invalid Qdrant location
        with self.assertRaises(ValueError):
            VectorStoreConfig(engine="qdrant", qdrant_location="invalid_location")
        
        # Missing Qdrant cloud URL
        with self.assertRaises(ValueError):
            VectorStoreConfig(engine="qdrant", qdrant_location="cloud")
    
    def test_llm_config_api_key(self):
        """Test LLMSettings API key handling."""
        # Set up test environment variable
        os.environ["TEST_API_KEY"] = "test_key"
        
        # Create LLM configuration
        from core.config import LLMSettings
        llm_settings = LLMSettings(api_key_env_var="TEST_API_KEY")
        
        # Check API key extraction
        self.assertEqual(llm_settings.api_key, "test_key")
        
        # Clean up
        del os.environ["TEST_API_KEY"]


class TestConfigManager(unittest.TestCase):
    """Test the configuration manager."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary configuration files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test_data directory within temp_dir for input validation
        self.test_data_dir = os.path.join(self.temp_dir.name, "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create main configuration file
        self.config_path = os.path.join(self.temp_dir.name, "config.yaml")
        with open(self.config_path, "w") as f:
            yaml.dump({
                "project_name": "test-project",
                "input_directory": self.test_data_dir,
                "embedder": {
                    "provider": "huggingface",
                    "model_name": "test-model"
                }
            }, f)
        
        # Create environment-specific configuration file
        self.env_config_path = os.path.join(self.temp_dir.name, "config.production.yaml")
        with open(self.env_config_path, "w") as f:
            yaml.dump({
                "project_name": "test-project-prod",
                "input_directory": self.test_data_dir,
                "embedder": {
                    "provider": "openai",
                    "model_name": "text-embedding-3-small"
                }
            }, f)
        
        # Create environment file
        self.env_file = os.path.join(self.temp_dir.name, ".env")
        # Set environment variables for testing
        os.environ["RAG_EMBEDDER__PROVIDER"] = "cohere"
        os.environ["RAG_EMBEDDER__MODEL_NAME"] = "embed-english-v3.0"
        os.environ["RAG_INPUT_DIRECTORY"] = self.test_data_dir
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_config_loading(self):
        """Test loading configuration from file."""
        config_manager = ConfigManager(
            config_path=self.config_path,
            env_file=self.env_file
        )
        
        # Load config without overrides
        config = config_manager.get_config(UnifiedConfig)
        
        # Check values from file
        self.assertEqual(config.project_name, "test-project")
        self.assertEqual(config.input_directory, self.test_data_dir)
        
        # Check values from environment variables (highest precedence)
        self.assertEqual(config.embedder.provider, "cohere")
        self.assertEqual(config.embedder.model_name, "embed-english-v3.0")
    
    def test_environment_specific_config(self):
        """Test environment-specific configuration."""
        # Set environment to production
        config_manager = ConfigManager(
            config_path=self.config_path,
            env_file=self.env_file,
            environment="production"
        )
        
        # Load config without overrides
        config = config_manager.get_config(UnifiedConfig)
        
        # Check values from production config
        self.assertEqual(config.project_name, "test-project-prod")
        
        # Check values from environment variables (highest precedence)
        self.assertEqual(config.embedder.provider, "cohere")
        self.assertEqual(config.embedder.model_name, "embed-english-v3.0")
    
    def test_config_overrides(self):
        """Test configuration overrides."""
        config_manager = ConfigManager(
            config_path=self.config_path,
            env_file=self.env_file
        )
        
        # Load config with overrides
        config = config_manager.get_config(
            UnifiedConfig,
            overrides={
                "project_name": "override-project",
                "input_directory": "./data",  # Ensure input_directory is provided
                "embedder": {
                    "provider": "override-provider",
                    "model_name": "override-model"
                }
            }
        )
        
        # Check overridden values
        self.assertEqual(config.project_name, "override-project")
        self.assertEqual(config.embedder.provider, "override-provider")
        self.assertEqual(config.embedder.model_name, "override-model")
    
    def test_unified_config_embedding_consistency(self):
        """Test ensuring embedding consistency in unified config."""
        config_manager = ConfigManager(
            config_path=self.config_path,
            env_file=self.env_file
        )
        
        # Mock retrieval config with embedder settings
        mock_retrieval_config = MagicMock()
        mock_retrieval_config.retriever_strategy = "vector"
        mock_retrieval_config.embedder_provider = "different-provider"
        mock_retrieval_config.embedder_model = "different-model"
        
        # Create mock query pipeline config
        mock_query_pipeline_config = MagicMock()
        mock_query_pipeline_config.retrieval = mock_retrieval_config
        
        # Create mock unified config
        mock_unified_config = MagicMock()
        mock_unified_config.embedder.provider = "test-provider"
        mock_unified_config.embedder.model_name = "test-model"
        mock_unified_config.vector_store.engine = "chroma"
        mock_unified_config.query_pipeline = mock_query_pipeline_config
        
        # Mock the get_config method to return our mock config
        with patch.object(config_manager, 'get_config', return_value=mock_unified_config):
            # Get unified config with embedding consistency
            config = config_manager.get_unified_config(ensure_embedder_consistency=True)
            
            # Verify that embedding settings were made consistent
            self.assertEqual(
                mock_retrieval_config.embedder_provider, 
                mock_unified_config.embedder.provider
            )
            self.assertEqual(
                mock_retrieval_config.embedder_model, 
                mock_unified_config.embedder.model_name
            )


if __name__ == "__main__":
    unittest.main()