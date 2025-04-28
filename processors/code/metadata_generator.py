# --- processors/code/metadata_generator.py ---

import datetime
from typing import List, Dict, Optional, Any
import logging
import os

from llama_index.core.schema import TextNode
from llama_index.core.llms import LLM

from core.interfaces import IMetadataEnricher
from core.config import LLMConfig

logger = logging.getLogger(__name__)


class CodeMetadataGenerator:
    """Generates metadata for code chunks using LLMs."""
    
    def __init__(self, llm: Optional[LLM] = None):
        """Initialize metadata generator with LLM.
        
        Args:
            llm: Optional LLM instance
        """
        self.llm = llm
    
    def supports_node_type(self, node_type: str) -> bool:
        """Check if the enricher supports a given node type.
        
        Args:
            node_type: The node type to check
            
        Returns:
            True if the node type is "code", False otherwise
        """
        return node_type.lower() == "code"
    
    def enrich(self, nodes: List[TextNode]) -> List[Dict]:
        """Enrich code nodes with metadata.
        
        Args:
            nodes: List of code nodes to enrich
            
        Returns:
            List of metadata dictionaries
        """
        # If no LLM is provided, just add basic metadata
        if self.llm is None:
            for node in nodes:
                self._add_basic_metadata(node)
            return [node.metadata for node in nodes]
        
        # If LLM is provided, try to use it for enrichment
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.aenrich(nodes))
    
    async def aenrich(self, nodes: List[TextNode]) -> List[Dict]:
        """Asynchronously enrich code nodes with metadata.
        
        Args:
            nodes: List of code nodes to enrich
            
        Returns:
            List of metadata dictionaries
        """
        # If no LLM is provided, just add basic metadata
        if self.llm is None:
            for node in nodes:
                self._add_basic_metadata(node)
            return [node.metadata for node in nodes]
        
        import asyncio
        tasks = [self._process_node(node) for node in nodes]
        await asyncio.gather(*tasks)
        return [node.metadata for node in nodes]
    
    def _add_basic_metadata(self, node: TextNode) -> None:
        """Add basic metadata to a node without using LLM.
        
        Args:
            node: The node to process
        """
        # Extract language from file extension if available
        file_path = node.metadata.get("file_path", "")
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            language = self._get_language_from_extension(ext)
            node.metadata["language"] = language
        else:
            node.metadata["language"] = "unknown"
        
        # Set default metadata
        node.metadata["title"] = os.path.basename(file_path) if file_path else "Code Snippet"
        node.metadata["description"] = f"Code snippet from {node.metadata.get('file_name', 'unknown source')}"
        
        # Create source info with version timestamp
        today = datetime.date.today().strftime("%Y-%m-%d")
        source_path = node.metadata.get("file_path", "unknown_source")
        node.metadata["source"] = f"{source_path}#{today}_snippet_{node.node_id[:8] if hasattr(node, 'node_id') else 'unknown'}"
        
        # Add node type
        node.metadata["node_type"] = "code"
        node.metadata["context7_format"] = True
    
    async def _process_node(self, node: TextNode) -> None:
        """Process a single node to add metadata using LLM.
        
        Args:
            node: The node to process
        """
        # First add basic metadata
        self._add_basic_metadata(node)
        
        # If we have an LLM, enhance with it
        if self.llm is None:
            return
        
        code = node.text
        
        # Generate title
        title_prompt = f"Generate a concise technical title for this code snippet that describes its functionality:\n{code}\nTitle:"
        try:
            title_response = await self.llm.acomplete(title_prompt)
            node.metadata["title"] = title_response.text.strip()
        except Exception as e:
            logger.error(f"Error generating title: {str(e)}")
            # Keep the basic title set earlier
        
        # Generate description
        desc_prompt = f"Write a detailed technical description of what this code does and its main purpose:\n{code}\nDescription:"
        try:
            desc_response = await self.llm.acomplete(desc_prompt)
            node.metadata["description"] = desc_response.text.strip()
        except Exception as e:
            logger.error(f"Error generating description: {str(e)}")
            # Keep the basic description set earlier
        
        # Detect language if not already determined from file extension
        if node.metadata.get("language") == "unknown":
            lang_prompt = f"What programming language is this code written in? Answer with just the language name:\n{code}"
            try:
                lang_response = await self.llm.acomplete(lang_prompt)
                node.metadata["language"] = lang_response.text.strip().lower()
            except Exception as e:
                logger.error(f"Error detecting language: {str(e)}")
    
    def _get_language_from_extension(self, ext: str) -> str:
        """Get programming language from file extension.
        
        Args:
            ext: File extension
            
        Returns:
            Programming language name
        """
        extension_to_language = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".rs": "rust",
            ".sh": "bash",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
        }
        
        return extension_to_language.get(ext, "unknown")
