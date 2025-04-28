# --- processors/code/code_splitter.py ---

from typing import List, Optional
import logging
from typing import Dict, List, Optional

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import CodeSplitter

from core.config import CodeProcessorConfig, CodeLanguage

logger = logging.getLogger(__name__)


class CodeSplitterAdapter:
    """Adapter for LlamaIndex's CodeSplitter."""
    
    # Mapping from CodeLanguage enum values to tree-sitter supported languages
    # Based on available languages in tree-sitter-language-pack
    LANGUAGE_MAPPING: Dict[str, str] = {
        CodeLanguage.PYTHON.value: "python",
        CodeLanguage.JAVASCRIPT.value: "javascript",
        CodeLanguage.TYPESCRIPT.value: "typescript",
        CodeLanguage.JAVA.value: "java",
        CodeLanguage.CPP.value: "cpp",
        CodeLanguage.C.value: "c",
        CodeLanguage.GO.value: "go",
        CodeLanguage.CSHARP.value: "c_sharp",  # tree-sitter uses c_sharp
        CodeLanguage.RUBY.value: "ruby",
        CodeLanguage.PHP.value: "php",
        CodeLanguage.RUST.value: "rust",
        CodeLanguage.SWIFT.value: "swift",
        CodeLanguage.KOTLIN.value: "kotlin",
        CodeLanguage.OTHER.value: "python",  # Fallback
        CodeLanguage.UNKNOWN.value: "python",  # Fallback
    }
    
    def __init__(self, config: Optional[CodeProcessorConfig] = None):
        """Initialize code splitter with configuration.
        
        Args:
            config: Optional code processor configuration
        """
        self.config = config or CodeProcessorConfig()
        
        # Initialize code splitter with updated API
        # Note: chunk_overlap_lines is no longer supported in newer versions
        # Use 'python' as default language instead of 'auto' to avoid tree-sitter error
        default_language = "python"
        
        # Don't initialize with 'auto' as it's not a valid tree-sitter language
        # The actual language will be determined in the split method
        self.code_splitter = CodeSplitter(
            language=default_language,
            chunk_lines=self.config.chunk_lines,
            max_chars=self.config.max_chars
        )
    
    def split(self, document: Document) -> List[TextNode]:
        """Split a code document into chunks.
        
        Args:
            document: The code document to split
            
        Returns:
            List of text nodes representing code chunks
        """
        try:
            # Extract language from document metadata if available
            raw_language = document.metadata.get("language", "python")
            
            # If language_detection is not set to "auto" in config, use the configured language
            if self.config.language_detection != "auto":
                raw_language = self.config.language_detection
            
            # Map the detected language to a tree-sitter supported language
            tree_sitter_language = self._map_to_tree_sitter_language(raw_language)
            logger.debug(f"Mapped language '{raw_language}' to tree-sitter language '{tree_sitter_language}'")
                
            # Create a new code splitter with the detected language
            code_splitter = CodeSplitter(
                language=tree_sitter_language,
                chunk_lines=self.config.chunk_lines,
                max_chars=self.config.max_chars
            )
            
            nodes = code_splitter.get_nodes_from_documents([document])
            logger.debug(f"Split document {document.doc_id} into {len(nodes)} code chunks using language: {tree_sitter_language}")
            
            # Add the original language to the metadata of each node
            for node in nodes:
                node.metadata["original_language"] = raw_language
            
            return nodes
        
        except Exception as e:
            logger.error(f"Error splitting code document {document.doc_id}: {str(e)}")
            logger.error(f"Falling back to Python language parser")
            
            # Fallback to Python parser if there's an error with the detected language
            try:
                code_splitter = CodeSplitter(
                    language="python",
                    chunk_lines=self.config.chunk_lines,
                    max_chars=self.config.max_chars
                )
                nodes = code_splitter.get_nodes_from_documents([document])
                logger.warning(f"Successfully split document {document.doc_id} using fallback Python parser")
                return nodes
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                raise
    
    def _map_to_tree_sitter_language(self, language: str) -> str:
        """Map a language string to a valid tree-sitter language.
        
        Args:
            language: The language string to map
            
        Returns:
            A valid tree-sitter language string
        """
        # Convert to lowercase for case-insensitive matching
        language_lower = language.lower()
        
        # Check if it's in our mapping
        if language_lower in self.LANGUAGE_MAPPING:
            return self.LANGUAGE_MAPPING[language_lower]
        
        # Handle file extensions if they were passed directly
        if language_lower.startswith('.'):
            # Try to map common file extensions
            ext_mapping = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.java': 'java',
                '.cpp': 'cpp',
                '.c': 'c',
                '.go': 'go',
                '.cs': 'c_sharp',
                '.rb': 'ruby',
                '.php': 'php',
                '.rs': 'rust',
                '.swift': 'swift',
                '.kt': 'kotlin',
            }
            if language_lower in ext_mapping:
                return ext_mapping[language_lower]
        
        # If we can't map it, default to Python
        logger.warning(f"Could not map language '{language}' to a tree-sitter language, using Python as fallback")
        return "python"
