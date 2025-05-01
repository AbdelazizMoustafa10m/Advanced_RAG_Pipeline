# --- processors/code/code_splitter.py ---

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import os
import importlib.util
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeRelationship
from llama_index.core.node_parser import CodeSplitter

from core.config import CodeProcessorConfig, CodeLanguage
from processors.code.language_detector import LanguageDetector

logger = logging.getLogger(__name__)

# Check if Chonkie is available
CHONKIE_AVAILABLE = False
CHONKIE_TOKENIZER = None

try:
    # First check if the package is installed
    import importlib.metadata
    try:
        chonkie_version = importlib.metadata.version("chonkie")
        logger.info(f"Found Chonkie version {chonkie_version}")
        
        # Try to import the required modules
        from chonkie import CodeChunker
        
        # Check if we have a tokenizer available
        try:
            from tokenizers import Tokenizer
            # Use GPT-2 tokenizer as default
            CHONKIE_TOKENIZER = "gpt2"
            logger.info("Using gpt2 tokenizer for Chonkie")
        except ImportError:
            logger.warning("Tokenizers package not found. Using token counting function for Chonkie.")
            # If tokenizers package is not available, we'll use a simple token counter function
            CHONKIE_TOKENIZER = lambda text: len(text.split())
        
        # Verify tree-sitter works by creating a simple test chunker
        try:
            # Simple initialization test with minimal code
            test_code = "def test(): pass\n"
            test_chunker = CodeChunker(
                language="python", 
                tokenizer_or_token_counter=CHONKIE_TOKENIZER,
                chunk_size=512,
                include_nodes=False
            )
            test_chunks = test_chunker.chunk(test_code)
            
            # If we get here, Chonkie is working properly
            CHONKIE_AVAILABLE = True
            logger.info(f"Chonkie with code support successfully initialized and tested")
            logger.info(f"Test chunking generated {len(test_chunks)} chunks")
        except Exception as e:
            logger.warning(f"Chonkie initialization test failed: {str(e)}")
            logger.warning("Make sure tree-sitter and language packs are properly installed")
    except importlib.metadata.PackageNotFoundError:
        logger.info("Chonkie package not found. Install with 'pip install chonkie[code]' for enhanced code chunking.")
    except ImportError as e:
        logger.warning(f"Chonkie package found but could not import required modules: {str(e)}")
        logger.warning("Make sure you have installed chonkie with code support: pip install 'chonkie[code]'")
    except Exception as e:
        logger.warning(f"Unexpected error initializing Chonkie: {str(e)}")
        
except Exception as e:
    logger.warning(f"Error checking for Chonkie: {str(e)}")
    logger.info("Falling back to LlamaIndex code splitting only")

if not CHONKIE_AVAILABLE:
    logger.warning("Chonkie support is disabled. Using LlamaIndex code splitting only.")
    # Make imports optional to avoid errors
    CodeChunker = None


@dataclass
class ChunkingStrategy:
    """Configuration for a code chunking strategy."""
    name: str
    enabled: bool = True
    priority: int = 0  # Higher number = higher priority
    
    # Strategy-specific parameters
    chunk_lines: Optional[int] = None
    chunk_size: Optional[int] = None  # For token-based chunkers
    max_chars: Optional[int] = None
    include_metadata: bool = True


class EnhancedCodeSplitter:
    """Enhanced code splitter that combines LlamaIndex and Chonkie capabilities.
    
    This splitter uses a multi-strategy approach to code chunking:
    1. AST-based chunking using tree-sitter (via LlamaIndex or Chonkie)
    2. Line-based chunking with configurable overlap
    3. Token-based chunking with size limits
    4. Character-based chunking as a fallback mechanism
    
    The splitter automatically selects the best strategy based on the code language,
    available dependencies, and configuration.
    """
    
    def __init__(self, config: Optional[CodeProcessorConfig] = None):
        """Initialize the enhanced code splitter.
        
        Args:
            config: Optional code processor configuration
        """
        self.config = config or CodeProcessorConfig()
        self.language_detector = LanguageDetector()
        
        # Define chunking strategies based on config
        self.strategies = []
        
        # Get enabled strategies from config
        enabled_strategy_names = self.config.chunking_strategies
        
        # Add strategies in the order specified in config
        strategy_configs = {
            "chonkie_ast": ChunkingStrategy(
                name="chonkie_ast",
                enabled=CHONKIE_AVAILABLE and "chonkie_ast" in enabled_strategy_names,
                priority=100,
                chunk_size=self.config.chunk_size
            ),
            "llamaindex_ast": ChunkingStrategy(
                name="llamaindex_ast",
                enabled="llamaindex_ast" in enabled_strategy_names,
                priority=90,
                chunk_lines=self.config.chunk_lines,
                max_chars=self.config.max_chars
            ),
            "semantic_line": ChunkingStrategy(
                name="semantic_line",
                enabled="semantic_line" in enabled_strategy_names,
                priority=80,
                chunk_lines=self.config.chunk_lines,
                max_chars=self.config.max_chars
            ),
            "basic_line": ChunkingStrategy(
                name="basic_line",
                enabled="basic_line" in enabled_strategy_names,
                priority=70,
                chunk_lines=self.config.chunk_lines,
                max_chars=self.config.max_chars
            ),
        }
        
        # Add strategies in the order specified in config
        for strategy_name in enabled_strategy_names:
            if strategy_name in strategy_configs:
                self.strategies.append(strategy_configs[strategy_name])
        
        # If no strategies were enabled, enable all as fallback
        if not self.strategies:
            logger.warning("No chunking strategies enabled in config, enabling all available strategies as fallback")
            for strategy in strategy_configs.values():
                # For Chonkie, still respect availability
                if strategy.name == "chonkie_ast" and not CHONKIE_AVAILABLE:
                    continue
                strategy.enabled = True
                self.strategies.append(strategy)
        
        # Log available strategies
        enabled_strategies = [s.name for s in self.strategies if s.enabled]
        logger.info(f"Enabled code chunking strategies (in priority order): {enabled_strategies}")
        
        # Cache for parsers to avoid repeated initialization
        self._parser_cache = {}
        
    def split(self, document: Document) -> List[TextNode]:
        """Split a code document into chunks using the best available strategy.
        
        Args:
            document: The code document to split
            
        Returns:
            List of text nodes representing code chunks
        """
        # Detect language
        lang_info = self.language_detector.detect_language(document)
        language = lang_info.get("language", "unknown")
        tree_sitter_language = lang_info.get("tree_sitter_language")
        
        logger.info(f"Detected language: {language} (tree-sitter: {tree_sitter_language})")
        
        # Try each strategy in order of priority until one succeeds
        enabled_strategies = sorted(
            [s for s in self.strategies if s.enabled],
            key=lambda s: s.priority,
            reverse=True
        )
        
        errors = []
        for strategy in enabled_strategies:
            try:
                logger.debug(f"Trying chunking strategy: {strategy.name}")
                
                if strategy.name == "chonkie_ast" and tree_sitter_language:
                    nodes = self._split_with_chonkie(document, tree_sitter_language, strategy)
                elif strategy.name == "llamaindex_ast" and tree_sitter_language:
                    nodes = self._split_with_llamaindex(document, tree_sitter_language, strategy)
                elif strategy.name == "semantic_line":
                    nodes = self._split_with_semantic_lines(document, language, strategy)
                elif strategy.name == "basic_line":
                    nodes = self._split_with_basic_lines(document, strategy)
                else:
                    continue
                
                if nodes:
                    logger.info(f"Successfully split document {document.doc_id} using {strategy.name} strategy")
                    # Add metadata to nodes
                    for node in nodes:
                        node.metadata["chunking_strategy"] = strategy.name
                        node.metadata["original_language"] = language
                        node.metadata["tree_sitter_language"] = tree_sitter_language
                    return nodes
            except Exception as e:
                error_msg = f"Error with {strategy.name} strategy: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
        
        # If all strategies failed, raise an exception with details
        error_details = "\n".join(errors)
        raise ValueError(f"All chunking strategies failed for document {document.doc_id}:\n{error_details}")
    
    def _split_with_chonkie(self, document: Document, language: str, strategy: ChunkingStrategy) -> List[TextNode]:
        """Split code using Chonkie's AST-based chunker.
        
        Args:
            document: The document to split
            language: The tree-sitter language to use
            strategy: Chunking strategy configuration
            
        Returns:
            List of text nodes
        """
        if not CHONKIE_AVAILABLE:
            raise ImportError("Chonkie is not available or properly initialized")
        
        try:
            # Get or create a chunker for this language
            chunker_key = f"chonkie_{language}_{strategy.chunk_size}"
            
            # Check if we need to create a new chunker
            if chunker_key not in self._parser_cache:
                logger.debug(f"Creating new Chonkie chunker for language: {language}")
                
                try:
                    # Create chunker with proper tokenizer and language
                    self._parser_cache[chunker_key] = CodeChunker(
                        language=language,
                        tokenizer_or_token_counter=CHONKIE_TOKENIZER,
                        chunk_size=strategy.chunk_size,
                        include_nodes=True,
                        return_type="chunks"
                    )
                    logger.debug(f"Successfully created Chonkie chunker for {language}")
                except Exception as e:
                    logger.warning(f"Failed to create Chonkie chunker for {language}: {str(e)}")
                    
                    # Try with different languages as fallback
                    fallback_languages = ["python", "javascript", "typescript"]
                    success = False
                    
                    # Only try fallbacks if we're not already using one of them
                    if language not in fallback_languages:
                        for fallback_lang in fallback_languages:
                            try:
                                logger.info(f"Trying to create Chonkie chunker with {fallback_lang} language as fallback")
                                self._parser_cache[chunker_key] = CodeChunker(
                                    language=fallback_lang,
                                    tokenizer_or_token_counter=CHONKIE_TOKENIZER,
                                    chunk_size=strategy.chunk_size,
                                    include_nodes=True,
                                    return_type="chunks"
                                )
                                success = True
                                logger.info(f"Successfully created Chonkie chunker with {fallback_lang} as fallback")
                                break
                            except Exception as fallback_error:
                                logger.warning(f"Failed to create Chonkie chunker with {fallback_lang} fallback: {str(fallback_error)}")
                    
                    # If all fallbacks failed, re-raise the original error
                    if not success:
                        raise ImportError(f"Could not initialize Chonkie chunker with any supported language: {str(e)}")
            
            # Get the chunker from cache
            chunker = self._parser_cache[chunker_key]
            
            # Process the document with detailed logging
            logger.debug(f"Chunking document with Chonkie: {document.doc_id}")
            
            # Actual chunking happens here
            try:
                code_chunks = chunker.chunk(document.text)
                logger.info(f"Chonkie successfully generated {len(code_chunks)} chunks for document {document.doc_id}")
            except Exception as chunk_error:
                logger.error(f"Error during Chonkie chunking operation: {str(chunk_error)}")
                raise
            
            # Convert Chonkie chunks to LlamaIndex nodes
            nodes = []
            for i, chunk in enumerate(code_chunks):
                # Create node with detailed metadata
                node = TextNode(
                    text=chunk.text,
                    metadata={
                        **document.metadata,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "token_count": chunk.token_count,
                        "chunk_index": i,
                        "total_chunks": len(code_chunks),
                        "node_count": len(chunk.nodes) if chunk.nodes else 0,
                        "chunker": "chonkie",
                        "language": language
                    }
                )
                
                # Set relationships to document
                node.relationships[NodeRelationship.SOURCE] = document.node_id
                
                nodes.append(node)
            
            return nodes
        except Exception as e:
            logger.error(f"Error in Chonkie chunking for {document.doc_id}: {str(e)}")
            # Provide detailed error information to help with debugging
            if hasattr(e, "__traceback__"):
                import traceback
                tb_str = ''.join(traceback.format_tb(e.__traceback__))
                logger.debug(f"Traceback: {tb_str}")
            
            raise ImportError(f"Chonkie chunking failed: {str(e)}")
    
    def _split_with_llamaindex(self, document: Document, language: str, strategy: ChunkingStrategy) -> List[TextNode]:
        """Split code using LlamaIndex's CodeSplitter.
        
        Args:
            document: The document to split
            language: The tree-sitter language to use
            strategy: Chunking strategy configuration
            
        Returns:
            List of text nodes
        """
        try:
            # Get or create a code splitter for this language
            splitter_key = f"llamaindex_{language}_{strategy.chunk_lines}_{strategy.max_chars}"
            if splitter_key not in self._parser_cache:
                logger.debug(f"Creating new LlamaIndex code splitter for language: {language}")
                self._parser_cache[splitter_key] = CodeSplitter(
                    language=language,
                    chunk_lines=strategy.chunk_lines,
                    max_chars=strategy.max_chars
                )
            
            code_splitter = self._parser_cache[splitter_key]
            
            # Process the document
            logger.debug(f"Chunking document with LlamaIndex: {document.doc_id}")
            nodes = code_splitter.get_nodes_from_documents([document])
            logger.info(f"LlamaIndex generated {len(nodes)} chunks for document {document.doc_id}")
            
            # Add chunker type to metadata
            for node in nodes:
                node.metadata["chunker"] = "llamaindex"
            
            return nodes
        except Exception as e:
            logger.error(f"Error in LlamaIndex chunking for {document.doc_id}: {str(e)}")
            raise
    
    def _split_with_semantic_lines(self, document: Document, language: str, strategy: ChunkingStrategy) -> List[TextNode]:
        """Split code using a semantic line-based approach.
        
        This approach tries to keep related code blocks together based on indentation
        and common patterns in the given language.
        
        Args:
            document: The document to split
            language: The programming language
            strategy: Chunking strategy configuration
            
        Returns:
            List of text nodes
        """
        try:
            logger.debug(f"Chunking document with semantic line approach: {document.doc_id}")
            lines = document.text.split("\n")
            chunk_lines = strategy.chunk_lines or 40
            max_chars = strategy.max_chars or 1500
            
            nodes = []
            current_chunk = []
            current_chars = 0
            in_block = False
            block_indent = 0
            
            for i, line in enumerate(lines):
                # Calculate indentation level
                indent = len(line) - len(line.lstrip())
                stripped = line.strip()
                
                # Check if this line starts a new block
                starts_block = False
                ends_block = False
                
                # Language-specific block detection
                if language == CodeLanguage.PYTHON.value:
                    starts_block = stripped.endswith(":") or stripped.startswith("@")
                    ends_block = indent <= block_indent and in_block and stripped
                elif language in [CodeLanguage.JAVASCRIPT.value, CodeLanguage.TYPESCRIPT.value, 
                                CodeLanguage.JAVA.value, CodeLanguage.CPP.value, 
                                CodeLanguage.C.value, CodeLanguage.CSHARP.value]:
                    starts_block = "{" in stripped and not stripped.startswith("//")
                    ends_block = "}" in stripped and indent <= block_indent and in_block
                
                # Update block state
                if starts_block:
                    in_block = True
                    block_indent = indent
                
                # Add line to current chunk
                current_chunk.append(line)
                current_chars += len(line) + 1  # +1 for newline
                
                # Check if we should end the chunk
                chunk_full = len(current_chunk) >= chunk_lines or current_chars >= max_chars
                at_block_end = ends_block and len(current_chunk) > 1
                
                if (chunk_full or at_block_end or i == len(lines) - 1) and current_chunk:
                    # Create a node for the current chunk
                    chunk_text = "\n".join(current_chunk)
                    node = TextNode(
                        text=chunk_text,
                        metadata={
                            **document.metadata,
                            "start_line": i - len(current_chunk) + 1,
                            "end_line": i,
                            "chunk_index": len(nodes),
                            "chunker": "semantic_line"
                        }
                    )
                    
                    # Set relationships to document
                    node.relationships[NodeRelationship.SOURCE] = document.node_id
                    
                    nodes.append(node)
                    
                    # Reset for next chunk
                    current_chunk = []
                    current_chars = 0
                
                # Update block state after processing
                if ends_block:
                    in_block = False
            
            # Add total chunks to metadata
            for i, node in enumerate(nodes):
                node.metadata["total_chunks"] = len(nodes)
            
            logger.info(f"Semantic line chunking generated {len(nodes)} chunks for document {document.doc_id}")
            return nodes
        except Exception as e:
            logger.error(f"Error in semantic line chunking for {document.doc_id}: {str(e)}")
            raise
    
    def _split_with_basic_lines(self, document: Document, strategy: ChunkingStrategy) -> List[TextNode]:
        """Split code using a basic line-based approach with overlap.
        
        Args:
            document: The document to split
            strategy: Chunking strategy configuration
            
        Returns:
            List of text nodes
        """
        try:
            logger.debug(f"Chunking document with basic line approach: {document.doc_id}")
            lines = document.text.split("\n")
            chunk_lines = strategy.chunk_lines or 40
            max_chars = strategy.max_chars or 1500
            
            nodes = []
            i = 0
            
            while i < len(lines):
                # Determine end of current chunk
                end_idx = min(i + chunk_lines, len(lines))
                
                # Create chunk from lines
                chunk_lines_content = lines[i:end_idx]
                chunk_text = "\n".join(chunk_lines_content)
                
                # Check if chunk exceeds max_chars
                if len(chunk_text) > max_chars and len(chunk_lines_content) > 1:
                    # Adjust end_idx to stay within max_chars
                    chars_so_far = 0
                    for j, line in enumerate(chunk_lines_content):
                        chars_so_far += len(line) + 1  # +1 for newline
                        if chars_so_far > max_chars:
                            end_idx = i + j
                            chunk_lines_content = chunk_lines_content[:j]
                            chunk_text = "\n".join(chunk_lines_content)
                            break
                
                # Create node
                node = TextNode(
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        "start_line": i,
                        "end_line": end_idx - 1,
                        "chunk_index": len(nodes),
                        "chunker": "basic_line"
                    }
                )
                
                # Set relationships to document
                node.relationships[NodeRelationship.SOURCE] = document.node_id
                
                nodes.append(node)
                
                # Move to next chunk with some overlap for context
                overlap = min(5, chunk_lines // 4)  # 25% overlap or 5 lines, whichever is smaller
                i = end_idx - overlap if end_idx < len(lines) else end_idx
            
            # Add total chunks to metadata
            for i, node in enumerate(nodes):
                node.metadata["total_chunks"] = len(nodes)
            
            logger.info(f"Basic line chunking generated {len(nodes)} chunks for document {document.doc_id}")
            return nodes
        except Exception as e:
            logger.error(f"Error in basic line chunking for {document.doc_id}: {str(e)}")
            raise


class CodeSplitterAdapter:
    """Adapter for backward compatibility with the original CodeSplitterAdapter.
    
    This adapter maintains the same interface as the original CodeSplitterAdapter
    but uses the enhanced implementation internally.
    """
    
    def __init__(self, config: Optional[CodeProcessorConfig] = None):
        """Initialize code splitter with configuration.
        
        Args:
            config: Optional code processor configuration
        """
        self.enhanced_splitter = EnhancedCodeSplitter(config)
    
    def split(self, document: Document) -> List[TextNode]:
        """Split a code document into chunks.
        
        Args:
            document: The code document to split
            
        Returns:
            List of text nodes representing code chunks
        """
        return self.enhanced_splitter.split(document)
