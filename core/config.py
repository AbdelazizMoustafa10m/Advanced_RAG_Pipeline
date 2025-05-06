# --- core/config.py ---

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import os
import logging
from pathlib import Path

# Create logger
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Enum for document types."""
    CODE = "code"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class CodeLanguage(Enum):
    """Enum for code languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    RUST = "rust"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    OTHER = "other"
    UNKNOWN = "unknown"


class DocumentFormat(Enum):
    """Enum for document formats."""
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    HTML = "html"
    TXT = "txt"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    console: bool = True


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    enabled: bool = True
    max_workers: int = 4


@dataclass
class DetectorConfig:
    """Configuration for document type detection."""
    use_file_extension: bool = True
    use_content_analysis: bool = False
    use_magic_numbers: bool = True
    parallel_processing: bool = True

    # File extension mappings
    code_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".java", ".cpp", ".go", ".cs", ".rb",
        ".php", ".rs", ".swift", ".kt", ".c", ".h", ".hpp", ".scala"
    ])

    document_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".md", ".html", ".txt", ".rtf", ".odt", ".csv",
        ".json", ".xml", ".yaml", ".yml", ".pptx", ".ppt"
    ])

    # Data file extensions
    data_extensions: List[str] = field(default_factory=lambda: [
        ".csv", ".json", ".xml", ".yaml", ".yml"
    ])

    # Presentation file extensions
    presentation_extensions: List[str] = field(default_factory=lambda: [
        ".pptx", ".ppt", ".odp", ".key"
    ])

    # Web file extensions
    web_extensions: List[str] = field(default_factory=lambda: [
        ".html", ".htm", ".css", ".js"
    ])

    # Confidence thresholds for detection
    min_confidence: float = 0.7
    high_confidence: float = 0.9
    medium_confidence: float = 0.7
    low_confidence: float = 0.4


@dataclass
class LoaderConfig:
    """Configuration for document loading."""
    recursive: bool = True
    exclude_patterns: List[str] = field(default_factory=lambda: [
        ".*", "node_modules", "__pycache__", "venv", ".git"
    ])
    include_patterns: List[str] = field(default_factory=lambda: [])

    # File extraction configurations
    file_extractors: Dict[str, str] = field(default_factory=dict)


@dataclass
class CodeProcessorConfig:
    """Configuration for code processing."""
    # Chunking parameters
    chunk_lines: int = 60
    chunk_overlap_lines: int = 15
    max_chars: int = 2000
    chunk_size: int = 512  # Token size for Chonkie chunker (in tokens)

    # Chunking strategy options
    chunking_strategies: List[str] = field(default_factory=lambda: [
        "chonkie_ast",  # AST-based chunking with Chonkie (highest priority)
        "llamaindex_ast",  # AST-based chunking with LlamaIndex
        "semantic_line",  # Semantic line-based chunking
        "basic_line"  # Basic line-based chunking with overlap
    ])

    # Language detection
    language_detection: str = "auto"  # Use "auto" to detect language or specify like "python", "javascript", etc.

    # Embedding parameters
    include_imports: bool = True
    include_comments: bool = True
    include_docstrings: bool = True


@dataclass
class DoclingConfig:
    """Configuration for Docling document processing."""
    # OCR settings
    do_ocr: bool = True

    # Table extraction settings
    do_table_structure: bool = True
    table_structure_mode: str = "ACCURATE"  # "ACCURATE" or "FAST"

    # Enrichment settings
    do_code_enrichment: bool = True
    do_formula_enrichment: bool = True

    # Chunking settings
    chunking_strategy: str = "semantic"  # "semantic", "fixed", or "hierarchical"
    chunk_size: int = 1000
    chunk_overlap: int = 200


# --- NEW: Configuration for a specific LLM instance/role ---
@dataclass
class LLMSettings:
    """Configuration for a specific LLM instance."""
    model_name: Optional[str] = None
    provider: str = "groq" # Default provider
    api_key_env_var: Optional[str] = None # Env var for THIS specific LLM's key
    enabled: bool = True  # Whether this LLM is enabled
    temperature: float = 0.1 # Example default setting
    max_tokens: Optional[int] = None # Example default setting
    request_timeout: int = 60
    max_retries: int = 3
    # Add other provider-specific args as needed using Dict[str, Any]
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> Optional[str]:
        """Get API key from environment variable if specified."""
        if self.api_key_env_var:
            return os.getenv(self.api_key_env_var)
        logger.warning(f"API key env var not specified for model {self.model_name}")
        return None


# --- REVISED: Main LLM Configuration holding settings for different roles ---
@dataclass
class LLMConfig:
    """Configuration holder for different LLM roles."""
    # Define settings for each role
    metadata_llm: LLMSettings = field(default_factory=lambda: LLMSettings(
        model_name=os.getenv("METADATA_LLM_MODEL", "llama-3.1-8b-instant"),
        provider=os.getenv("METADATA_LLM_PROVIDER", "groq"),
        api_key_env_var=os.getenv("METADATA_LLM_API_KEY_ENV_VAR", "GROQ_API_KEY") # Example: reuse groq key by default
    ))
    query_llm: LLMSettings = field(default_factory=lambda: LLMSettings(
        model_name=os.getenv("QUERY_LLM_MODEL", "llama-3.1-70b-versatile"),
        provider=os.getenv("QUERY_LLM_PROVIDER", "groq"),
        api_key_env_var=os.getenv("QUERY_LLM_API_KEY_ENV_VAR", "GROQ_API_KEY") # Example: reuse groq key by default
    ))
    coding_llm: LLMSettings = field(default_factory=lambda: LLMSettings(
         model_name=os.getenv("CODING_LLM_MODEL", "llama-3.1-70b-instant"),
         provider=os.getenv("CODING_LLM_PROVIDER", "groq"),
         api_key_env_var=os.getenv("CODING_LLM_API_KEY_ENV_VAR")
    ))
    # Metadata enrichment controls by document type
    enrich_documents: bool = True  # Controls enrichment for PDF, DOCX, etc. via Docling
    enrich_code: bool = True       # Controls enrichment for code files via CodeSplitter
    use_cache: bool = True
    cache_dir: str = "./.cache/llm"


@dataclass
class EmbedderConfig:
    """Configuration for node embedding using LlamaIndex embedding models."""
    # Embedding provider and model
    provider: str = "huggingface"  # Options: huggingface, openai, cohere, vertex, bedrock, etc.
    model_name: str = "BAAI/bge-small-en-v1.5"  # Default model

    # API configuration for hosted models
    api_key_env_var: Optional[str] = None  # Environment variable name for API key
    api_base: Optional[str] = None  # Base URL for API

    # Performance settings
    embed_batch_size: int = 10  # Number of texts to embed in a single batch (uses LlamaIndex default if not specified)

    # Caching settings
    use_cache: bool = True  # Cache embeddings to avoid recomputing
    cache_dir: str = "./.cache/embeddings"  # Directory for embedding cache

    # Advanced settings for specific providers
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)  # Additional kwargs for specific providers

    # Fallback settings
    fallback_provider: Optional[str] = None  # Fallback provider if primary fails
    fallback_model: Optional[str] = None  # Fallback model if primary fails

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Create cache directory if caching is enabled
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)


@dataclass
class VectorStoreConfig:
    """Configuration for vector storage."""
    # General settings
    engine: str = "chroma"  # Options: "chroma", "qdrant"
    collection_name: str = "unified_knowledge"
    distance_metric: str = "cosine"  # Options: "cosine", "euclid", "dot"

    # ChromaDB specific settings
    vector_db_path: str = "./vector_db"  # Local path for ChromaDB
    persist_directory: Optional[str] = None  # Optional alternative persist directory

    # Qdrant specific settings
    qdrant_location: str = "local"  # Options: "local", "cloud"
    qdrant_url: Optional[str] = None  # Required for cloud, format: "https://your-cluster-url.qdrant.io"
    qdrant_api_key: Optional[str] = None  # Required for cloud
    qdrant_local_path: str = "./qdrant_db"  # Path for local Qdrant storage
    qdrant_grpc_port: int = 6334  # gRPC port for local Qdrant
    qdrant_prefer_grpc: bool = False  # Whether to prefer gRPC over HTTP

    # Advanced Qdrant settings
    qdrant_timeout: float = 10.0  # Timeout for Qdrant operations in seconds
    qdrant_vector_size: Optional[int] = None  # Vector size, determined from first vector if None
    qdrant_shard_number: Optional[int] = None  # Number of shards for Qdrant collection
    qdrant_replication_factor: Optional[int] = None  # Replication factor for Qdrant collection
    qdrant_write_consistency_factor: Optional[int] = None  # Write consistency factor
    qdrant_on_disk_payload: bool = True  # Whether to store payload on disk

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate engine
        if self.engine not in ["chroma", "qdrant"]:
            raise ValueError(f"Unsupported vector store engine: {self.engine}. Supported engines: chroma, qdrant")

        # Validate Qdrant configuration if selected
        if self.engine == "qdrant":
            if self.qdrant_location not in ["local", "cloud"]:
                raise ValueError(f"Unsupported Qdrant location: {self.qdrant_location}. Supported locations: local, cloud")

            if self.qdrant_location == "cloud" and not self.qdrant_url:
                raise ValueError("Qdrant cloud location requires qdrant_url to be set")

            if self.qdrant_location == "cloud" and not self.qdrant_api_key:
                raise ValueError("Qdrant cloud location requires qdrant_api_key to be set")

        # Create directories
        if self.engine == "chroma":
            os.makedirs(self.vector_db_path, exist_ok=True)
        elif self.engine == "qdrant" and self.qdrant_location == "local":
            os.makedirs(self.qdrant_local_path, exist_ok=True)
@dataclass
class QueryConfig:
    """Configuration for query processing."""
    similarity_top_k: int = 5
    response_mode: str = "compact"  # "compact", "tree", "refine", etc.

    # Filtering settings
    default_filter_mode: str = "AND"  # "AND" or "OR"

    # Language detection for code queries
    detect_language_in_query: bool = True

# --- NEW: Registry Configuration ---
@dataclass
class RegistryConfig:
    """Configuration for document registry."""
    enabled: bool = True
    db_path: Optional[str] = None  # If None, will default to output_dir/document_registry.db
    reset_stalled_after_seconds: int = 3600  # Reset stalled docs after 1 hour

@dataclass
class QueryTransformationConfig:
    """Configuration for query transformation."""
    # Enable/disable features
    enable_query_expansion: bool = True
    enable_query_rewriting: bool = True

    # HyDE (Hypothetical Document Embeddings) settings
    use_hyde: bool = True
    hyde_prompt_template: str = "Please write a passage that answers the question: {query}"

    # Query expansion settings
    expansion_limit: int = 3
    expansion_technique: str = "llm"  # Options: "llm", "keyword", "hybrid"

    # Query decomposition settings
    enable_decomposition: bool = False
    decomposition_mode: str = "step"  # Options: "step", "sub_question"
    max_sub_questions: int = 5

    # Query rewriting settings
    rewriting_technique: str = "instruct"  # Options: "instruct", "reflexion", "step_back"
    query_iterations: int = 1  # Number of query rewrites to attempt


@dataclass
class RetrievalConfig:
    """Configuration for retrieval components."""
    # General retrieval settings
    retriever_strategy: str = "vector"  # Options: "vector", "keyword", "hybrid", "ensemble"
    similarity_top_k: int = 5
    use_filter_cache: bool = True

    # Hybrid search settings
    use_hybrid_search: bool = False
    hybrid_mode: str = "OR"  # Options: "AND", "OR"
    hybrid_alpha: float = 0.5  # Weight between vector (alpha) and keyword (1-alpha) scores

    # Keyword search settings
    keyword_top_k: int = 10
    use_bm25: bool = True
    use_splade: bool = False


@dataclass
class RerankerConfig:
    """Configuration for reranking components."""
    enable_reranking: bool = True
    reranker_type: str = "semantic"  # Options: "semantic", "llm", "fusion", "cohere"
    rerank_top_n: int = 10  # Number of results to rerank
    batch_size: int = 5  # Batch size for processing nodes

    # Semantic reranking settings
    semantic_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM reranking settings
    rerank_prompt_template: str = """
    Query: {query}

    Document: {context}

    On a scale of 1 to 10, how relevant is this document to the query?
    Respond with just a number.
    """

    # Cohere reranking settings
    cohere_api_key: str = os.environ.get("COHERE_API_KEY", "")


@dataclass
class SynthesisConfig:
    """Configuration for answer synthesis."""
    synthesis_strategy: str = "refine"  # Options: "refine", "tree", "compact", "simple"
    streaming: bool = False
    use_async: bool = False

    # Refine strategy settings
    refine_prompt_template: str = """
    The original query is: {query}

    We have provided an existing answer: {existing_answer}

    We have the opportunity to refine the existing answer with new context: {context}

    Given the new context, refine the original answer. If the context isn't useful, return the original answer.
    """

    # Tree strategy settings
    tree_width: int = 3  # Number of chunks to process at once in the tree

    # Response formatting
    include_citations: bool = True
    include_metadata: bool = True
    structured_answer_filtering: bool = False


@dataclass
class QueryPipelineConfig:
    """Configuration for the query pipeline."""
    # Component configurations
    transformation: QueryTransformationConfig = field(default_factory=QueryTransformationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)

    # Pipeline settings
    timeout_seconds: int = 60
    async_mode: bool = False
    streaming_response: bool = False
    cache_results: bool = True
    cache_dir: str = "./.cache/query_results"
    verbose: bool = False

    def __post_init__(self):
        """Create cache directory if caching is enabled."""
        if self.cache_results and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

# --- REVISED: Main Unified Configuration ---
@dataclass
class UnifiedConfig:
    """Main configuration for the unified document parsing system."""
    # --- Global settings ---
    input_directory: str # Required parameter - must come first
    project_name: str = "unified-parser"
    output_dir: str = "./output" # Optional output dir

    # --- Component configs ---
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    code_processor: CodeProcessorConfig = field(default_factory=CodeProcessorConfig)
    docling: DoclingConfig = field(default_factory=DoclingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig) # Uses the new LLMConfig
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig) # New embedder config
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    query_pipeline: QueryPipelineConfig = field(default_factory=QueryPipelineConfig)  # New query pipeline config

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Input directory validation
        input_path = Path(self.input_directory)
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError(f"Input directory does not exist: {self.input_directory}")

        # Ensure API keys are set if models are specified
        if self.llm.metadata_llm.model_name and not self.llm.metadata_llm.api_key:
            raise ValueError(f"Metadata LLM model specified ('{self.llm.metadata_llm.model_name}') but API key env var '{self.llm.metadata_llm.api_key_env_var}' not set or empty.")
        if self.llm.query_llm.model_name and not self.llm.query_llm.api_key:
             raise ValueError(f"Query LLM model specified ('{self.llm.query_llm.model_name}') but API key env var '{self.llm.query_llm.api_key_env_var}' not set or empty.")
        if self.llm.coding_llm and self.llm.coding_llm.model_name and not self.llm.coding_llm.api_key:
             logger.warning(f"Coding LLM model specified ('{self.llm.coding_llm.model_name}') but API key env var '{self.llm.coding_llm.api_key_env_var}' not set or empty.")

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        if self.vector_store.vector_db_path:
            os.makedirs(self.vector_store.vector_db_path, exist_ok=True)
        if self.llm.use_cache and self.llm.cache_dir:
            os.makedirs(self.llm.cache_dir, exist_ok=True)
        if self.query_pipeline.cache_results and self.query_pipeline.cache_dir:
            os.makedirs(self.query_pipeline.cache_dir, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> UnifiedConfig:
    """Load configuration from a file or use defaults.

    Args:
        config_path: Path to configuration file (YAML or JSON)

    Returns:
        Loaded configuration
    """
    if config_path is None:
        return UnifiedConfig()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load configuration from file
    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML is not installed. Using default configuration instead.")
            return UnifiedConfig()
    elif config_path.suffix.lower() == '.json':
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

    # TODO: Convert dict to nested dataclasses recursively
    # For now, return default config
    return UnifiedConfig()
