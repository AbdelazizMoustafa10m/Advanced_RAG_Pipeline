# --- processors/code/language_detector.py ---

from typing import Dict, List, Optional, Any
import logging
import os
import re

from llama_index.core import Document
from core.config import CodeLanguage

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Detects programming languages from code content and file metadata."""
    
    # Mapping from CodeLanguage enum values to tree-sitter supported languages
    TREE_SITTER_MAPPING: Dict[str, str] = {
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
        CodeLanguage.OTHER.value: None,  # No mapping
        CodeLanguage.UNKNOWN.value: None,  # No mapping
    }
    
    # File extension to language mapping
    EXTENSION_MAPPING: Dict[str, str] = {
        # Python
        ".py": CodeLanguage.PYTHON.value,
        ".pyi": CodeLanguage.PYTHON.value,
        ".ipynb": CodeLanguage.PYTHON.value,
        
        # JavaScript
        ".js": CodeLanguage.JAVASCRIPT.value,
        ".jsx": CodeLanguage.JAVASCRIPT.value,
        ".mjs": CodeLanguage.JAVASCRIPT.value,
        
        # TypeScript
        ".ts": CodeLanguage.TYPESCRIPT.value,
        ".tsx": CodeLanguage.TYPESCRIPT.value,
        
        # Java
        ".java": CodeLanguage.JAVA.value,
        
        # C++
        ".cpp": CodeLanguage.CPP.value,
        ".cc": CodeLanguage.CPP.value,
        ".cxx": CodeLanguage.CPP.value,
        ".hpp": CodeLanguage.CPP.value,
        ".hxx": CodeLanguage.CPP.value,
        
        # C
        ".c": CodeLanguage.C.value,
        ".h": CodeLanguage.C.value,
        
        # Go
        ".go": CodeLanguage.GO.value,
        
        # C#
        ".cs": CodeLanguage.CSHARP.value,
        
        # Ruby
        ".rb": CodeLanguage.RUBY.value,
        
        # PHP
        ".php": CodeLanguage.PHP.value,
        
        # Rust
        ".rs": CodeLanguage.RUST.value,
        
        # Swift
        ".swift": CodeLanguage.SWIFT.value,
        
        # Kotlin
        ".kt": CodeLanguage.KOTLIN.value,
        ".kts": CodeLanguage.KOTLIN.value,
    }
    
    # Language detection patterns based on content
    CONTENT_PATTERNS: Dict[str, List[str]] = {
        CodeLanguage.PYTHON.value: [
            r"^\s*import\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*\s*$",
            r"^\s*from\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import\s+",
            r"^\s*def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
            r"^\s*class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*\(\s*[a-zA-Z_][a-zA-Z0-9_.,\s]*\s*\))?\s*:",
        ],
        CodeLanguage.JAVASCRIPT.value: [
            r"^\s*import\s+(?:{[^}]*}|\*\s+as\s+[a-zA-Z_$][a-zA-Z0-9_$]*|[a-zA-Z_$][a-zA-Z0-9_$]*)\s+from\s+['\"][^'\"]+['\"]",
            r"^\s*export\s+(?:default\s+)?(?:function|class|const|let|var)\s+",
            r"^\s*const\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*=",
            r"^\s*function\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*\(",
        ],
        CodeLanguage.TYPESCRIPT.value: [
            r"^\s*import\s+(?:{[^}]*}|\*\s+as\s+[a-zA-Z_$][a-zA-Z0-9_$]*|[a-zA-Z_$][a-zA-Z0-9_$]*)\s+from\s+['\"][^'\"]+['\"]",
            r"^\s*export\s+(?:default\s+)?(?:function|class|interface|type|const|let|var)\s+",
            r"^\s*interface\s+[a-zA-Z_$][a-zA-Z0-9_$]*(?:\s+extends\s+[a-zA-Z_$][a-zA-Z0-9_$,\s]*)?",
            r"^\s*type\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*=",
        ],
        CodeLanguage.JAVA.value: [
            r"^\s*package\s+[a-zA-Z_][a-zA-Z0-9_.]*\s*;",
            r"^\s*import\s+(?:static\s+)?[a-zA-Z_][a-zA-Z0-9_.]*(?:\.\*)?\s*;",
            r"^\s*public\s+class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[a-zA-Z_][a-zA-Z0-9_,\s]*)?",
            r"^\s*(?:public|private|protected)\s+(?:static\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
        ],
        CodeLanguage.CPP.value: [
            r"^\s*#include\s+[<\"][a-zA-Z0-9_./]+\.(h|hpp)[>\"]",
            r"^\s*namespace\s+[a-zA-Z_][a-zA-Z0-9_]*",
            r"^\s*class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:\s*(?:public|private|protected)\s+[a-zA-Z_][a-zA-Z0-9_]*)?",
            r"^\s*template\s*<",
        ],
        CodeLanguage.C.value: [
            r"^\s*#include\s+[<\"][a-zA-Z0-9_./]+\.h[>\"]",
            r"^\s*struct\s+[a-zA-Z_][a-zA-Z0-9_]*",
            r"^\s*typedef\s+",
            r"^\s*void\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
            r"^\s*int\s+main\s*\(",
        ],
        CodeLanguage.GO.value: [
            r"^\s*package\s+[a-zA-Z_][a-zA-Z0-9_]*\s*$",
            r"^\s*import\s+\(",
            r"^\s*import\s+\"",
            r"^\s*func\s+\([a-zA-Z_][a-zA-Z0-9_]*\s+\*?[a-zA-Z_][a-zA-Z0-9_]*\)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
            r"^\s*func\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
        ],
    }
    
    def __init__(self):
        """Initialize the language detector."""
        pass
    
    def detect_language(self, document: Document) -> Dict[str, Any]:
        """Detect the programming language of a document.
        
        Args:
            document: The document to analyze
            
        Returns:
            Dict containing language information
        """
        # Initialize result
        result = {
            "language": CodeLanguage.UNKNOWN.value,
            "confidence": 0.0,
            "tree_sitter_language": None,
            "detection_method": None
        }
        
        # Check if language is already in metadata
        if "language" in document.metadata:
            language = document.metadata["language"]
            # Validate language
            if language in [lang.value for lang in CodeLanguage]:
                result["language"] = language
                result["confidence"] = 1.0
                result["detection_method"] = "metadata"
                result["tree_sitter_language"] = self.TREE_SITTER_MAPPING.get(language)
                return result
        
        # Check file extension
        file_path = document.metadata.get("file_path", "")
        if file_path:
            extension = os.path.splitext(file_path)[1].lower()
            if extension in self.EXTENSION_MAPPING:
                language = self.EXTENSION_MAPPING[extension]
                result["language"] = language
                result["confidence"] = 0.8
                result["detection_method"] = "file_extension"
                result["tree_sitter_language"] = self.TREE_SITTER_MAPPING.get(language)
                
                # Special handling for C vs C++ header files
                if extension == ".h" and self._looks_like_cpp(document.text):
                    result["language"] = CodeLanguage.CPP.value
                    result["tree_sitter_language"] = self.TREE_SITTER_MAPPING.get(CodeLanguage.CPP.value)
                
                return result
        
        # Content analysis
        if document.text:
            language_scores = self._analyze_content(document.text)
            if language_scores:
                # Get the language with the highest score
                best_language, score = max(language_scores.items(), key=lambda item: item[1])
                
                # Only assign if confidence is reasonable
                if score > 0.5:
                    result["language"] = best_language
                    result["confidence"] = score
                    result["detection_method"] = "content_analysis"
                    result["tree_sitter_language"] = self.TREE_SITTER_MAPPING.get(best_language)
                    return result
        
        # If we get here, we couldn't confidently detect the language
        # Map to tree-sitter language anyway for best-effort parsing
        result["tree_sitter_language"] = self.TREE_SITTER_MAPPING.get(result["language"])
        
        return result
    
    def _analyze_content(self, text: str) -> Dict[str, float]:
        """Analyze code content to determine the language.
        
        Args:
            text: The code text to analyze
            
        Returns:
            Dict mapping languages to confidence scores
        """
        scores = {}
        
        # Split the text into lines for analysis
        lines = text.split("\n")
        
        # Only use the first 100 lines for performance
        lines = lines[:100]
        
        # Track matches for each language
        matches = {lang: 0 for lang in self.CONTENT_PATTERNS.keys()}
        
        # Check each line for language patterns
        for line in lines:
            for language, patterns in self.CONTENT_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, line):
                        matches[language] += 1
                        break  # Only count one match per line
        
        # Calculate scores based on match ratios
        total_lines = len(lines)
        if total_lines > 0:
            for language, match_count in matches.items():
                scores[language] = match_count / total_lines
        
        return scores
    
    def _looks_like_cpp(self, text: str) -> bool:
        """Check if a C header file is actually a C++ header.
        
        Args:
            text: The code text to analyze
            
        Returns:
            True if it looks like C++, False otherwise
        """
        # Look for C++ specific features
        cpp_patterns = [
            r"class\s+[a-zA-Z_][a-zA-Z0-9_]*",
            r"namespace\s+[a-zA-Z_][a-zA-Z0-9_]*",
            r"template\s*<",
            r"::\s*[a-zA-Z_][a-zA-Z0-9_]*",
            r"std::",
            r"public:|private:|protected:"
        ]
        
        for pattern in cpp_patterns:
            if re.search(pattern, text):
                return True
                
        return False