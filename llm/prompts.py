# --- Prompts for Document Nodes (from your CustomDoclingEnricher) ---

DOC_TITLE_PROMPT = """\
Context:
- {formatted_source} 
- {formatted_location}
- {formatted_headings}
- {formatted_label}

Text Chunk:
--- START ---
{text_chunk}
--- END ---

Task: Based *only* on the Text Chunk provided above and its formatted context, generate a concise and functional title (5-10 words maximum) that accurately describes the main topic, purpose, or key entities discussed *specifically within this chunk*. Avoid generic phrases.

Functional Title:""" # <-- UPDATED CONTEXT PLACEHOLDERS

DOC_SUMMARY_PROMPT = """\
Context:
- {formatted_source} 
- {formatted_location}
- {formatted_headings}
- {formatted_label}

Text Chunk:
--- START ---
{text_chunk}
--- END ---

Task: Write a 1-2 sentence summary explaining the main point, function, or key information presented *specifically within the Text Chunk above*. Consider its formatted context. Be objective and concise.

Concise Summary:""" # <-- UPDATED CONTEXT PLACEHOLDERS

# Using simplified output format
DOC_QUESTIONS_PROMPT = """\
Context:
- {formatted_source} 
- {formatted_location}
- {formatted_headings}
- {formatted_label}

Text Chunk:
--- START ---
{text_chunk}
--- END ---

Task: Generate exactly {num_questions} specific questions based *solely* on the information present in the Text Chunk above.
- The questions should be answerable *only* using the provided Text Chunk.
- Do NOT ask questions about the context metadata itself.
- Separate each question with the delimiter '|||'. Example: Question 1?|||Question 2?

Questions:""" # <-- UPDATED CONTEXT PLACEHOLDERS

# --- Prompts for Code Nodes (from Context7MetadataExtractor logic) ---

CODE_TITLE_PROMPT = """\
Context:
- File Path: {file_path}
- Language: {language}
- Chunk Position: Lines {start_line}-{end_line} (approx)
- Chunking Method: {chunking_strategy} 

Code Snippet:
```{language}
{code_chunk}

Task: Based on the Code Snippet and its context, generate a concise technical title (5-10 words max) describing its specific functionality or purpose within the file.

Concise Title:"""

CODE_DESC_PROMPT = """
Context:
File Path: {file_path}
Language: {language}
Chunk Position: Lines {start_line}-{end_line} (approx)
Chunking Method: {chunking_strategy}
Code Snippet:
```{language}
{code_chunk}

Task: Write a brief technical description (1-3 sentences) of what this specific code snippet does and its main purpose in the context of the file it belongs to. Focus on functionality and its role.
Description:"""

# Note: Language detection via LLM might be slow/expensive. Relying on file extension
# or tree-sitter (if CodeSplitter provides it) is often better.
CODE_LANG_PROMPT = """\
What programming language is this code snippet most likely written in? Answer with just the lowercase language name (e.g., python, javascript, java). If unsure, answer 'unknown'.

Code Snippet:
{code_chunk}

Language:"""

# Optional: A separate QA prompt for code if desired
CODE_QUESTIONS_PROMPT = """\
Context:
- File Path: {file_path}
- Language: {language}
- Chunk Position: Lines {start_line}-{end_line} (approx)
- Chunking Method: {chunking_strategy}
Code Snippet:
```{language}
{code_chunk}
```
Task: Generate exactly {num_questions} specific questions a developer might ask about the functionality, usage, or integration of this code snippet within its file context.
Questions:"""
