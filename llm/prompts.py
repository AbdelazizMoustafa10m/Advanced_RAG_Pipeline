# --- Prompts for Document Nodes (from your CustomDoclingEnricher) ---

DOC_TITLE_PROMPT = """\
Context:
- Document Section Headings: [{headings}]
- Node Type: {node_type}
- Page Label: {page_label}

Text Chunk:
--- START ---
{text_chunk}
--- END ---

Task: Based *only* on the Text Chunk provided above and its context (headings, type), generate a concise and functional title (5-10 words maximum) that accurately describes the main topic, purpose, or key entities discussed *specifically within this chunk*. Avoid generic phrases.

Functional Title:"""

DOC_SUMMARY_PROMPT = """\
Context:
- Document Section Headings: [{headings}]
- Node Type: {node_type}
- Page Label: {page_label}

Text Chunk:
--- START ---
{text_chunk}
--- END ---

Task: Write a 1-2 sentence summary explaining the main point, function, or key information presented *specifically within the Text Chunk above*. Consider its context (headings, type). Be objective and concise.

Concise Summary:"""

# Using simplified output format
DOC_QUESTIONS_PROMPT = """\
Context:
- Document Section Headings: [{headings}]
- Node Type: {node_type}
- Page Label: {page_label}

Text Chunk:
--- START ---
{text_chunk}
--- END ---

Task: Generate exactly {num_questions} specific questions based *solely* on the information present in the Text Chunk above.
- The questions should be answerable *only* using the provided Text Chunk.
- Do NOT ask questions about the context (headings, node type, page label) itself.
- Separate each question with the delimiter '|||'. Example: Question 1?|||Question 2?

Questions:"""

# --- Prompts for Code Nodes (from Context7MetadataExtractor logic) ---

CODE_TITLE_PROMPT = """\
Generate a concise technical title (5-10 words max) for this code snippet that describes its core functionality or purpose.

Code Snippet:
{code_chunk}

Concise Title:"""

CODE_DESC_PROMPT = """\
Write a brief technical description (1-3 sentences) of what this code snippet does and its main purpose. Focus on functionality.

Code Snippet:
{code_chunk}
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
Generate exactly {num_questions} specific questions a developer might ask about the functionality or usage of this code snippet, based *only* on the provided code. Separate each question with the delimiter '|||'.

Code Snippet:
{code_chunk}

Questions:"""
