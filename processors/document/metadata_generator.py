# processors/document/metadata_generator.py
import asyncio
import json
import logging
import os
import re
from typing import List, Dict, Any, Optional, Sequence
from llama_index.core.schema import BaseNode, TextNode, MetadataMode
from llama_index.core.llms import LLM
from core.interfaces import IMetadataEnricher # Adjust import
from llm.prompts import DOC_TITLE_PROMPT, DOC_SUMMARY_PROMPT, DOC_QUESTIONS_PROMPT # Central prompts
from core.models import QAPair # Central data model

logger = logging.getLogger(__name__)

class DoclingMetadataGenerator(IMetadataEnricher):
    """Enriches document nodes (chunked by Docling) using LLMs."""

    def __init__(self, llm: LLM, num_questions: int = 2):
        self.llm = llm
        self.num_questions = num_questions

    def supports_node_type(self, node_type: str) -> bool:
        # This enricher works on document nodes
        return node_type.lower() == "document"

    def _format_context(self, node: BaseNode) -> Dict[str, Any]:
        """Safely extracts text and relevant Docling metadata for prompts."""
        # Reusing the formatting logic from your CustomDoclingEnricher
        text_chunk = node.get_content(metadata_mode=MetadataMode.NONE)

        headings = node.metadata.get('headings', [])
        headings_str = ", ".join([str(h) for h in headings]) if isinstance(headings, list) else str(headings or "N/A")

        node_type = "text" # Default
        doc_items = node.metadata.get('doc_items', []) # Specific to DoclingNodeParser output
        try:
            if isinstance(doc_items, list) and len(doc_items) > 0 and isinstance(doc_items[0], dict):
                 node_type = doc_items[0].get('label', 'text')
        except Exception:
            node_type = "text" # Fallback

        page_label = str(node.metadata.get('page_label', 'N/A'))

        return {
            "text_chunk": text_chunk,
            "headings": headings_str,
            "node_type": node_type, # Docling label (paragraph, list_item etc)
            "page_label": page_label,
        }


        
    async def _aprocess_node(self, node: BaseNode):
        """Async processing for a single document node."""
        if not isinstance(node, TextNode) or not node.text:
            return
            
        context = self._format_context(node)
        tasks = []
            
        # --- Title Generation ---
        async def _gen_title():
            try:
                title_prompt = DOC_TITLE_PROMPT.format(**context)
                title_response = await self.llm.acomplete(title_prompt)
                node.metadata['functional_title'] = title_response.text.strip()
            except Exception as e:
                logger.error(f"Error generating doc title for node {node.node_id}: {repr(e)}")
                node.metadata['functional_title'] = "Error: Could not generate title"
        tasks.append(_gen_title())

        # --- Summary Generation ---
        async def _gen_summary():
            try:
                summary_prompt = DOC_SUMMARY_PROMPT.format(**context)
                summary_response = await self.llm.acomplete(summary_prompt)
                node.metadata['concise_summary'] = summary_response.text.strip()
            except Exception as e:
                logger.error(f"Error generating doc summary for node {node.node_id}: {repr(e)}")
                node.metadata['concise_summary'] = "Error: Could not generate summary"
        tasks.append(_gen_summary())

        # --- Q&A Generation ---
        async def _gen_qa():
            raw_llm_response_text = "Error: LLM call failed"
            try:
                qa_prompt = DOC_QUESTIONS_PROMPT.format(num_questions=self.num_questions, **context)
                try:
                    qa_response = await self.llm.acomplete(qa_prompt)
                    raw_llm_response_text = qa_response.text.strip()
                except Exception as llm_e:
                    logger.error(f"Caught Exception during Doc Q&A LLM call for node {node.node_id}: {repr(llm_e)}")
                    node.metadata['generated_questions_list'] = ["Error: LLM call failed"]
                    return

                if raw_llm_response_text and raw_llm_response_text != "Error: LLM call failed":
                    try:
                        questions_list = [q.strip() for q in raw_llm_response_text.split('|||') if q.strip()]
                        if questions_list:
                            # Store as list of strings or convert to QAPair if needed later
                            node.metadata['generated_questions_list'] = questions_list[:self.num_questions]
                        else:
                            logger.warning(f"No questions found after splitting for doc node {node.node_id}. Raw: {raw_llm_response_text}")
                            node.metadata['generated_questions_list'] = ["Error: No questions generated/parsed"]
                    except Exception as parse_e:
                         logger.error(f"Error splitting/processing Doc Q&A string for node {node.node_id}: {repr(parse_e)}. Raw: {raw_llm_response_text}")
                         node.metadata['generated_questions_list'] = ["Error: Could not process Q&A string"]
                else:
                    logger.warning(f"Empty or error response from LLM for Doc Q&A, node {node.node_id}. Raw: {raw_llm_response_text}")
                    node.metadata['generated_questions_list'] = ["Error: Empty/failed LLM response for questions"]
            except Exception as outer_e:
                logger.error(f"Unexpected outer error in Doc _gen_qa for node {node.node_id}: {repr(outer_e)}")
                node.metadata['generated_questions_list'] = ["Error: Unexpected error in Q&A generation"]
        tasks.append(_gen_qa())

        await asyncio.gather(*tasks, return_exceptions=True)
        



    async def aenrich(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        if not nodes: return []
        await asyncio.gather(*[self._aprocess_node(node) for node in nodes])
        return [node.metadata for node in nodes]

    def enrich(self, nodes: Sequence[BaseNode]) -> List[Dict]:
         if not nodes: return []
         # Simplified sync wrapper (consider proper async context handling for production)
         return asyncio.run(self.aenrich(nodes))