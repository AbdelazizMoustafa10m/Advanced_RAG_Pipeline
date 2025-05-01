# processors/document/metadata_generator.py
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Sequence
from llama_index.core.schema import BaseNode, TextNode, MetadataMode
from llama_index.core.llms import LLM
from core.interfaces import IMetadataEnricher
from llm.prompts import DOC_TITLE_PROMPT, DOC_SUMMARY_PROMPT, DOC_QUESTIONS_PROMPT
from core.models import QAPair

logger = logging.getLogger(__name__)

class DoclingMetadataGenerator(IMetadataEnricher):
    """Enriches document nodes (chunked by Docling) using LLMs."""

    def __init__(self, llm: LLM, num_questions: int = 2):
        self.llm = llm
        self.num_questions = num_questions

    def supports_node_type(self, node_type: str) -> bool:
        return node_type.lower() == "document"

    def _format_context(self, node: BaseNode) -> Dict[str, Any]:
        """
        Extracts text and *formatted* metadata (created by 
        CustomDoclingNodeFormatter) for prompts.
        """
        text_chunk = node.get_content(metadata_mode=MetadataMode.NONE)
        metadata = node.metadata or {}

        # Fetch the formatted keys created by the formatter transform
        context = {
            "text_chunk": text_chunk,
            "formatted_source": metadata.get("formatted_source", "Source: Unknown"),
            "formatted_location": metadata.get("formatted_location", "Location: Unknown"),
            "formatted_headings": metadata.get("formatted_headings", "Section: Unknown"),
            "formatted_label": metadata.get("formatted_label", "ContentType: Unknown"),
            # Include num_questions for the QA prompt later
            "num_questions": self.num_questions 
        }
        logger.debug(f"Formatted context for Docling LLM prompt (Node {node.node_id}): {context}")
        return context

    async def _aprocess_node(self, node: BaseNode):
        """Async processing for a single document node."""
        if not isinstance(node, TextNode) or not node.text or self.llm is None:
             logger.debug(f"Skipping enrichment for node {node.node_id}: Not a TextNode, no text, or no LLM.")
             return # Don't add error metadata here, let it pass through
            
        context = self._format_context(node)
        if not context.get("text_chunk"): # Double check if text is empty after formatting
             logger.debug(f"Skipping enrichment for node {node.node_id}: Empty text chunk.")
             return

        tasks = []
            
        # --- Title Generation ---
        async def _gen_title():
            try:
                # Use the UPDATED prompt with formatted context keys
                title_prompt = DOC_TITLE_PROMPT.format(**context) 
                title_response = await self.llm.acomplete(title_prompt)
                # Use update for async safety
                node.metadata.update({'functional_title': title_response.text.strip()}) 
            except Exception as e:
                logger.error(f"Error generating doc title for node {node.node_id}: {repr(e)}")
                node.metadata.update({'functional_title': "Error: Title Generation Failed"})
        tasks.append(_gen_title())

        # --- Summary Generation ---
        async def _gen_summary():
            try:
                 # Use the UPDATED prompt with formatted context keys
                summary_prompt = DOC_SUMMARY_PROMPT.format(**context) 
                summary_response = await self.llm.acomplete(summary_prompt)
                node.metadata.update({'concise_summary': summary_response.text.strip()})
            except Exception as e:
                logger.error(f"Error generating doc summary for node {node.node_id}: {repr(e)}")
                node.metadata.update({'concise_summary': "Error: Summary Generation Failed"})
        tasks.append(_gen_summary())

        # --- Q&A Generation ---
        async def _gen_qa():
            raw_llm_response_text = "Error: LLM call failed" # Default
            try:
                # Use the UPDATED prompt with formatted context keys
                qa_prompt = DOC_QUESTIONS_PROMPT.format(**context) 
                try:
                    qa_response = await self.llm.acomplete(qa_prompt)
                    raw_llm_response_text = qa_response.text.strip()
                except Exception as llm_e:
                    logger.error(f"Caught Exception during Doc Q&A LLM call for node {node.node_id}: {repr(llm_e)}")
                    node.metadata.update({'generated_questions_list': ["Error: LLM call failed"]})
                    return

                if raw_llm_response_text and raw_llm_response_text != "Error: LLM call failed":
                    try:
                        # Split questions using the specified delimiter
                        questions_list = [q.strip() for q in raw_llm_response_text.split('|||') if q.strip()]
                        if questions_list:
                            node.metadata.update({'generated_questions_list': questions_list[:self.num_questions]})
                        else:
                            logger.warning(f"No questions found after splitting for doc node {node.node_id}. Raw: {raw_llm_response_text}")
                            node.metadata.update({'generated_questions_list': ["Error: No questions generated/parsed"]})
                    except Exception as parse_e:
                         logger.error(f"Error splitting/processing Doc Q&A string for node {node.node_id}: {repr(parse_e)}. Raw: {raw_llm_response_text}")
                         node.metadata.update({'generated_questions_list': ["Error: Could not process Q&A string"]})
                else:
                    logger.warning(f"Empty or error response from LLM for Doc Q&A, node {node.node_id}. Raw: {raw_llm_response_text}")
                    node.metadata.update({'generated_questions_list': ["Error: Empty/failed LLM response for questions"]})
            except Exception as outer_e:
                logger.error(f"Unexpected outer error in Doc _gen_qa for node {node.node_id}: {repr(outer_e)}")
                node.metadata.update({'generated_questions_list': ["Error: Unexpected error in Q&A generation"]})
        tasks.append(_gen_qa())

        await asyncio.gather(*tasks, return_exceptions=True)
        # Metadata is updated in place on the node object

    # --- aenrich and enrich methods remain the same ---
    async def aenrich(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        if not nodes: return []
        # Filter nodes that are suitable for this enricher before processing
        supported_nodes = [node for node in nodes if self.supports_node_type(node.metadata.get("node_type", ""))]
        if not supported_nodes:
             logger.debug("DoclingMetadataGenerator: No supported document nodes found for enrichment.")
             return [node.metadata for node in nodes] # Return original metadata for all nodes
             
        logger.info(f"DoclingMetadataGenerator starting enrichment for {len(supported_nodes)} document nodes.")
        await asyncio.gather(*[self._aprocess_node(node) for node in supported_nodes])
        logger.info(f"DoclingMetadataGenerator finished enrichment.")
        return [node.metadata for node in nodes] # Return metadata for *all* nodes passed in

    def enrich(self, nodes: Sequence[BaseNode]) -> List[Dict]:
         if not nodes: return []
         # Consider using nest_asyncio if running in an environment like Jupyter
         # import nest_asyncio
         # nest_asyncio.apply()
         try:
             loop = asyncio.get_event_loop()
             if loop.is_running():
                 # If loop is running, create a new task and wait for it
                 # Note: This might not work perfectly in all nested async scenarios
                 logger.warning("Detected running event loop. Attempting to run enrich task within it.")
                 # Schedule the coroutine to run and wait for it to complete
                 future = asyncio.ensure_future(self.aenrich(nodes))
                 # This simple approach might block if the outer loop doesn't process tasks correctly.
                 # A more robust solution might involve a dedicated thread or async queue.
                 return loop.run_until_complete(future) 
             else:
                  return loop.run_until_complete(self.aenrich(nodes))
         except RuntimeError: # If no event loop exists
             return asyncio.run(self.aenrich(nodes))