#!/usr/bin/env python
# test_metadata_modes.py
# Test script to verify metadata inclusion in different modes

import logging
import sys
from pathlib import Path
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.llms import OpenAI

from pipeline.orchestrator import PipelineOrchestrator
from core.config import UnifiedConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def display_node_metadata(node, index, total):
    """Display node metadata in all modes."""
    print(f"\n--- Node {index+1}/{total} ---")
    print(f"file_type: {node.metadata.get('file_type', 'N/A')}")
    print(f"document_type: {node.metadata.get('document_type', 'N/A')}")
    print(f"file_path: {node.metadata.get('file_path', 'N/A')}")
    
    # Display ALL metadata
    print("\nALL Metadata:")
    all_metadata = node.get_metadata(metadata_mode=MetadataMode.ALL)
    for key in sorted(all_metadata.keys()):
        # Skip complex objects for clean output
        if isinstance(all_metadata[key], (dict, list)) and key not in ['generated_questions_list']:
            continue
        print(f"{key}: {all_metadata[key]}")
    
    print("\n------------------------------")
    print("The LLM sees this: ")
    llm_metadata = node.get_metadata(metadata_mode=MetadataMode.LLM)
    for key in sorted(llm_metadata.keys()):
        if isinstance(llm_metadata[key], (dict, list)) and key not in ['generated_questions_list']:
            continue
        print(f"{key}: {llm_metadata[key]}")
    
    print("\n------------------------------")
    print("The Embedding model sees this: ")
    embed_metadata = node.get_metadata(metadata_mode=MetadataMode.EMBED)
    for key in sorted(embed_metadata.keys()):
        if isinstance(embed_metadata[key], (dict, list)) and key not in ['generated_questions_list']:
            continue
        print(f"{key}: {embed_metadata[key]}")
    print("------------------------------")

def main():
    # Load default config
    config = UnifiedConfig()
    
    # Check for input directory argument
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = "./data"  # Default directory
    
    config.input_directory = input_dir
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    # Run the pipeline
    processed_nodes = orchestrator.run()
    
    if not processed_nodes:
        logger.error("No nodes were processed!")
        return
    
    # Display metadata for each node
    total_nodes = len(processed_nodes)
    for i, node in enumerate(processed_nodes):
        display_node_metadata(node, i, total_nodes)
        
        # Limit to first 10 nodes if there are many
        if i >= 9:  # 0-indexed, so this is the 10th node
            print(f"\nShowing 10/{total_nodes} nodes. Run with --all to see all nodes.")
            break

if __name__ == "__main__":
    main()
