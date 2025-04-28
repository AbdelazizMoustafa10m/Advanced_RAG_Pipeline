import json
import logging

logging.basicConfig(level=logging.DEBUG)

from llama_index.core import Document
from llama_index.readers.docling import DoclingReader

# Initialize the reader
reader = DoclingReader()

# Load the PDF document
docs = reader.load_data(file_path=["./data/FBL_Validation_Strategies.pdf"])

# Print metadata for debugging
print(f"Number of documents loaded: {len(docs)}")
if docs:
    # Extract the first document for inspection
    doc = docs[0]
    print(f"Document ID: {doc.doc_id}")
    print(f"Metadata keys: {list(doc.metadata.keys())}")
    
    # Print specific metadata fields we care about
    print("Important fields:")
    for field in ["origin", "schema_name", "file_path", "source"]:
        if field in doc.metadata:
            value = doc.metadata[field]
            if isinstance(value, dict):
                print(f"{field}: {json.dumps(value, indent=2)}")
            else:
                print(f"{field}: {value}")
    
    # Check if origin contains filename
    if "origin" in doc.metadata:
        origin = doc.metadata["origin"]
        if isinstance(origin, dict) and "filename" in origin:
            print(f"Found filename in origin: {origin['filename']}")
