#!/usr/bin/env python3
"""
Index Slack documents into Solr for mock search API
"""
import json
import requests
from tqdm import tqdm

SOLR_URL = "http://localhost:8983/solr/slack"
DOCUMENTS_FILE = "../synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/documents.jsonl"

def index_documents():
    # Load documents
    documents = []
    with open(DOCUMENTS_FILE, 'r') as f:
        for line in f:
            if "{" in line:
                doc = json.loads(line)
                documents.append({
                    "id": doc["id"],
                    "content": doc["content"]
                })
    
    print(f"Loaded {len(documents)} documents")
    
    # Index in batches
    batch_size = 100
    total_indexed = 0
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Indexing"):
        batch = documents[i:i + batch_size]
        
        # Prepare Solr update request
        solr_docs = []
        for doc in batch:
            solr_docs.append({
                "id": doc["id"],
                "content": doc["content"]
            })
        
        # Send to Solr
        response = requests.post(
            f"{SOLR_URL}/update/json",
            json=solr_docs,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            total_indexed += len(batch)
        else:
            print(f"Error indexing batch: {response.text}")
            break
    
    # Commit the changes
    requests.post(f"{SOLR_URL}/update?commit=true")
    
    print(f"Successfully indexed {total_indexed} documents")

if __name__ == "__main__":
    index_documents()