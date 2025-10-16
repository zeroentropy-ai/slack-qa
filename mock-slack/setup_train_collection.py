#!/usr/bin/env python3
"""
Setup new Solr collection for training data and index documents
"""
import json
import requests
from tqdm import tqdm

SOLR_URL = "http://localhost:8983/solr"
COLLECTION_NAME = "train_data"

def create_collection():
    """Create a new Solr collection for training data"""
    print(f"Creating collection '{COLLECTION_NAME}'...")
    
    ## Delete collection if it exists
    #try:
    #    response = requests.get(f"{SOLR_URL}/admin/collections", params={
    #        "action": "DELETE",
    #        "name": COLLECTION_NAME
    #    })
    #    print(f"Deleted existing collection (if any)")
    #except:
    #    pass
    #
    ## Create new collection
    #response = requests.get(f"{SOLR_URL}/admin/collections", params={
    #    "action": "CREATE",
    #    "name": COLLECTION_NAME,
    #    "numShards": 1,
    #    "replicationFactor": 1,
    #    "configSet": "_default"
    #})
    #
    #if response.status_code == 200:
    #    print(f"✅ Collection '{COLLECTION_NAME}' created successfully")
    #else:
    #    print(f"❌ Failed to create collection: {response.text}")
    #    return False
    
    return True

def index_documents():
    """Index all training documents into the new collection"""
    print("Loading training documents...")
    with open("traingen_documents.json", 'r') as f:
        documents = json.load(f)
    
    print(f"Indexing {len(documents):,} documents...")
    
    # Index in batches
    batch_size = 1000
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Indexing batches"):
        batch = documents[i:i + batch_size]
        
        # Format for Solr
        solr_docs = []
        for doc in batch:
            solr_doc = {
                "id": doc["id"],
                "content": doc["content"],
                "dataset": doc["metadata"]["dataset"],
                "title": doc["metadata"].get("title", ""),
                "query_context": doc["metadata"]["query_context"]
            }
            solr_docs.append(solr_doc)
        
        # Send to Solr
        response = requests.post(
            f"{SOLR_URL}/{COLLECTION_NAME}/update/json/docs",
            json=solr_docs,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"❌ Error indexing batch {i//batch_size + 1}: {response.text}")
            return False
    
    # Commit changes
    print("Committing changes...")
    response = requests.post(f"{SOLR_URL}/{COLLECTION_NAME}/update", 
                           data="<commit/>", 
                           headers={"Content-Type": "text/xml"})
    
    if response.status_code == 200:
        print(f"✅ Successfully indexed {len(documents):,} documents")
        
        # Verify count
        response = requests.get(f"{SOLR_URL}/{COLLECTION_NAME}/select", params={"q": "*:*", "rows": 0})
        if response.status_code == 200:
            result = response.json()
            count = result["response"]["numFound"]
            print(f"✅ Verified: {count:,} documents in collection")
        
        return True
    else:
        print(f"❌ Failed to commit: {response.text}")
        return False

if __name__ == "__main__":
    print("🔧 SETTING UP TRAIN_DATA COLLECTION")
    print("=" * 50)
    
    if create_collection():
        if index_documents():
            print("\n✅ Setup complete! Collection 'train_data' is ready for use.")
        else:
            print("\n❌ Failed to index documents")
    else:
        print("\n❌ Failed to create collection")
