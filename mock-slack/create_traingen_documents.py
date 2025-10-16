#!/usr/bin/env python3
"""
Extract documents from selected datasets to create traingen_documents.json
"""
import json
from pathlib import Path
from tqdm import tqdm

# Selected datasets for ~100K documents
SELECTED_DATASETS = [
    "messirve",
    "cprecom", 
    "clerc",
    "bioasq",
    "cosqa"
]

def extract_documents():
    """Extract all documents from selected datasets"""
    all_documents = []
    datasets_dir = Path("datasets")
    
    for dataset_name in SELECTED_DATASETS:
        dataset_path = datasets_dir / dataset_name
        ai_scores_file = dataset_path / "ai_scores.json"
        
        print(f"Processing {dataset_name}...")
        
        if not ai_scores_file.exists():
            print(f"  WARNING: {ai_scores_file} not found!")
            continue
        
        with open(ai_scores_file, 'r') as f:
            data = json.load(f)
        
        scored_pairs = data.get("scored_pairs", [])
        dataset_docs = 0
        
        for pair_data in tqdm(scored_pairs, desc=f"  Extracting from {dataset_name}"):
            pair = pair_data["pair"]
            
            # Extract document_a
            doc_a = pair["document_a"]
            all_documents.append({
                "id": doc_a["document_id"],
                "content": doc_a["content"],
                "metadata": {
                    "dataset": dataset_name,
                    "title": doc_a.get("metadata", {}).get("title", ""),
                    "query_context": pair["query"]  # Keep query for context
                }
            })
            dataset_docs += 1
            
            # Extract document_b
            doc_b = pair["document_b"]
            all_documents.append({
                "id": doc_b["document_id"],
                "content": doc_b["content"],
                "metadata": {
                    "dataset": dataset_name,
                    "title": doc_b.get("metadata", {}).get("title", ""),
                    "query_context": pair["query"]  # Keep query for context
                }
            })
            dataset_docs += 1
        
        print(f"  Extracted {dataset_docs:,} documents from {dataset_name}")
    
    print(f"\nTotal documents extracted: {len(all_documents):,}")
    
    # Save to JSON
    print("Saving to traingen_documents.json...")
    with open("traingen_documents.json", 'w') as f:
        json.dump(all_documents, f, indent=2)
    
    print(f"Successfully created traingen_documents.json with {len(all_documents):,} documents")
    
    # Show dataset distribution
    print(f"\nDataset distribution:")
    dataset_counts = {}
    for doc in all_documents:
        dataset = doc["metadata"]["dataset"]
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    for dataset, count in sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dataset:<15} {count:>8,} docs")
    
    return all_documents

if __name__ == "__main__":
    documents = extract_documents()