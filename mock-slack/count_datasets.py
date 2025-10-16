#!/usr/bin/env python3
"""
Count documents across all datasets to plan for 100K document collection
"""
import json
import os
from pathlib import Path

def count_dataset_documents():
    """Count documents in each dataset that has ai_scores.json"""
    datasets_dir = Path("datasets")
    
    total_documents = 0
    dataset_counts = []
    
    for dataset_path in datasets_dir.iterdir():
        if dataset_path.is_dir():
            ai_scores_file = dataset_path / "ai_scores.json"
            if ai_scores_file.exists():
                try:
                    with open(ai_scores_file, 'r') as f:
                        data = json.load(f)
                    
                    scored_pairs = data.get("scored_pairs", [])
                    # Each pair has 2 documents (document_a and document_b)
                    doc_count = len(scored_pairs) * 2
                    
                    dataset_counts.append({
                        "dataset": dataset_path.name,
                        "pairs": len(scored_pairs),
                        "documents": doc_count
                    })
                    total_documents += doc_count
                    
                    print(f"{dataset_path.name:<25} {len(scored_pairs):>8,} pairs {doc_count:>10,} docs")
                    
                except Exception as e:
                    print(f"{dataset_path.name:<25} ERROR: {e}")
    
    print("=" * 60)
    print(f"{'TOTAL':<25} {total_documents//2:>8,} pairs {total_documents:>10,} docs")
    
    # Sort by document count
    dataset_counts.sort(key=lambda x: x["documents"], reverse=True)
    
    print(f"\nLargest datasets:")
    for ds in dataset_counts[:10]:
        print(f"  {ds['dataset']:<25} {ds['documents']:>8,} docs")
    
    # Find combination for ~100K documents
    print(f"\nTo get ~100K documents:")
    running_total = 0
    selected_datasets = []
    
    for ds in dataset_counts:
        if running_total < 100000:
            selected_datasets.append(ds)
            running_total += ds["documents"]
            print(f"  + {ds['dataset']:<25} {ds['documents']:>8,} docs (total: {running_total:,})")
        
        if running_total >= 100000:
            break
    
    return dataset_counts, selected_datasets

if __name__ == "__main__":
    print("DATASET DOCUMENT COUNTS")
    print("=" * 60)
    dataset_counts, selected = count_dataset_documents()
    
    print(f"\nSelected {len(selected)} datasets for training data generation")
    print(f"Total documents: {sum(ds['documents'] for ds in selected):,}")