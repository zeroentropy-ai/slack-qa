#!/usr/bin/env python3
"""
Run few-shot prompted model on all test queries and save results for benchmarking
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from garbage import CHANNEL_ACTIVITY, TICKET_ACTIVITY, HELP_US_HELP

# Combine all garbage document IDs
GARBAGE_DOCUMENT_IDS = set(CHANNEL_ACTIVITY + TICKET_ACTIVITY + HELP_US_HELP)

def load_qrels(qrels_file="./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/qrels.jsonl"):
    """Load qrels for filtering garbage queries"""
    qrels_by_query_id = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            if "{" in line:
                qrel = json.loads(line)
                query_id = qrel["query_id"]
                if query_id not in qrels_by_query_id:
                    qrels_by_query_id[query_id] = []
                qrels_by_query_id[query_id].append(qrel)
    return qrels_by_query_id

def is_garbage_query(qrel_doc_ids):
    """Check if a query targets garbage documents"""
    return any(doc_id in GARBAGE_DOCUMENT_IDS for doc_id in qrel_doc_ids)

def load_test_queries(queries_file="./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/queries.jsonl"):
    """Load all test queries"""
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            if "{" in line:
                query = json.loads(line)
                queries[query["id"]] = query
    return queries

def run_single_query(query_text):
    """Run the completions API script for a single query"""
    try:
        # Run the script and capture output
        result = subprocess.run(
            ["python3", "call_completions_api.py", query_text],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Parse the output from completions API
            try:
                output_lines = result.stdout.strip().split('\n')
                
                # Find the generated search queries line
                generated_text = None
                for line in output_lines:
                    if line.startswith('[') and line.endswith(']'):
                        generated_text = line
                        break
                
                if generated_text:
                    searches = json.loads(generated_text)
                    return searches, None
                else:
                    # If no JSON found, return the raw output for debugging
                    return None, f"No JSON found in output: {result.stdout}"
                    
            except json.JSONDecodeError:
                return None, f"Invalid JSON: {result.stdout}"
        else:
            return None, f"Script error: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return None, "Timeout after 30 seconds"
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    print("Loading test queries...")
    queries = load_test_queries()
    print(f"Loaded {len(queries)} queries")
    
    print("Loading qrels...")
    qrels_by_query_id = load_qrels()
    
    # Filter out garbage queries
    non_garbage_queries = {}
    for query_id, query_data in queries.items():
        qrels = qrels_by_query_id.get(query_id, [])
        if qrels:
            qrel_doc_ids = [qrel["document_id"] for qrel in qrels]
            if not is_garbage_query(qrel_doc_ids):
                non_garbage_queries[query_id] = query_data
    
    print(f"Filtered out {len(queries) - len(non_garbage_queries)} garbage queries")
    print(f"Processing {len(non_garbage_queries)} non-garbage queries")
    queries = non_garbage_queries
    
    # Prepare output
    results = []
    start_time = datetime.now()
    
    # Process each query
    for i, (query_id, query_data) in enumerate(queries.items()):
        query_text = query_data["query"]
        print(f"\n[{i+1}/{len(queries)}] Processing: {query_text[:80]}...")
        
        # Run the model
        searches, error = run_single_query(query_text)
        
        if searches:
            print(f"  ✓ Generated {len(searches)} search queries")
            results.append({
                "query_id": query_id,
                "query": query_text,
                "generated_searches": searches,
                "status": "success"
            })
        else:
            print(f"  ✗ Failed: {error}")
            results.append({
                "query_id": query_id,
                "query": query_text,
                "generated_searches": [],
                "status": "error",
                "error": error
            })
        
    
    # Final save
    output_file = f"finetune_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Simple statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    
    print(f"\nCompleted: {success_count}/{len(queries)} successful")
    print(f"Results saved to: {output_file}")
    
if __name__ == "__main__":
    main()
