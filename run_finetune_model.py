#!/usr/bin/env python3
"""
Run few-shot prompted model on all test queries and save results for benchmarking
Can use either local finetuned model or OpenAI API with --openai flag
"""

import json
import argparse
import asyncio
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm
from garbage import CHANNEL_ACTIVITY, TICKET_ACTIVITY, HELP_US_HELP
from query_generators import generate_queries_local, generate_queries_openai

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

async def process_query_async(semaphore: asyncio.Semaphore, query_id: str, query_text: str, use_openai: bool, model_type: str) -> Dict:
    """Process a single query asynchronously with semaphore control"""
    async with semaphore:
        if use_openai:
            # Run OpenAI in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            searches, error = await loop.run_in_executor(None, generate_queries_openai, query_text)
        else:
            # Run local model in thread pool
            loop = asyncio.get_event_loop()
            searches, error = await loop.run_in_executor(None, generate_queries_local, query_text)
        
        if searches:
            return {
                "query_id": query_id,
                "query": query_text,
                "generated_searches": searches,
                "status": "success",
                "model_type": model_type
            }
        else:
            return {
                "query_id": query_id,
                "query": query_text,
                "generated_searches": [],
                "status": "error",
                "error": error,
                "model_type": model_type
            }

def run_single_query(query_text: str, use_openai: bool = False) -> Tuple[List[str], str]:
    """
    Legacy function for backward compatibility (synchronous version)
    """
    if use_openai:
        return generate_queries_openai(query_text)
    else:
        return generate_queries_local(query_text)

async def main_async():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run few-shot prompted model on test queries")
    parser.add_argument("--openai", action="store_true", 
                       help="Use OpenAI API instead of local finetuned model")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of queries to process (for testing)")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Use parallel processing (default: True)")
    args = parser.parse_args()
    
    model_type = "OpenAI" if args.openai else "Finetuned"
    print(f"Using {model_type} model...")
    
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
    
    # Apply limit if specified
    if args.limit:
        query_items = list(queries.items())[:args.limit]
        queries = dict(query_items)
        print(f"Limited to first {len(queries)} queries for testing")
    
    start_time = datetime.now()
    
    if args.parallel and args.openai:
        # Use parallel processing with semaphore for OpenAI
        print(f"Using parallel processing with semaphore of 16...")
        semaphore = asyncio.Semaphore(16)
        
        # Create tasks for all queries
        tasks = []
        for query_id, query_data in queries.items():
            query_text = query_data["query"]
            task = process_query_async(semaphore, query_id, query_text, args.openai, model_type)
            tasks.append(task)
        
        # Process all queries with progress bar
        results = []
        with tqdm(total=len(tasks), desc="Processing queries") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
                
                # Show status in progress bar
                if result["status"] == "success":
                    pbar.set_postfix({"✓": f"{len(result['generated_searches'])} queries"})
                else:
                    pbar.set_postfix({"✗": "Failed"})
    
    else:
        # Use sequential processing (for local model or when parallel is disabled)
        print("Using sequential processing...")
        results = []
        
        with tqdm(total=len(queries), desc="Processing queries") as pbar:
            for query_id, query_data in queries.items():
                query_text = query_data["query"]
                
                # Run the model
                searches, error = run_single_query(query_text, use_openai=args.openai)
                
                if searches:
                    results.append({
                        "query_id": query_id,
                        "query": query_text,
                        "generated_searches": searches,
                        "status": "success",
                        "model_type": model_type
                    })
                    pbar.set_postfix({"✓": f"{len(searches)} queries"})
                else:
                    results.append({
                        "query_id": query_id,
                        "query": query_text,
                        "generated_searches": [],
                        "status": "error",
                        "error": error,
                        "model_type": model_type
                    })
                    pbar.set_postfix({"✗": "Failed"})
                
                pbar.update(1)
    
    # Final save with model type in filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_suffix = "openai" if args.openai else "finetuned"
    output_file = f"{model_suffix}_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Simple statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    elapsed_time = datetime.now() - start_time
    
    print(f"\nCompleted: {success_count}/{len(queries)} successful")
    print(f"Elapsed time: {elapsed_time}")
    print(f"Results saved to: {output_file}")
    
    if args.openai:
        print(f"\n💡 To compare with finetuned model, run:")
        print(f"   python run_finetune_model.py --limit {len(queries)}")
    else:
        print(f"\n💡 To compare with OpenAI model, run:")
        print(f"   python run_finetune_model.py --openai --limit {len(queries)}")

def main():
    """Synchronous wrapper for async main"""
    asyncio.run(main_async())
    
if __name__ == "__main__":
    main()