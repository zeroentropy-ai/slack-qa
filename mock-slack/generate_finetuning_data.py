#!/usr/bin/env python3
"""
Generate fine-tuning data from search query test results.
For each question, extract the top 5 performing search queries.
"""
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict

def load_search_results(file_path: str = "search_queries_with_results.json") -> List[Dict[str, Any]]:
    """Load search query test results"""
    with open(file_path, 'r') as f:
        return json.load(f)

def group_by_question(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group search results by query_id (question)"""
    grouped = defaultdict(list)
    
    for result in results:
        query_id = result['query_id']
        grouped[query_id].append(result)
    
    return dict(grouped)

def get_recall_at_20_queries(query_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get all search queries that have recall@20 = 1 (target found in top 20).
    
    Returns list of query results that found the target within top 20.
    """
    recall_20_queries = []
    
    for result in query_results:
        target_rank = result['target_rank']
        
        # Check if target was found within top 20 (recall@20 = 1)
        if target_rank is not None and target_rank <= 20:
            recall_20_queries.append(result)
    
    return recall_20_queries

def generate_finetuning_data(
    input_file: str = "search_queries_with_results.json",
    output_file: str = "finetuning_data.json",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate fine-tuning data from search query test results.
    
    For each question, extracts the top-k performing search queries.
    
    Output format:
    {
        "question": "How do I fix Modal GPU errors?",
        "successful_search_queries": ["modal gpu error", "gpu memory", "modal timeout"],
        "query_id": "uuid-here",
        "performance_stats": {
            "total_queries_tested": 15,
            "successful_queries": 3,
            "best_rank": 2,
            "success_rate": 0.2
        }
    }
    """
    
    print(f"Loading search results from {input_file}...")
    results = load_search_results(input_file)
    
    print(f"Loaded {len(results):,} search query results")
    
    # Group by question
    grouped_results = group_by_question(results)
    
    print(f"Found {len(grouped_results)} unique questions")
    
    finetuning_data = []
    
    for query_id, query_results in grouped_results.items():
        # Get the question (same for all results in this group)
        question = query_results[0]['original_user_query']
        
        # Get all queries with recall@20 = 1
        recall_20_queries = get_recall_at_20_queries(query_results)
        
        # Randomly select up to top_k queries from recall@20 = 1 queries for diversity
        import random
        if len(recall_20_queries) > top_k:
            selected_queries = random.sample(recall_20_queries, top_k)
        else:
            selected_queries = recall_20_queries
        
        # Extract just the search query strings
        successful_queries = [result['search_query'] for result in selected_queries]
        
        # Calculate performance stats
        total_queries = len(query_results)
        successful_count = sum(1 for r in query_results if r['target_rank'] is not None)
        best_rank = min((r['target_rank'] for r in query_results if r['target_rank'] is not None), default=None)
        success_rate = successful_count / total_queries if total_queries > 0 else 0
        
        # Only include questions that had at least one successful query
        if successful_queries:
            finetuning_entry = {
                "question": question,
                "successful_search_queries": successful_queries,
                "query_id": query_id,
                "performance_stats": {
                    "total_queries_tested": total_queries,
                    "successful_queries": successful_count,
                    "best_rank": best_rank,
                    "success_rate": success_rate
                },
                "top_query_details": [
                    {
                        "search_query": result['search_query'],
                        "rank": result['target_rank'],
                        "keywords_matched": result['keywords_matched'],
                        "score": result.get('score', 0)
                    }
                    for result in selected_queries
                ]
            }
            
            finetuning_data.append(finetuning_entry)
    
    # Sort by success rate (best performing questions first)
    finetuning_data.sort(key=lambda x: x['performance_stats']['success_rate'], reverse=True)
    
    print(f"\nGenerated fine-tuning data:")
    print(f"Questions with successful queries: {len(finetuning_data)}")
    
    if finetuning_data:
        avg_success_rate = sum(entry['performance_stats']['success_rate'] for entry in finetuning_data) / len(finetuning_data)
        avg_successful_queries = sum(len(entry['successful_search_queries']) for entry in finetuning_data) / len(finetuning_data)
        
        print(f"Average success rate: {avg_success_rate:.1%}")
        print(f"Average successful queries per question: {avg_successful_queries:.1f}")
        
        # Show distribution of successful query counts
        query_counts = [len(entry['successful_search_queries']) for entry in finetuning_data]
        for count in range(1, top_k + 1):
            num_questions = sum(1 for qc in query_counts if qc >= count)
            print(f"Questions with {count}+ successful queries: {num_questions} ({100*num_questions/len(finetuning_data):.1f}%)")
    
    # Save fine-tuning data
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(finetuning_data, f, indent=2)
    
    print(f"✅ Fine-tuning data saved with {len(finetuning_data)} questions")
    
    # Show examples
    print(f"\nExample fine-tuning entries:")
    for i, entry in enumerate(finetuning_data[:3], 1):
        print(f"\n{i}. Question: {entry['question']}")
        print(f"   Success Rate: {entry['performance_stats']['success_rate']:.1%}")
        print(f"   Best Rank: {entry['performance_stats']['best_rank']}")
        print(f"   Successful Queries: {entry['successful_search_queries']}")
    
    return finetuning_data

def convert_to_training_format(
    finetuning_data: List[Dict[str, Any]], 
    output_file: str = "training_pairs.jsonl"
) -> None:
    """
    Convert fine-tuning data to training format (JSONL with input/output pairs)
    """
    
    print(f"\nConverting to training format...")
    
    training_pairs = []
    
    for entry in finetuning_data:
        question = entry['question']
        search_queries = entry['successful_search_queries']
        
        # Create training pair
        training_pair = {
            "input": f"Question: {question}\n\nGenerate search queries:",
            "output": json.dumps(search_queries),
            "metadata": {
                "query_id": entry['query_id'],
                "success_rate": entry['performance_stats']['success_rate'],
                "best_rank": entry['performance_stats']['best_rank']
            }
        }
        
        training_pairs.append(training_pair)
    
    # Save as JSONL
    with open(output_file, 'w') as f:
        for pair in training_pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"✅ Training pairs saved to {output_file} ({len(training_pairs)} pairs)")

def main():
    print("🎯 Fine-tuning Data Generator")
    print("=" * 50)
    
    # Configuration
    input_file = "search_queries_with_results.json"
    top_k = int(input("Number of top search queries per question (default 5): ") or "5")
    
    # Check if input file exists
    import os
    if not os.path.exists(input_file):
        print(f"❌ Input file {input_file} not found!")
        print("Please run test_search_queries.py first.")
        return
    
    try:
        # Generate fine-tuning data
        finetuning_data = generate_finetuning_data(top_k=top_k)
        
        # Convert to training format
        convert_to_training_format(finetuning_data)
        
        print(f"\n✅ Successfully generated fine-tuning data!")
        
    except KeyboardInterrupt:
        print("\n❌ Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")

if __name__ == "__main__":
    main()