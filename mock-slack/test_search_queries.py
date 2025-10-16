#!/usr/bin/env python3
"""
Test search queries against Solr using masked search and record results.
Takes search_queries_step_0.json and augments with Solr search results.
"""
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from masked_solr_library import masked_solr_search

def load_search_queries(file_path: str = "search_queries_step_0.json") -> List[Dict[str, Any]]:
    """Load search queries from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def test_single_query(search_query: str, target_document_id: str, collection: str = "train_data") -> Dict[str, Any]:
    """
    Test a single search query against Solr and return results.
    
    Returns:
        Dict with:
        - solr_results: List of document IDs returned by search
        - target_rank: Rank of target document (1-indexed, None if not found)
        - keywords_matched: Number of keywords that matched
        - total_results: Total number of results returned
    """
    try:
        # Use masked Solr search
        docs, keywords_matched = masked_solr_search(search_query, collection, rows=100)
        
        # Extract document IDs from results
        doc_ids = [doc['id'] for doc in docs]
        
        # Find rank of target document
        target_rank = None
        for i, doc_id in enumerate(doc_ids, 1):
            if doc_id == target_document_id:
                target_rank = i
                break
        
        return {
            "solr_results": doc_ids,
            "target_rank": target_rank,
            "keywords_matched": keywords_matched,
            "total_results": len(doc_ids)
        }
        
    except Exception as e:
        print(f"Error testing query '{search_query}': {e}")
        return {
            "solr_results": [],
            "target_rank": None,
            "keywords_matched": 0,
            "total_results": 0,
            "error": str(e)
        }

def save_run_metadata(
    results: List[Dict[str, Any]], 
    collection: str,
    input_file: str,
    output_file: str
):
    """Save metadata about this run including prompts and stats"""
    import time
    from datetime import datetime
    
    # Calculate summary statistics
    total_queries = len(results)
    found_count = sum(1 for r in results if r['target_rank'] is not None)
    found_rate = found_count / total_queries if total_queries > 0 else 0
    
    # Rank distribution for found documents
    ranks = [r['target_rank'] for r in results if r['target_rank'] is not None]
    rank_stats = {}
    if ranks:
        rank_stats = {
            "rank_1": sum(1 for r in ranks if r == 1),
            "rank_1_to_5": sum(1 for r in ranks if r <= 5),
            "rank_1_to_10": sum(1 for r in ranks if r <= 10),
            "rank_1_to_20": sum(1 for r in ranks if r <= 20),
            "average_rank": sum(ranks) / len(ranks),
            "median_rank": sorted(ranks)[len(ranks)//2]
        }
    
    # Keywords matched distribution
    keywords_stats = {}
    all_keywords = [r['keywords_matched'] for r in results]
    if all_keywords:
        for k in range(1, max(all_keywords) + 1):
            count = sum(1 for km in all_keywords if km == k)
            if count > 0:
                keywords_stats[f"{k}_keywords"] = count
    
    # Per-question performance
    query_performance = {}
    for entry in results:
        query_id = entry['query_id']
        if query_id not in query_performance:
            query_performance[query_id] = {
                'original_question': entry['original_user_query'],
                'total_queries': 0,
                'found_queries': 0,
                'best_rank': None
            }
        
        perf = query_performance[query_id]
        perf['total_queries'] += 1
        
        if entry['target_rank'] is not None:
            perf['found_queries'] += 1
            if perf['best_rank'] is None or entry['target_rank'] < perf['best_rank']:
                perf['best_rank'] = entry['target_rank']
    
    # Calculate question-level success rates
    question_success_rates = []
    for perf in query_performance.values():
        success_rate = perf['found_queries'] / perf['total_queries']
        question_success_rates.append(success_rate)
    
    # OpenAI prompts used in previous step (from generate_search_queries.py)
    search_query_generation_prompt = """Given a user question and the target document that should answer it, generate 10-20 diverse Slack search queries that someone might use to find this information.

The search queries should be:
- Short keyword phrases (1-4 words typically)
- What someone would actually type in Slack search
- Diverse approaches to finding the same information
- Technical terms, error messages, key concepts from the document

User Question: {question}

Target Document Content: {document_content[:800]}

Generate search queries as a JSON list of strings. Focus on different ways someone might search for this information:"""
    
    # Metadata object
    metadata = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "input_file": input_file,
            "output_file": output_file,
            "collection": collection,
            "total_queries_tested": total_queries
        },
        "openai_prompts": {
            "search_query_generation": {
                "template": search_query_generation_prompt,
                "model": "gpt-3.5-turbo",
                "temperature": 0.8,
                "max_tokens": 500,
                "description": "Prompt used to generate search queries from user questions"
            }
        },
        "summary_statistics": {
            "overall": {
                "total_queries": total_queries,
                "documents_found": found_count,
                "found_rate": found_rate,
                "avg_results_per_query": sum(r['total_results'] for r in results) / total_queries if total_queries > 0 else 0
            },
            "rank_distribution": rank_stats,
            "keywords_matched_distribution": keywords_stats,
            "question_level_performance": {
                "total_questions": len(query_performance),
                "avg_success_rate": sum(question_success_rates) / len(question_success_rates) if question_success_rates else 0,
                "questions_with_any_success": sum(1 for rate in question_success_rates if rate > 0),
                "questions_with_perfect_success": sum(1 for rate in question_success_rates if rate == 1.0)
            }
        },
        "top_performing_questions": []
    }
    
    # Add top 5 performing questions
    sorted_performance = sorted(
        query_performance.items(),
        key=lambda x: (x[1]['found_queries'] / x[1]['total_queries'], -x[1]['best_rank'] if x[1]['best_rank'] else 999),
        reverse=True
    )
    
    for query_id, perf in sorted_performance[:5]:
        metadata["top_performing_questions"].append({
            "query_id": query_id,
            "question": perf['original_question'],
            "success_rate": perf['found_queries'] / perf['total_queries'],
            "best_rank": perf['best_rank'],
            "total_queries": perf['total_queries'],
            "found_queries": perf['found_queries']
        })
    
    # Save metadata
    metadata_file = output_file.replace('.json', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📊 Run metadata saved to {metadata_file}")

def test_search_queries(
    input_file: str = "search_queries_step_0.json",
    output_file: str = "search_queries_with_results.json",
    collection: str = "train_data"
) -> List[Dict[str, Any]]:
    """
    Test all search queries and augment with Solr results.
    
    Args:
        input_file: Input file with search queries in step_0 format
        output_file: Output file with results added
        collection: Solr collection to search in
    """
    
    print(f"Loading search queries from {input_file}...")
    search_queries = load_search_queries(input_file)
    
    print(f"Loaded {len(search_queries):,} search queries")
    print(f"Testing against Solr collection: {collection}")
    
    results = []
    found_count = 0
    total_results = 0
    
    for entry in tqdm(search_queries, desc="Testing search queries"):
        # Copy original entry
        result_entry = entry.copy()
        
        # Test the search query
        search_results = test_single_query(
            entry['search_query'],
            entry['target_document_id'],
            collection
        )
        
        # Merge results into the entry
        result_entry.update(search_results)
        
        results.append(result_entry)
        
        # Update statistics
        if search_results['target_rank'] is not None:
            found_count += 1
        total_results += search_results['total_results']
    
    # Calculate statistics
    found_rate = found_count / len(search_queries) if search_queries else 0
    avg_results = total_results / len(search_queries) if search_queries else 0
    
    print(f"\nResults Summary:")
    print(f"Total queries tested: {len(search_queries):,}")
    print(f"Target documents found: {found_count:,} ({found_rate*100:.1f}%)")
    print(f"Average results per query: {avg_results:.1f}")
    
    # Rank distribution for found documents
    ranks = [r['target_rank'] for r in results if r['target_rank'] is not None]
    if ranks:
        print(f"\nRank distribution (found documents):")
        print(f"  Rank 1: {sum(1 for r in ranks if r == 1)} ({100*sum(1 for r in ranks if r == 1)/len(ranks):.1f}%)")
        print(f"  Rank 1-5: {sum(1 for r in ranks if r <= 5)} ({100*sum(1 for r in ranks if r <= 5)/len(ranks):.1f}%)")
        print(f"  Rank 1-10: {sum(1 for r in ranks if r <= 10)} ({100*sum(1 for r in ranks if r <= 10)/len(ranks):.1f}%)")
        print(f"  Rank 1-20: {sum(1 for r in ranks if r <= 20)} ({100*sum(1 for r in ranks if r <= 20)/len(ranks):.1f}%)")
    
    # Keywords matched distribution
    keywords_stats = [r['keywords_matched'] for r in results]
    if keywords_stats:
        print(f"\nKeywords matched distribution:")
        for k in range(1, max(keywords_stats) + 1):
            count = sum(1 for km in keywords_stats if km == k)
            if count > 0:
                print(f"  {k} keywords: {count} queries ({100*count/len(keywords_stats):.1f}%)")
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved with {len(results)} entries")
    
    # Save run metadata
    save_run_metadata(results, collection, input_file, output_file)
    
    # Show some examples
    print(f"\nExample results:")
    for i, entry in enumerate(results[:3], 1):
        print(f"\n{i}. Query: '{entry['search_query']}'")
        print(f"   Target Document: {entry['target_document_id']}")
        print(f"   Rank: {entry['target_rank']}")
        print(f"   Keywords Matched: {entry['keywords_matched']}")
        print(f"   Total Results: {entry['total_results']}")
        print(f"   First 3 Results: {entry['solr_results'][:3]}")
    
    return results

def analyze_query_performance(results: List[Dict[str, Any]]):
    """Analyze performance by query_id to see which questions work best"""
    
    query_performance = {}
    
    for entry in results:
        query_id = entry['query_id']
        if query_id not in query_performance:
            query_performance[query_id] = {
                'original_question': entry['original_user_query'],
                'total_queries': 0,
                'found_queries': 0,
                'best_rank': None,
                'successful_queries': []
            }
        
        perf = query_performance[query_id]
        perf['total_queries'] += 1
        
        if entry['target_rank'] is not None:
            perf['found_queries'] += 1
            if perf['best_rank'] is None or entry['target_rank'] < perf['best_rank']:
                perf['best_rank'] = entry['target_rank']
            perf['successful_queries'].append({
                'search_query': entry['search_query'],
                'rank': entry['target_rank']
            })
    
    # Sort by success rate
    sorted_performance = sorted(
        query_performance.items(),
        key=lambda x: x[1]['found_queries'] / x[1]['total_queries'],
        reverse=True
    )
    
    print(f"\nTop performing questions (by success rate):")
    for i, (query_id, perf) in enumerate(sorted_performance[:5], 1):
        success_rate = perf['found_queries'] / perf['total_queries']
        print(f"\n{i}. Success Rate: {success_rate*100:.1f}% ({perf['found_queries']}/{perf['total_queries']})")
        print(f"   Question: {perf['original_question']}")
        print(f"   Best Rank: {perf['best_rank']}")
        if perf['successful_queries']:
            print(f"   Best Queries: {[q['search_query'] for q in perf['successful_queries'][:3]]}")

def main():
    print("🔍 Search Query Tester")
    print("=" * 50)
    
    # Configuration
    input_file = "search_queries_step_0.json"
    collection = input("Solr collection name (default 'train_data'): ").strip() or "train_data"
    
    # Check if input file exists
    import os
    if not os.path.exists(input_file):
        print(f"❌ Input file {input_file} not found!")
        print("Please run generate_search_queries.py first.")
        return
    
    try:
        results = test_search_queries(collection=collection)
        
        # Additional analysis
        analyze_query_performance(results)
        
        print(f"\n✅ Successfully tested all search queries!")
        
    except KeyboardInterrupt:
        print("\n❌ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")

if __name__ == "__main__":
    main()