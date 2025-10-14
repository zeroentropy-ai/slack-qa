#!/usr/bin/env python3
"""
Extract the best performing search calls from agent traces.
Finds up to 5 search calls that had the target document somewhere in their results.
"""

import json
import sys
from typing import List, Dict, Tuple


def extract_best_search_calls(traces_file: str = "automated_agent_traces.json", output_file: str = "best_search_calls.json"):
    """Extract the best performing search calls for each query"""
    
    # Load traces
    try:
        with open(traces_file, "r") as f:
            traces = json.load(f)
    except FileNotFoundError:
        print(f"Error: {traces_file} not found")
        return
    
    results = []
    
    for query_id, trace in traces.items():
        # Get the user query
        user_query = trace.get("query", "")
        
        # Collect all search calls that found the target document
        successful_searches = []
        
        for step in trace.get("steps", []):
            search_calls = step.get("search_calls", [])
            individual_ranks = step.get("individual_query_ranks", [])
            
            # Check each individual search call
            if search_calls and individual_ranks and len(search_calls) == len(individual_ranks):
                for search_call, rank in zip(search_calls, individual_ranks):
                    if rank > 0:  # Found the target
                        successful_searches.append({
                            "query": search_call,
                            "ground_truth_rank": rank
                        })
        
        if successful_searches:
            # Sort by rank (best first) and take top 5
            successful_searches.sort(key=lambda x: x["ground_truth_rank"])
            
            # Remove duplicates while preserving order and rank info
            seen = set()
            unique_searches = []
            for search in successful_searches:
                if search["query"] not in seen:
                    seen.add(search["query"])
                    unique_searches.append(search)
                    if len(unique_searches) >= 5:
                        break
            
            results.append({
                "query_id": query_id,
                "query": user_query,
                "ground_truth": True,  # Since we're only including queries that found the target
                "slack_queries": unique_searches
            })
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"Extracted best search calls for {len(results)} queries")
    print(f"Results saved to: {output_file}")
    
    # Show some examples
    print("\nExamples of best performing search calls:")
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. Query: {result['query']}")
        print(f"   Query ID: {result['query_id']}")
        if result['slack_queries']:
            best = result['slack_queries'][0]
            print(f"   Best call: '{best['query']}' (rank {best['ground_truth_rank']})")
            print(f"   Total successful calls: {len(result['slack_queries'])}")
    
    # Statistics
    total_calls = sum(len(r["slack_queries"]) for r in results)
    avg_calls = total_calls / len(results) if results else 0
    print(f"\nStatistics:")
    print(f"- Total queries with successful calls: {len(results)}")
    print(f"- Average successful calls per query: {avg_calls:.1f}")
    
    # Distribution of number of successful calls
    call_distribution = {}
    for result in results:
        num_calls = len(result["slack_queries"])
        call_distribution[num_calls] = call_distribution.get(num_calls, 0) + 1
    
    print(f"\nDistribution of successful calls per query:")
    for num_calls in sorted(call_distribution.keys()):
        count = call_distribution[num_calls]
        print(f"  {num_calls} calls: {count} queries")


def main():
    # Allow custom file paths via command line
    traces_file = "automated_agent_traces.json"
    output_file = "best_search_calls.json"
    
    if len(sys.argv) > 1:
        traces_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    extract_best_search_calls(traces_file, output_file)


if __name__ == "__main__":
    main()
