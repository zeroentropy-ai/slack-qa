#!/usr/bin/env python3
"""
Utility to process solr_with_freqs.json files from comparison directories
"""

import json
from pathlib import Path
from typing import Callable, Any
from solr_search import filter_like_slack

# Load training data to get target document IDs
_training_data = None
def load_training_data():
    global _training_data
    if _training_data is None:
        try:
            with open("training_data_step_0.json", 'r') as f:
                _training_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load training data: {e}")
            _training_data = []
    return _training_data

def get_target_document_id(query_id: str) -> str:
    """Get the target document ID for a given query ID"""
    training_data = load_training_data()
    for item in training_data:
        if item.get('query_id') == query_id:
            return item.get('document_id')
    return None

def find_target_rank_slack(slack_matches: list, target_doc_id: str) -> int:
    """Find the rank of target document in Slack results (1-based), 0 if not found"""
    for i, match in enumerate(slack_matches, 1):
        # Build document ID from Slack match data
        workspace = "Manifest"  # Assuming Manifest workspace
        channel_id = match.get('channel', {}).get('id', '')
        ts = match.get('ts', '')
        slack_doc_id = f"{workspace}_{channel_id}_{ts}"
        
        if slack_doc_id == target_doc_id:
            return i
    return 0

def find_target_rank_solr(solr_docs: list, target_doc_id: str) -> int:
    """Find the rank of target document in Solr results (1-based), 0 if not found"""
    for i, doc in enumerate(solr_docs, 1):
        if doc.get('id') == target_doc_id:
            return i
    return 0

def process_result_dirs(processor_func: Callable[..., Any]) -> list[Any]:
    comparison_dir = Path("comparison")
    results : list[Any] = []

    if not comparison_dir.exists():
        print("❌ comparison/ directory not found")
        return results

    processed_count = 0
    error_count = 0

    for query_dir in comparison_dir.iterdir():
        if not query_dir.is_dir() or query_dir.name.startswith('.'):
            continue

        for search_dir in query_dir.iterdir():
            if not search_dir.is_dir():
                continue

            result = processor_func(search_dir)
            if result is not None:
                results.append(result)

            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} dirs...")

    print(f"\n✅ Processed {processed_count} solr_with_freqs.json files")
    if error_count > 0:
        print(f"❌ {error_count} errors encountered")

    return results


def compare_results(search_dir: Path):
    solr_file = search_dir / "solr_with_freqs.json"
    slack_file = search_dir / "slack.json"

    if not solr_file.exists():
        # print(f"Missing solr file: {solr_file}")
        return None
    if not slack_file.exists():
        # print(f"Missing slack file: {slack_file}")
        return None

    try:
        # Load Solr and Slack results
        with open(solr_file, 'r') as f:
            solr_data = json.load(f)
        with open(slack_file, 'r') as f:
            slack_data = json.load(f)

        # Get search query from directory name
        search_query = search_dir.name.replace('_', ' ')
        
        # Get target document ID from query_id (directory parent name)
        query_id = search_dir.parent.name
        target_doc_id = get_target_document_id(query_id)
        if not target_doc_id:
            return None

        # Process Slack results
        slack_matches = slack_data.get('response', {}).get('messages', {}).get('matches', [])
        slack_target_rank = find_target_rank_slack(slack_matches, target_doc_id)

        # Process Solr results with Slack-like filtering
        solr_docs = solr_data.get('documents', [])
        filtered_docs = filter_like_slack(search_query, solr_docs)
        # Sort by score and take top 100
        filtered_docs = sorted(filtered_docs, key=lambda x: x.get('score', 0), reverse=True)[:100]
        solr_target_rank = find_target_rank_solr(filtered_docs, target_doc_id)

        return {
            "query_id": query_id,
            "search_query": search_query,
            "slack_target_rank": slack_target_rank,
            "solr_target_rank": solr_target_rank,
            "slack_total": len(slack_matches),
            "solr_total": len(filtered_docs)
        }
    except Exception as e:
        print(f"Error processing {search_dir}: {e}")
        return None

def compute_comparison():
    """
    Compute recall statistics comparing Slack and Solr search results.
    Recall@100 = 1 if target document is found in top 100 results, 0 otherwise.
    Only counts queries where both slack.json and solr_with_freqs.json exist and are valid.
    """
    print("Computing comparison statistics...")
    
    # Get all comparison results
    comparison_results = process_result_dirs(compare_results)
    total_processed = len(comparison_results)
    valid_results = [r for r in comparison_results if r is not None]
    skipped_count = total_processed - len(valid_results)
    
    print(f"📁 Total directories processed: {total_processed}")
    print(f"⚠️  Skipped (missing/invalid files): {skipped_count}")
    print(f"✅ Valid comparisons: {len(valid_results)}")
    
    if not valid_results:
        print("❌ No valid comparison results found")
        return
    
    # Count different recall scenarios
    slack_0_solr_0 = 0  # Neither found target
    slack_0_solr_1 = 0  # Only Solr found target  
    slack_1_solr_0 = 0  # Only Slack found target
    slack_1_solr_1 = 0  # Both found target
    
    for result in valid_results:
        slack_found = 1 if result['slack_target_rank'] > 0 else 0
        solr_found = 1 if result['solr_target_rank'] > 0 else 0
        
        if slack_found == 0 and solr_found == 0:
            slack_0_solr_0 += 1
        elif slack_found == 0 and solr_found == 1:
            slack_0_solr_1 += 1
        elif slack_found == 1 and solr_found == 0:
            slack_1_solr_0 += 1
        elif slack_found == 1 and solr_found == 1:
            slack_1_solr_1 += 1
    
    total = len(valid_results)
    
    print(f"\n📊 Recall@100 Comparison Results")
    print(f"📋 Based on {total} queries with both valid Slack and Solr results")
    print("=" * 60)
    print(f"Neither found target:     {slack_0_solr_0:4d} ({slack_0_solr_0/total*100:.1f}%)")
    print(f"Only Solr found target:   {slack_0_solr_1:4d} ({slack_0_solr_1/total*100:.1f}%)")
    print(f"Only Slack found target:  {slack_1_solr_0:4d} ({slack_1_solr_0/total*100:.1f}%)")
    print(f"Both found target:        {slack_1_solr_1:4d} ({slack_1_solr_1/total*100:.1f}%)")
    print("-" * 60)
    
    slack_recall = (slack_1_solr_0 + slack_1_solr_1) / total
    solr_recall = (slack_0_solr_1 + slack_1_solr_1) / total
    
    print(f"Slack Overall Recall@100: {slack_recall:.3f} ({slack_recall*100:.1f}%)")
    print(f"Solr Overall Recall@100:  {solr_recall:.3f} ({solr_recall*100:.1f}%)")
    
    if slack_recall > 0:
        improvement = (solr_recall - slack_recall) / slack_recall * 100
        print(f"Solr vs Slack improvement: {improvement:+.1f}%")
    
    return {
        'total_queries': total,
        'slack_0_solr_0': slack_0_solr_0,
        'slack_0_solr_1': slack_0_solr_1, 
        'slack_1_solr_0': slack_1_solr_0,
        'slack_1_solr_1': slack_1_solr_1,
        'slack_recall': slack_recall,
        'solr_recall': solr_recall
    }

if __name__ == "__main__":
    # Run the comparison analysis
    print("🔍 Slack vs Solr Search Comparison")
    print("=" * 50)
    
    # Compute and display comparison statistics
    stats = compute_comparison()
    
    if stats:
        print(f"\n💾 Processed {stats['total_queries']} queries total")
        
        # Show some example comparisons
        print(f"\n📋 Sample individual results:")
        sample_results = process_result_dirs(compare_results)
        valid_samples = [r for r in sample_results if r is not None][:5]
        
        for result in valid_samples:
            slack_found = "✅" if result['slack_target_rank'] > 0 else "❌"
            solr_found = "✅" if result['solr_target_rank'] > 0 else "❌"
            print(f"  Query: '{result['search_query'][:50]}...'")
            print(f"    Slack: {slack_found} (rank {result['slack_target_rank']})")
            print(f"    Solr:  {solr_found} (rank {result['solr_target_rank']})")
            print()
