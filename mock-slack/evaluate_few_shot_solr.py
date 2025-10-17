#!/usr/bin/env python3
"""
Evaluate few-shot model results using Solr masked search and calculate recall@20
"""

import json
import sys
import os
import argparse
from typing import List, Dict, Any
import asyncio

from masked_solr_library import masked_solr_search
sys.path.append('..')
from ai import AIRerankModel, ai_rerank
from garbage import CHANNEL_ACTIVITY, TICKET_ACTIVITY, HELP_US_HELP

# Solr configuration
SOLR_COLLECTION = "slack"

# Rerank model configuration
RERANK_MODEL = AIRerankModel(
    company="zeroentropy",
    model="zerank-1",
)

# Combine all garbage document IDs
GARBAGE_DOCUMENT_IDS = set(CHANNEL_ACTIVITY + TICKET_ACTIVITY + HELP_US_HELP)


def is_garbage_query(qrel_doc_ids: List[str]) -> bool:
    """Check if a query targets garbage documents"""
    return any(doc_id in GARBAGE_DOCUMENT_IDS for doc_id in qrel_doc_ids)


def load_search_cache(cache_file: str = "solr_search_cache.json") -> Dict[str, Dict]:
    """Load search cache from disk"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
                print(f"Loaded {len(cache)} cached searches from {cache_file}")
                return cache
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Could not load cache from {cache_file}, starting fresh")
    return {}


def save_search_cache(cache: Dict[str, Dict], cache_file: str = "solr_search_cache.json"):
    """Save search cache to disk atomically"""
    temp_file = f"{cache_file}.tmp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(cache, f, indent=2)
        os.replace(temp_file, cache_file)
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"Error saving cache: {e}")


async def execute_searches(search_terms: List[str], cache: Dict[str, Dict] = None) -> List[Dict]:
    """Execute multiple searches using Solr masked search with caching"""
    if cache is None:
        cache = {}
    
    results = []
    for term in search_terms:
        # Check cache first
        if term in cache:
            print(f"  💾 Cache hit for '{term}'")
            results.append(cache[term])
            continue
        
        try:
            # Use masked_solr_search directly
            docs, keywords_matched = masked_solr_search(term, SOLR_COLLECTION)
            
            result_dict = {
                "search_results": docs,
                "keywords_matched": keywords_matched
            }
            
            # Store in cache
            cache[term] = result_dict
            results.append(result_dict)
            
            matches_count = len(docs)
            print(f"    Search '{term}' returned {matches_count} matches (keywords: {keywords_matched})")
            
        except Exception as e:
            print(f"  ⚠️  Error searching '{term}': {e}")
            result_dict = {"search_results": [], "keywords_matched": 0}
            results.append(result_dict)
    
    return results


def rrf(rankings: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion to combine multiple rankings"""
    doc_id_and_scores = {}
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in doc_id_and_scores:
                doc_id_and_scores[doc_id] = 0.0
            doc_id_and_scores[doc_id] += 1.0 / (k + rank + 1)
    
    doc_id_and_scores = list(doc_id_and_scores.items())
    doc_id_and_scores.sort(key=lambda x: -x[1])
    return [doc_id for doc_id, score in doc_id_and_scores]


def transform_results(solr_results: List[Dict]) -> List[str]:
    """Transform Solr results to ranked document IDs using RRF"""
    all_rankings = []

    for result in solr_results:
        ranking = []
        search_results = result.get("search_results", [])
        
        # Build ranking for this query - extract document IDs from Solr results
        for doc in search_results:
            doc_id = doc.get("id", "")
            if doc_id:
                ranking.append(doc_id)
        all_rankings.append(ranking)

    # Apply RRF to get unified ranking of document IDs
    document_ids = rrf(all_rankings)
    return document_ids


def get_individual_query_ranks(solr_results: List[Dict], qrel_doc_ids: List[str]) -> List[int]:
    """Get the rank of the target document in each individual search"""
    individual_ranks = []
    
    for result in solr_results:
        search_results = result.get("search_results", [])
        query_rank = 0
        
        # Find rank of qrel document in this individual search
        for rank_idx, doc in enumerate(search_results):
            doc_id = doc.get("id", "")
            if doc_id in qrel_doc_ids:
                query_rank = rank_idx + 1
                break
        
        individual_ranks.append(query_rank)
    
    return individual_ranks


def load_documents(documents_file: str = "documents.jsonl"):
    """Load documents for reranking"""
    documents = {}
    
    # Try different possible documents file locations
    possible_paths = [
        documents_file,
        f"../synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/{documents_file}",
        f"./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/{documents_file}",
        f"./mock-slack/{documents_file}"
    ]
    
    docs_path = None
    for path in possible_paths:
        if os.path.exists(path):
            docs_path = path
            break
    
    if not docs_path:
        print(f"❌ Could not find documents file in any of: {possible_paths}")
        return {}
    
    print(f"Loading documents from: {docs_path}")
    with open(docs_path) as f:
        for line in f:
            if "{" not in line:
                continue
            doc = json.loads(line)
            documents[doc["id"]] = doc
    
    print(f"Loaded {len(documents)} documents")
    return documents


def load_qrels(qrels_file: str = "qrels.jsonl"):
    """Load qrels for evaluation"""
    qrels_by_query_id = {}
    
    # Try different possible qrels file locations
    possible_paths = [
        qrels_file,
        f"../synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/{qrels_file}",
        f"./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/{qrels_file}",
        f"./mock-slack/{qrels_file}"
    ]
    
    qrels_path = None
    for path in possible_paths:
        if os.path.exists(path):
            qrels_path = path
            break
    
    if not qrels_path:
        print(f"❌ Could not find qrels file in any of: {possible_paths}")
        return {}
    
    print(f"Loading qrels from: {qrels_path}")
    with open(qrels_path) as f:
        for line in f:
            if "{" not in line:
                continue
            qrel = json.loads(line)
            query_id = qrel["query_id"]
            if query_id not in qrels_by_query_id:
                qrels_by_query_id[query_id] = []
            qrels_by_query_id[query_id].append(qrel)
    
    print(f"Loaded qrels for {len(qrels_by_query_id)} queries")
    return qrels_by_query_id


def find_target_document_ids(query_id: str, qrels_by_query_id: Dict[str, List[Dict]]) -> List[str]:
    """Find target document IDs for a query from qrels"""
    qrels = qrels_by_query_id.get(query_id, [])
    return [qrel["document_id"] for qrel in qrels]


async def evaluate_query(
    query_id: str,
    query_text: str,
    search_queries: List[str],
    qrels_by_query_id: Dict[str, List[Dict]],
    documents: Dict[str, Dict],
    search_cache: Dict[str, Dict]
) -> Dict[str, Any]:
    """Evaluate a single query"""
    
    # Execute searches using Solr
    solr_results = await execute_searches(search_queries, search_cache)
    
    # Get target document IDs from qrels
    qrel_doc_ids = find_target_document_ids(query_id, qrels_by_query_id)
    
    # Get individual query ranks
    individual_query_ranks = get_individual_query_ranks(solr_results, qrel_doc_ids)
    
    # Transform results using RRF
    document_ids = transform_results(solr_results)
    
    # Rerank all documents if we have results
    if document_ids:
        # Get document texts for reranking
        texts_to_rerank = []
        for doc_id in document_ids:
            doc = documents.get(doc_id, {})
            content = doc.get("content", "")
            texts_to_rerank.append(content)
        
        # Rerank using the zerank-1 model
        print(f"  🔄 Reranking {len(document_ids)} results with zerank-1...")
        try:
            rerank_scores = await ai_rerank(
                model=RERANK_MODEL,
                query=query_text,
                texts=texts_to_rerank,
            )
            
            # Sort documents by rerank scores
            doc_score_pairs = list(zip(document_ids, rerank_scores))
            doc_score_pairs.sort(key=lambda x: -x[1])  # Sort by score descending
            
            # Update document_ids with reranked order
            document_ids = [doc_id for doc_id, _ in doc_score_pairs]
            print(f"  ✅ Reranking completed")
            
        except Exception as e:
            print(f"  ⚠️  Reranking failed: {e}. Using RRF results.")
            # Continue with RRF results if reranking fails
    
    # Calculate recall@20
    qrel_rank = 0
    for i, doc_id in enumerate(document_ids):
        if doc_id in qrel_doc_ids:
            qrel_rank = i + 1
            break
    
    recall_at_20 = 1 if qrel_rank > 0 and qrel_rank <= 20 else 0
    
    # Calculate additional metrics
    total_results = sum(len(result.get('search_results', [])) for result in solr_results)
    keywords_matched = [result.get('keywords_matched', 0) for result in solr_results]
    
    return {
        "query_id": query_id,
        "query": query_text,
        "search_queries": search_queries,
        "individual_query_ranks": individual_query_ranks,
        "qrel_rank": qrel_rank,
        "recall_at_20": recall_at_20,
        "num_results": len(document_ids),
        "total_individual_results": total_results,
        "keywords_matched_per_query": keywords_matched,
        "rrf_combined_results": len(document_ids)
    }


async def main():
    parser = argparse.ArgumentParser(description="Evaluate few-shot results using Solr")
    parser.add_argument("few_shot_results", help="Path to few-shot results JSON file")
    parser.add_argument("--cache-file", default="solr_search_cache.json", 
                       help="Cache file for search results")
    parser.add_argument("--output-file", default="few_shot_solr_evaluation.json",
                       help="Output file for evaluation results")
    
    args = parser.parse_args()
    
    # Load few-shot results
    print(f"Loading few-shot results from {args.few_shot_results}...")
    try:
        with open(args.few_shot_results, 'r') as f:
            few_shot_results = json.load(f)
    except FileNotFoundError:
        print(f"❌ Could not find file: {args.few_shot_results}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in file: {args.few_shot_results}")
        sys.exit(1)
    
    # Filter only successful results
    successful_results = [r for r in few_shot_results if r.get("status") == "success" and r.get("generated_searches")]
    print(f"Found {len(successful_results)} successful results out of {len(few_shot_results)}")
    
    # Load qrels, documents, and search cache
    print("Loading qrels...")
    qrels_by_query_id = load_qrels()
    print("Loading documents...")
    documents = load_documents()
    search_cache = load_search_cache(args.cache_file)
    
    # Filter out garbage queries
    non_garbage_results = []
    for result in successful_results:
        query_id = result.get("query_id", f"query_{len(non_garbage_results)}")
        qrels = qrels_by_query_id.get(query_id, [])
        if qrels:
            qrel_doc_ids = [qrel["document_id"] for qrel in qrels]
            if not is_garbage_query(qrel_doc_ids):
                non_garbage_results.append(result)
    
    print(f"Found {len(non_garbage_results)} non-garbage queries (filtered out {len(successful_results) - len(non_garbage_results)} garbage queries)")
    successful_results = non_garbage_results
    
    if not successful_results:
        print("❌ No successful non-garbage results to evaluate")
        sys.exit(1)
    
    # Process queries
    evaluation_results = []
    total_queries = len(successful_results)
    recall_20_count = 0
    
    print(f"\n🔍 Evaluating {total_queries} queries against Solr collection '{SOLR_COLLECTION}'...")
    
    for i, result in enumerate(successful_results):
        query_id = result.get("query_id", f"query_{i}")
        query_text = result.get("query", "")
        search_queries = result.get("generated_searches", [])
        
        if not search_queries:
            print(f"[{i+1}/{total_queries}] ⚠️  No search queries for {query_id}, skipping...")
            continue
        
        print(f"\n[{i+1}/{total_queries}] Evaluating: {query_text[:80]}...")
        print(f"  Search queries: {search_queries}")
        
        eval_result = await evaluate_query(query_id, query_text, search_queries, qrels_by_query_id, documents, search_cache)
        evaluation_results.append(eval_result)
        
        if eval_result["recall_at_20"] == 1:
            recall_20_count += 1
        
        # Print result summary
        num_results = eval_result["num_results"]
        total_individual = eval_result["total_individual_results"]
        keywords_matched = eval_result["keywords_matched_per_query"]
        
        print(f"  📊 Individual results: {total_individual} total")
        print(f"  📊 Keywords matched per query: {keywords_matched}")
        print(f"  📊 RRF combined results: {num_results}")
        print(f"  📊 Recall@20: {'✅' if eval_result['recall_at_20'] else '❌'} (rank {eval_result['qrel_rank']})")
        
        # Print running stats
        current_recall_rate = (recall_20_count / len(evaluation_results)) * 100
        print(f"  📊 Running recall@20: {recall_20_count}/{len(evaluation_results)} ({current_recall_rate:.1f}%)")
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Final statistics
    total_evaluated = len(evaluation_results)
    final_recall_rate = (recall_20_count / total_evaluated * 100) if total_evaluated > 0 else 0
    
    # Calculate additional statistics
    avg_individual_results = sum(r["total_individual_results"] for r in evaluation_results) / total_evaluated if total_evaluated > 0 else 0
    avg_rrf_results = sum(r["num_results"] for r in evaluation_results) / total_evaluated if total_evaluated > 0 else 0
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total queries evaluated: {total_evaluated}")
    print(f"Recall@20: {recall_20_count}/{total_evaluated} ({final_recall_rate:.1f}%)")
    print(f"Average individual search results: {avg_individual_results:.1f}")
    print(f"Average RRF combined results: {avg_rrf_results:.1f}")
    print(f"Results saved to: {args.output_file}")
    
    # Save cache
    save_search_cache(search_cache, args.cache_file)
    print(f"Search cache: {len(search_cache)} unique queries cached in {args.cache_file}")
    
    print(f"\nQrels loaded for {len(qrels_by_query_id)} queries from Modal Community dataset")


if __name__ == "__main__":
    asyncio.run(main())