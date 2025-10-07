#!/usr/bin/env python3
"""
Search Configuration Evaluation System.
Tests different embedding + reranking combinations against ground truth data.
"""

import json
import numpy as np
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import statistics

from hybrid_search import TurbopufferHybridSearcher


@dataclass
class SearchResult:
    """Single search result with metadata."""
    text: str
    url: str
    chunk_idx: int
    similarity: float
    rank: int


@dataclass
class EvaluationConfig:
    """Configuration for search evaluation."""
    embedding_provider: str  # 'qwen' or 'openai'
    reranker: str           # 'cohere', 'zerank', 'none'
    top_k_retrieval: int    # Number of candidates to retrieve before reranking
    top_k_final: int        # Number of final results to evaluate


class Reranker:
    """Base class for reranking implementations."""

    def __init__(self, name: str):
        self.name = name

    async def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank candidates based on query relevance.

        Returns:
            Reranked list of SearchResult objects
        """
        raise NotImplementedError()


class CohereReranker(Reranker):
    """Cohere reranking using their API."""

    def __init__(self):
        super().__init__("cohere")
        try:
            import cohere
            self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        except ImportError:
            raise ImportError("Install cohere: pip install cohere")

    async def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """Rerank using Cohere's rerank API."""
        if not candidates:
            return candidates

        try:
            # Prepare documents for reranking
            documents = [result.text for result in candidates]

            # Call Cohere rerank API
            response = self.client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=len(documents),
                return_documents=False
            )

            # Map results back to SearchResult objects
            reranked = []
            for i, result in enumerate(response.results):
                original_idx = result.index
                original_result = candidates[original_idx]

                # Update rank and create new result
                reranked_result = SearchResult(
                    text=original_result.text,
                    url=original_result.url,
                    chunk_idx=original_result.chunk_idx,
                    similarity=result.relevance_score,  # Use rerank score
                    rank=i + 1
                )
                reranked.append(reranked_result)

            return reranked

        except Exception as e:
            print(f"Warning: Cohere reranking failed: {e}")
            return candidates  # Return original order on failure


class ZerankReranker(Reranker):
    """ZeRank-1 reranking using ZeroEntropy API."""

    def __init__(self):
        super().__init__("zerank")
        try:
            import httpx
            import ssl

            self.api_key = os.getenv("ZEROENTROPY_API_KEY")
            self.base_url = os.getenv("ZEROENTROPY_BASE_URL", "https://api.zeroentropy.dev")

            if not self.api_key:
                raise ValueError("ZEROENTROPY_API_KEY environment variable not set")

            # Ensure base URL doesn't end with slash
            self.base_url = self.base_url.rstrip('/')

            # Create SSL context that's more permissive for custom instances
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            ssl_context.set_ciphers('DEFAULT')

            # Create client with custom SSL context
            self.client = httpx.AsyncClient(verify=ssl_context)

        except ImportError:
            raise ImportError("Install httpx: pip install httpx")

    async def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """Rerank using ZeroEntropy API."""
        if not candidates:
            return candidates

        try:
            # Prepare documents for reranking
            documents = [{"text": result.text} for result in candidates]

            # Call ZeroEntropy rerank API using correct endpoint and parameters
            response = await self.client.post(
                f"{self.base_url}/v1/models/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "zerank-1",
                    "query": query,
                    "documents": [doc["text"] for doc in documents],  # Send as string array, not objects
                    "top_n": len(documents)
                },
                timeout=30.0
            )
            response.raise_for_status()

            result_data = response.json()

            # Map results back to SearchResult objects
            reranked = []
            for i, result in enumerate(result_data["results"]):
                original_idx = result["index"]
                original_result = candidates[original_idx]

                # Update rank and create new result
                reranked_result = SearchResult(
                    text=original_result.text,
                    url=original_result.url,
                    chunk_idx=original_result.chunk_idx,
                    similarity=result["relevance_score"],  # Use rerank score
                    rank=i + 1
                )
                reranked.append(reranked_result)

            return reranked

        except Exception as e:
            print(f"Warning: ZeRank reranking failed: {e}")
            return candidates  # Return original order on failure


class NoReranker(Reranker):
    """No reranking - just returns original embedding order."""

    def __init__(self):
        super().__init__("none")

    async def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        """Return candidates in original order."""
        return candidates


class SearchEvaluator:
    """Evaluates search configurations against ground truth."""

    def __init__(self, tenant_name: str):
        self.tenant_name = tenant_name
        self.searchers = {
            'qwen': TurbopufferHybridSearcher(provider='qwen'),
            'openai': TurbopufferHybridSearcher(provider='openai')
        }

        # Initialize rerankers
        self.rerankers = {
            'none': NoReranker(),
            'cohere': None,  # Initialize on demand
            'zerank': None   # Initialize on demand
        }

        # Load ground truth
        self.ground_truth = self._load_ground_truth()

        print(f"Initialized evaluator for {tenant_name}")
        print(f"Ground truth loaded: {len(self.ground_truth)} queries")

    def _load_ground_truth(self) -> Dict[str, Dict]:
        """Load ground truth data for the tenant."""
        gt_file = f"data/ground_truth/{self.tenant_name}_ground_truth.json"

        if not os.path.exists(gt_file):
            # Try to find any ground truth file with curated queries
            curated_gt_file = f"data/ground_truth/{self.tenant_name}_curated_ground_truth.json"
            if os.path.exists(curated_gt_file):
                gt_file = curated_gt_file
            else:
                raise FileNotFoundError(f"No ground truth found for {self.tenant_name}")

        with open(gt_file, 'r') as f:
            return json.load(f)

    def _get_reranker(self, reranker_name: str) -> Reranker:
        """Get reranker instance, initializing if needed."""
        if reranker_name not in self.rerankers:
            raise ValueError(f"Unknown reranker: {reranker_name}")

        if self.rerankers[reranker_name] is None:
            if reranker_name == 'cohere':
                self.rerankers[reranker_name] = CohereReranker()
            elif reranker_name == 'zerank':
                self.rerankers[reranker_name] = ZerankReranker()

        return self.rerankers[reranker_name]

    async def _search_with_config(self, query: str, config: EvaluationConfig) -> List[SearchResult]:
        """Perform hybrid search with given configuration."""

        # Retrieve candidates using hybrid search
        searcher = self.searchers[config.embedding_provider]
        raw_results = await searcher.search_file(
            self.tenant_name,
            query,
            config.top_k_retrieval
        )

        # Convert to SearchResult objects
        candidates = []
        for i, result in enumerate(raw_results):
            search_result = SearchResult(
                text=result['text'],
                url=result['url'],
                chunk_idx=result['chunk_idx'],
                similarity=result['similarity'],
                rank=i + 1
            )
            candidates.append(search_result)

        # Apply reranking if specified
        if config.reranker != 'none':
            reranker = self._get_reranker(config.reranker)
            candidates = await reranker.rerank(query, candidates)

        # Return top-K final results
        return candidates[:config.top_k_final]

    def _evaluate_query(self, query: str, search_results: List[SearchResult], ground_truth_data: Dict) -> Dict:
        """Evaluate search results for a single query against ground truth."""

        if 'error' in ground_truth_data:
            return {'error': ground_truth_data['error']}

        # Get ground truth candidates with relevance scores
        gt_candidates = ground_truth_data.get('candidates', [])

        # Create mapping of (url, chunk_idx) -> relevance score
        gt_relevance = {}
        for candidate in gt_candidates:
            key = (candidate['url'], candidate['chunk_idx'])
            relevance_score = candidate.get('openai_relevance_score', 0.0)
            gt_relevance[key] = relevance_score

        # Evaluate each search result
        metrics = {
            'query': query,
            'total_results': len(search_results),
            'ground_truth_size': len(gt_candidates),
            'results_with_gt': 0,
            'total_relevance': 0.0,
            'recall_at_k': {},
            'all_results': [],  # Store top 10 results for detailed analysis
            'top_results': []   # Keep top 5 for quick preview
        }

        relevance_scores = []

        for i, result in enumerate(search_results):
            result_key = (result.url, result.chunk_idx)
            relevance = gt_relevance.get(result_key, 0.0)

            relevance_scores.append(relevance)

            if relevance > 0:
                metrics['results_with_gt'] += 1

            metrics['total_relevance'] += relevance

            # Store top 10 result details for analysis
            if i < 10:  # Only store top 10
                result_detail = {
                    'rank': i + 1,
                    'url': result.url,
                    'chunk_idx': result.chunk_idx,
                    'similarity': result.similarity,  # Original embedding similarity or rerank score
                    'relevance': relevance,
                    'text_preview': result.text[:200] + '...' if len(result.text) > 200 else result.text,
                    'full_text': result.text  # Store full text for detailed inspection
                }
                metrics['all_results'].append(result_detail)

            # Store top result details for quick preview
            if i < 5:
                metrics['top_results'].append(result_detail)

        # Calculate Recall@K (how many relevant items we found out of all relevant items)
        total_relevant_items = sum(1 for score in gt_relevance.values() if score >= 0.5)
        metrics['total_relevant_items'] = total_relevant_items
        for k in [5, 10, 20]:
            if k <= len(relevance_scores):
                relevant_found_at_k = sum(1 for score in relevance_scores[:k] if score >= 0.5)
                metrics['recall_at_k'][k] = {
                    'recall': relevant_found_at_k / total_relevant_items if total_relevant_items > 0 else 0.0,
                    'found': relevant_found_at_k,
                    'total': total_relevant_items
                }

        # Overall metrics
        metrics['mean_relevance'] = metrics['total_relevance'] / len(search_results) if search_results else 0.0
        metrics['recall'] = metrics['results_with_gt'] / len(gt_candidates) if gt_candidates else 0.0

        return metrics

    async def evaluate_config(self, config: EvaluationConfig, max_queries: Optional[int] = None) -> Dict:
        """Evaluate a specific search configuration."""

        print(f"\n🔍 Evaluating hybrid config: {config.embedding_provider} + {config.reranker} (top-{config.top_k_retrieval} → top-{config.top_k_final})")

        queries = list(self.ground_truth.keys())
        if max_queries:
            queries = queries[:max_queries]

        results = {
            'config': config,
            'total_queries': len(queries),
            'query_results': [],
            'aggregate_metrics': {}
        }

        # Evaluate each query
        all_recalls = {k: [] for k in [5, 10, 20]}
        all_found = {k: [] for k in [5, 10, 20]}
        all_total = {k: [] for k in [5, 10, 20]}

        for i, query in enumerate(queries):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(queries)} queries")

            try:
                # Perform search
                search_results = await self._search_with_config(query, config)

                # Evaluate against ground truth
                query_metrics = self._evaluate_query(query, search_results, self.ground_truth[query])
                results['query_results'].append(query_metrics)

                if 'error' not in query_metrics:
                    # Aggregate recall metrics
                    for k in [5, 10, 20]:
                        if k in query_metrics['recall_at_k']:
                            all_recalls[k].append(query_metrics['recall_at_k'][k]['recall'])
                            all_found[k].append(query_metrics['recall_at_k'][k]['found'])
                            all_total[k].append(query_metrics['recall_at_k'][k]['total'])

            except Exception as e:
                print(f"   Error evaluating query '{query}': {e}")
                results['query_results'].append({'query': query, 'error': str(e)})

        # Calculate aggregate metrics
        successful_queries = len([scores for scores in all_recalls[5] if scores is not None])
        results['aggregate_metrics'] = {
            'recall_at_k': {},
            'successful_queries': successful_queries,
            'failed_queries': results['total_queries'] - successful_queries
        }
        
        for k in [5, 10, 20]:
            total_found = sum(all_found[k])
            total_relevant = sum(all_total[k])
            mean_recall = statistics.mean(all_recalls[k]) if all_recalls[k] else 0.0
            results['aggregate_metrics']['recall_at_k'][k] = {
                'mean_recall': mean_recall,
                'total_found': total_found,
                'total_relevant': total_relevant
            }

        print(f"   Completed: {results['aggregate_metrics']['successful_queries']}/{results['total_queries']} queries")
        for k in [5, 10, 20]:
            metrics = results['aggregate_metrics']['recall_at_k'][k]
            print(f"   Recall@{k}: {metrics['mean_recall']:.3f} ({metrics['total_found']}/{metrics['total_relevant']})")

        return results


async def run_hybrid_api_comparison(data_dir: str, max_queries: Optional[int] = None):
    """Compare Turbopuffer hybrid search with Slack API search"""

    print(f"🚀 Starting comparison for Slack data in {data_dir}")
    if max_queries:
        print(f"   Limited to {max_queries} queries per config")

    # Save results
    output_dir = "data/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    # Save full results with all search details
    output_file = f"{output_dir}/{tenant_name}_search_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save a more readable summary for quick analysis
    summary_file = f"{output_dir}/{tenant_name}_search_summary.json"
    summary_results = []
    for result in all_results:
        config = result['config']
        summary = {
            'config': {
                'embedding': config.embedding_provider,
                'reranker': config.reranker,
                'retrieval_k': config.top_k_retrieval,
                'final_k': config.top_k_final
            },
            'metrics': result['aggregate_metrics'],
            'sample_queries': []
        }

        # Add a few sample queries with their results for spot-checking
        for query_result in result['query_results'][:3]:  # First 3 queries
            if 'error' not in query_result:
                summary['sample_queries'].append({
                    'query': query_result['query'],
                    'precision_at_10': query_result.get('precision_at_k', {}).get(10, 0.0),
                    'top_3_results': query_result.get('top_results', [])[:3]
                })

        summary_results.append(summary)

    with open(summary_file, 'w') as f:
        json.dump(summary_results, f, indent=2, default=str)

    # Save detailed reranking comparisons for analysis
    comparison_file = f"{output_dir}/{tenant_name}_rerank_comparisons.jsonl"
    with open(comparison_file, 'w') as f:
        for result in all_results:
            config = result['config']
            for query_result in result['query_results']:
                if 'error' not in query_result and query_result.get('all_results'):
                    comparison_entry = {
                        'config': f"{config.embedding_provider}+{config.reranker}",
                        'query': query_result['query'],
                        'precision_at_10': query_result.get('precision_at_k', {}).get(10, 0.0),
                        'results': query_result['all_results']  # All ranked results
                    }
                    f.write(json.dumps(comparison_entry, default=str) + '\n')

    print(f"\n✅ Evaluation complete!")
    print(f"   Full results: {output_file}")
    print(f"   Summary: {summary_file}")
    print(f"   Rerank comparisons: {comparison_file}")
    print(f"   Total files: {len(all_results)} configs × {len(list(evaluator.ground_truth.keys()))} queries")

    # Print comprehensive summary
    print("\n📊 RECALL RESULTS SUMMARY:")
    print("=" * 80)
    print(f"{'Config':<20} {'R@5':<18} {'R@10':<18} {'R@20':<18}")
    print("-" * 80)

    for result in all_results:
        config = result['config']
        metrics = result['aggregate_metrics']
        config_name = f"{config.embedding_provider}+{config.reranker}"

        r = metrics['recall_at_k']

        r5_str = f"{r[5]['mean_recall']:.3f} ({r[5]['total_found']}/{r[5]['total_relevant']})"
        r10_str = f"{r[10]['mean_recall']:.3f} ({r[10]['total_found']}/{r[10]['total_relevant']})"
        r20_str = f"{r[20]['mean_recall']:.3f} ({r[20]['total_found']}/{r[20]['total_relevant']})"

        print(f"{config_name:<20} {r5_str:<18} {r10_str:<18} {r20_str:<18}")

    print("=" * 80)
    print("Legend: R@K = Mean Recall@K (chunks with score ≥0.5 found / total relevant chunks)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate search configurations")
    parser.add_argument("data_dir", help="Path containing {documents/queries/qrels}.jsonl")
    parser.add_argument("--max-queries", type=int, help="Limit number of queries for testing")

    args = parser.parse_args()

    asyncio.run(run_hybrid_api_comparison(args.data_dir, args.max_queries))