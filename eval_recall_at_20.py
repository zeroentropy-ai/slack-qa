#!/usr/bin/env python3
"""
Recall@20 Evaluation Pipeline
Evaluates retrieval performance using Turbopuffer + OpenAI embeddings
"""

import json
import asyncio
import os
import statistics
from typing import List, Dict
from collections import defaultdict
from pathlib import Path


class RecallEvaluator:
    """Evaluate Recall@K for documents, queries, and qrels."""
    
    def __init__(
        self,
        documents: Dict[str, Dict],
        queries: List[Dict],
        qrels: Dict[str, Dict[str, int]],
        namespace_name: str
    ):
        self.documents = documents
        self.queries = queries
        self.qrels = qrels
        self.namespace_name = namespace_name
        
        # Initialize Turbopuffer
        import turbopuffer as tpuf
        api_key = os.getenv("TURBOPUFFER_API_KEY")
        if not api_key:
            raise ValueError("TURBOPUFFER_API_KEY environment variable not set")
        
        self.client = tpuf.Turbopuffer(api_key=api_key, region='aws-us-west-2')
        self.namespace = self.client.namespace(namespace_name)
        
        print(f"✅ Initialized RecallEvaluator")
        print(f"   Namespace: {namespace_name}")
        print(f"   Documents: {len(documents)}")
        print(f"   Queries: {len(queries)}")
        print(f"   Qrels entries: {len(qrels)}")
    
    async def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Search Turbopuffer with OpenAI embeddings."""
        from embed_openai import embed_openai
        
        # Generate query embedding
        query_embedding = await embed_openai(query)
        
        # Search vector database
        results = await self.namespace.query(
            vector=query_embedding,
            top_k=top_k,
            include_vectors=False,
            include_attributes=['text', 'url', 'chunk_idx']
        )
        
        # Convert to list of dicts
        search_results = []
        for result in results.rows:
            search_results.append({
                'doc_id': result.id,
                'text': result.attributes.get('text', ''),
                'url': result.attributes.get('url', ''),
                'chunk_idx': result.attributes.get('chunk_idx', 0),
                'distance': result.dist  # Distance score (lower is better)
            })
        
        return search_results
    
    def calculate_recall_at_k(
        self,
        search_results: List[Dict],
        relevant_docs: Dict[str, int],
        k: int = 20
    ) -> Dict:
        """
        Calculate Recall@K.
        
        Recall@K = (# relevant docs found in top K) / (total # relevant docs)
        """
        # Get top-K results
        top_k_results = search_results[:k]
        
        # Find relevant documents (relevance >= 1)
        relevant_doc_ids = {
            doc_id for doc_id, rel in relevant_docs.items() if rel >= 1
        }
        
        # Count how many relevant docs we found
        found_relevant = sum(
            1 for result in top_k_results
            if result['doc_id'] in relevant_doc_ids
        )
        
        # Calculate recall
        total_relevant = len(relevant_doc_ids)
        recall = found_relevant / total_relevant if total_relevant > 0 else 0.0
        
        return {
            'recall': recall,
            'found': found_relevant,
            'total': total_relevant,
            'k': k
        }
    
    async def evaluate_query(self, query_data: Dict, k: int = 20) -> Dict:
        """Evaluate a single query."""
        query_id = query_data['query_id']
        query = query_data['query']
        
        # Get search results
        search_results = await self.search(query, top_k=k)
        
        # Get ground truth for this query
        relevant_docs = self.qrels.get(query_id, {})
        
        # Calculate recall
        recall_metrics = self.calculate_recall_at_k(
            search_results,
            relevant_docs,
            k=k
        )
        
        return {
            'query_id': query_id,
            'query': query,
            'recall_at_k': recall_metrics,
            'num_results': len(search_results),
            'top_5_results': [
                {
                    'doc_id': r['doc_id'],
                    'distance': r['distance'],
                    'is_relevant': r['doc_id'] in relevant_docs,
                    'relevance_score': relevant_docs.get(r['doc_id'], 0),
                    'text_preview': r['text'][:100] + '...' if len(r['text']) > 100 else r['text']
                }
                for r in search_results[:5]
            ]
        }
    
    async def evaluate_all_queries(self, k: int = 20) -> Dict:
        """Evaluate Recall@K for all queries."""
        all_recalls = []
        results = []
        
        print(f"\n🔍 Evaluating Recall@{k}...")
        
        for i, query_data in enumerate(self.queries):
            if i % 10 == 0 and i > 0:
                print(f"   Progress: {i}/{len(self.queries)} queries")
            
            try:
                query_result = await self.evaluate_query(query_data, k=k)
                all_recalls.append(query_result['recall_at_k']['recall'])
                results.append(query_result)
            except Exception as e:
                print(f"   ⚠️  Error evaluating query {query_data.get('query_id')}: {e}")
                results.append({
                    'query_id': query_data.get('query_id'),
                    'query': query_data.get('query'),
                    'error': str(e)
                })
        
        # Calculate aggregate metrics
        successful_queries = len(all_recalls)
        mean_recall = statistics.mean(all_recalls) if all_recalls else 0.0
        total_found = sum(r['recall_at_k']['found'] for r in results if 'recall_at_k' in r)
        total_relevant = sum(r['recall_at_k']['total'] for r in results if 'recall_at_k' in r)
        
        return {
            'mean_recall_at_k': mean_recall,
            'total_found': total_found,
            'total_relevant': total_relevant,
            'num_queries': len(self.queries),
            'successful_queries': successful_queries,
            'failed_queries': len(self.queries) - successful_queries,
            'query_results': results
        }


# Data loading functions
def load_documents(filepath: str) -> Dict[str, Dict]:
    """Load documents from JSONL file."""
    documents = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc_id = doc['doc_id']
                documents[doc_id] = doc
    return documents


def load_queries(filepath: str) -> List[Dict]:
    """Load queries from JSONL file."""
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def load_qrels(filepath: str) -> Dict[str, Dict[str, int]]:
    """
    Load relevance judgments from JSONL file.
    
    Returns:
        Dict mapping query_id -> {doc_id: relevance_score}
    """
    qrels = defaultdict(dict)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qrel = json.loads(line)
                query_id = qrel['query_id']
                doc_id = qrel['doc_id']
                relevance = qrel['relevance']
                qrels[query_id][doc_id] = relevance
    return dict(qrels)


async def index_documents_turbopuffer(
    documents: Dict[str, Dict],
    namespace_name: str,
    batch_size: int = 100
):
    """Index documents in Turbopuffer with OpenAI embeddings."""
    import turbopuffer as tpuf
    from embed_openai import embed_openai
    
    print(f"\n📊 Indexing {len(documents)} documents in Turbopuffer...")
    
    # Connect to Turbopuffer
    api_key = os.getenv("TURBOPUFFER_API_KEY")
    client = tpuf.Turbopuffer(api_key=api_key, region='aws-us-west-2')
    ns = client.namespace(namespace_name)
    
    # Prepare documents with embeddings
    batch = []
    indexed = 0
    
    for doc_id, doc in documents.items():
        # Generate embedding
        embedding = await embed_openai(doc['text'])
        
        # Prepare for upload
        batch.append({
            'id': doc_id,
            'vector': embedding,
            'attributes': {
                'text': doc['text'],
                'url': doc.get('url', ''),
                'chunk_idx': doc.get('chunk_idx', 0)
            }
        })
        
        # Upload in batches
        if len(batch) >= batch_size:
            await ns.upsert(batch)
            indexed += len(batch)
            print(f"   Indexed {indexed}/{len(documents)} documents...")
            batch = []
    
    # Upload remaining
    if batch:
        await ns.upsert(batch)
        indexed += len(batch)
    
    print(f"✅ Indexed {indexed} documents in namespace: {namespace_name}")


async def run_evaluation(data_dir: str, namespace_name: str, skip_indexing: bool = False):
    """Main evaluation pipeline."""
    
    print("=" * 70)
    print("🚀 Recall@20 Evaluation Pipeline")
    print("=" * 70)
    
    # Load data
    print("\n📁 Loading data...")
    documents_file = f"{data_dir}/documents.jsonl"
    queries_file = f"{data_dir}/queries.jsonl"
    qrels_file = f"{data_dir}/qrels.jsonl"
    
    documents = load_documents(documents_file)
    queries = load_queries(queries_file)
    qrels = load_qrels(qrels_file)
    
    print(f"   Documents: {len(documents)}")
    print(f"   Queries: {len(queries)}")
    print(f"   Qrels: {len(qrels)} queries with relevance judgments")
    
    # Index documents in Turbopuffer (unless skipped)
    if not skip_indexing:
        await index_documents_turbopuffer(documents, namespace_name)
    else:
        print(f"\n⏭️  Skipping indexing (using existing namespace: {namespace_name})")
    
    # Create evaluator
    evaluator = RecallEvaluator(documents, queries, qrels, namespace_name)
    
    # Evaluate Recall@20
    results = await evaluator.evaluate_all_queries(k=20)
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Mean Recall@20:     {results['mean_recall_at_k']:.4f}")
    print(f"Total Found:        {results['total_found']}/{results['total_relevant']}")
    print(f"Queries Evaluated:  {results['successful_queries']}/{results['num_queries']}")
    print(f"Failed Queries:     {results['failed_queries']}")
    print("=" * 70)
    
    # Save results
    output_dir = f"{data_dir}/results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/recall_at_20_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Full results saved to: {output_file}")
    
    # Save summary
    summary_file = f"{output_dir}/recall_at_20_summary.json"
    summary = {
        'namespace': namespace_name,
        'embedding_model': 'openai/text-embedding-3-small',
        'metrics': {
            'mean_recall_at_20': results['mean_recall_at_k'],
            'total_found': results['total_found'],
            'total_relevant': results['total_relevant'],
            'successful_queries': results['successful_queries'],
            'failed_queries': results['failed_queries']
        },
        'sample_queries': [
            {
                'query': r['query'],
                'recall': r['recall_at_k']['recall'],
                'found': r['recall_at_k']['found'],
                'total': r['recall_at_k']['total']
            }
            for r in results['query_results'][:5]
            if 'recall_at_k' in r
        ]
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Summary saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Recall@20 using Turbopuffer + OpenAI embeddings"
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing documents.jsonl, queries.jsonl, and qrels.jsonl"
    )
    parser.add_argument(
        "--namespace",
        default="recall_eval",
        help="Turbopuffer namespace name (default: recall_eval)"
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip document indexing (use existing namespace)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(args.data_dir, args.namespace, args.skip_indexing))
