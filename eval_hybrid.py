#!/usr/bin/env python3
"""
JSONL-based Search Evaluation System.
Loads documents, queries, and qrels from JSONL files, performs hybrid search,
and computes Recall@20 metrics.
"""

import json
import asyncio
import os
import uuid
import turbopuffer
import cohere
from zeroentropy import ZeroEntropy
from pathlib import Path
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import statistics

from openai_tools import embed_openai, query_reduce_openai
from hybrid_search import TurbopufferHybridSearcher
from slack_search import SlackSearch

@dataclass
class Document:
    """Document with id and content."""
    doc_id: str
    text: str
    metadata: Dict = None


@dataclass
class Query:
    """Query with id and text."""
    query_id: str
    text: str


@dataclass
class QRel:
    """Query-document relevance judgment."""
    query_id: str
    doc_id: str
    relevance: int  # 0 = not relevant, 1+ = relevant

def generate_deterministic_uuid(text: str, timestamp: float) -> str:
    """Generate a deterministic UUID from text and timestamp."""
    combined = f"{text}:{timestamp}"
    namespace = uuid.NAMESPACE_DNS # arbitrary namespace
    return str(uuid.uuid5(namespace, combined))


class JSONLDataLoader:
    """Loads documents, queries, and qrels from JSONL files."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_documents(self, filename: str = "documents.jsonl") -> List[Document]:
        """Load documents from JSONL file.
        
        Expected format: {"doc_id": "...", "text": "...", "metadata": {...}}
        """
        docs = []
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Documents file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    doc = Document(
                        doc_id=data['id'],
                        text=data['content'],
                        metadata=data.get('metadata', {})
                    )
                    docs.append(doc)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping invalid line {line_num} in {filename}: {e}")
                    
        print(f"Loaded {len(docs)} documents from {filename}")
        return docs
    
    def load_queries(self, filename: str = "queries.jsonl") -> List[Query]:
        """Load queries from JSONL file.
        
        Expected format: {"query_id": "...", "text": "..."}
        """
        queries = []
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Queries file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    query = Query(
                        query_id=data['id'],
                        text=data['query']
                    )
                    queries.append(query)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping invalid line {line_num} in {filename}: {e}")
                    
        print(f"Loaded {len(queries)} queries from {filename}")
        return queries
    
    def load_qrels(self, filename: str = "qrels.jsonl") -> List[QRel]:
        """Load query-document relevance judgments from JSONL file.
        
        Expected format: {"query_id": "...", "doc_id": "...", "relevance": 1}
        """
        qrels = []
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Qrels file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    qrel = QRel(
                        query_id=data['query_id'],
                        doc_id=data['document_id'],
                        relevance=data.get('score', 1)
                    )
                    qrels.append(qrel)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping invalid line {line_num} in {filename}: {e}")
                    
        print(f"Loaded {len(qrels)} qrels from {filename}")
        return qrels


class TurboPufferIndexer:
    """Handles document indexing into TurboPuffer."""
    
    def __init__(self, namespace: str, provider: str = 'openai'):
        self.namespace = namespace
        self.searcher = TurbopufferHybridSearcher(provider=provider)
        
    async def index_documents(self, documents: List[Document], batch_size: int = 100, empty_first: bool = False, start_index: int = 0):
        """Index documents into TurboPuffer with embeddings."""
        print(f"\n📥 Indexing {len(documents)} documents into namespace '{self.namespace}'...")
        
        tpuf = turbopuffer.Turbopuffer(
            api_key=os.getenv("TURBOPUFFER_API_KEY"),
            region="aws-us-west-2"
        )
        ns = tpuf.namespace(self.namespace)

        # Clear this namespace out first
        if empty_first:
            try:
                ns.delete_all()
                print(f"Cleared existing data from namespace {self.namespace}")
            except:
                print(f"No existing data to clear in namespace {self.namespace}")

        pbar = tqdm(
            desc="Document Embeddings",
            total=len(documents),
        )

        pbar.update(start_index)

        # Index the documents
        for i in range(start_index, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"   Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            # Format documents for TurboPuffer
            # Note: You'll need to adapt this based on your actual TurboPuffer schema
            upsert_docs = []
            for doc in batch:
                upsert_docs.append({
                    'id': doc.doc_id,
                    'vector': await embed_openai(doc.text),
                    'content': doc.text,
                    **doc.metadata
                })
            
            # Index batch (you'll need to implement this in your hybrid_search module)
            # await self.searcher.index_documents(self.namespace, formatted_docs)
            ns.write(upsert_rows=upsert_docs,
                     distance_metric="cosine_distance",
                     schema={"content": {"type" : "string", "full_text_search": True}})
            pbar.update(len(batch))
        
        pbar.close()
            
        print(f"✅ Indexing complete!")


class SearchEvaluator:
    """Evaluates search performance using Recall@K metrics."""
    
    def __init__(self, namespace: str, provider: str = 'openai'):
        self.namespace = namespace
        self.searcher = TurbopufferHybridSearcher(provider=provider)
        
    def build_relevance_map(self, qrels: List[QRel]) -> Dict[str, Set[str]]:
        """Build mapping of query_id -> set of relevant doc_ids."""
        relevance_map = defaultdict(set)
        
        for qrel in qrels:
            if qrel.relevance > 0:  # Only count positive relevance
                relevance_map[qrel.query_id].add(qrel.doc_id)
                
        return dict(relevance_map)
    
    async def rerank_results_cohere(self, results: List, query: str) -> List:
        """Rerank results using Cohere embeddings."""
        if not os.getenv("COHERE_API_KEY"):
            print("COHERE_API_KEY not set, skipping reranking")
            return results
        try:
            co = cohere.Client(os.getenv("COHERE_API_KEY"))
            reranked = co.rerank(query=query, documents=[r['text'] for r in results], top_n=len(results)).results
            for r in reranked:
                results[r.index]['$dist'] = r.relevance_score
            return [results[r.index] for r in reranked]
        except ImportError:
            print("Cohere library not installed, skipping reranking")
            return results
        
    async def rerank_results_ze(self, results: List, query: str) -> List:
        """Rerank results using ZeroEntropy embeddings."""
        if not os.getenv("ZEROENTROPY_API_KEY"):
            print("ZEROENTROPY_API_KEY not set, skipping reranking")
            return results
        try:
            zclient = ZeroEntropy(api_key=os.getenv("ZEROENTROPY_API_KEY"))
            response = zclient.models.rerank(
                model="zerank-1",
                query=query,
                documents=[r['text'] for r in results]
            ).results
            for r in response:
                results[r.index]['$dist'] = r.relevance_score
            return [results[r.index] for r in response]
        except ImportError:
            print("ZeroEntropy library not installed, skipping reranking")
            return results

    async def search_query(self, query: str, top_k: int = 20) -> List[str]:
        """Perform hybrid search and return list of doc_ids."""
        try:
            results = await self.searcher.search_file(
                self.namespace,
                query,
                top_k
            )

            return results
        except Exception as e:
            print(f"Error searching query: {e}")
            return []
    
    def convert_to_doc_ids(self, results: List) -> List[str]:
        """Extract doc_ids from search results."""
        doc_ids = []
        for result in results:
            # Adapt this based on your result format
            doc_id = result.get('id') or result.get('doc_id')
            if doc_id:
                doc_ids.append(doc_id)
                
        return doc_ids

    async def search_slack(self, query: str, top_k: int = 20) -> List[str]:
        """Perform Slack search and return list of message IDs."""
        try:
            trunc = query[:150]
            keywords = await query_reduce_openai(query)
            print(f"Searching slack for '{keywords}', distilled from {trunc}...")

            results = self.slacksearch.search(keywords, search_type="messages", count=top_k)
            message_ids = []
            for match in results.matches:
                text, ts = match.get('text', ''), float(match.get('ts', '0'))
                mid = generate_deterministic_uuid(text, ts)
                print(f" ===> generated message ID = {mid} for ts={ts}, text={repr(text)[:50]}...")
            return results
        except Exception as e:
            print(f"Error searching Slack: {e}")
            return []

    def calculate_recall_at_k(
        self, 
        retrieved_docs: List[str], 
        relevant_docs: Set[str], 
        k: int
    ) -> float:
        """Calculate Recall@K metric.
        
        Recall@K = (# relevant docs in top-K) / (# total relevant docs)
        """
        if not relevant_docs:
            return 0.0
            
        top_k_docs = set(retrieved_docs[:k])
        relevant_retrieved = top_k_docs.intersection(relevant_docs)
        
        return len(relevant_retrieved) / len(relevant_docs)
    
    async def evaluate_hybrid(
        self, 
        queries: List[Query], 
        qrels: List[QRel],
        k_values: List[int] = None,
        rerank: bool = False
    ) -> Dict:
        """Evaluate search performance across all queries."""
        if k_values is None:
            k_values = [5, 10, 20]
            
        print(f"\n🔍 Evaluating hybrid search performance...")
        print(f"   Queries: {len(queries)}")
        print(f"   Metrics: Recall@{k_values}")
        
        # Build relevance map
        relevance_map = self.build_relevance_map(qrels)
        
        # Track results
        recall_scores = {k: [] for k in k_values}
        query_results = []
        
        for i, query in enumerate(queries):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(queries)} queries")
            
            # Get relevant documents for this query
            relevant_docs = relevance_map.get(query.query_id, set())
            
            if not relevant_docs:
                print(f"   Warning: No relevant docs for query {query.query_id}")
                continue
            
            # Perform Turbopuffer vector search
            retrieved_documents = await self.search_query(query.text, 50)

            # Rerank results with Cohere if specified
            if rerank:
                retrieved_documents = await self.rerank_results_ze(retrieved_documents, query.text)

            retrieved_doc_ids = self.convert_to_doc_ids(retrieved_documents)

            # Calculate recall at different K values
            query_recall = {}
            for k in k_values:
                recall = self.calculate_recall_at_k(retrieved_doc_ids, relevant_docs, k)
                recall_scores[k].append(recall)
                query_recall[f'recall@{k}'] = recall
            
            query_results.append({
                'query_id': query.query_id,
                'query_text': query.text,
                'relevant_docs_count': len(relevant_docs),
                'retrieved_docs': retrieved_doc_ids[:max(k_values)],
                **query_recall
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = {}
        for k in k_values:
            if recall_scores[k]:
                aggregate_metrics[f'recall@{k}'] = {
                    'mean': statistics.mean(recall_scores[k]),
                    'median': statistics.median(recall_scores[k]),
                    'min': min(recall_scores[k]),
                    'max': max(recall_scores[k])
                }
            else:
                aggregate_metrics[f'recall@{k}'] = {
                    'mean': 0.0,
                    'median': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
        
        results = {
            'aggregate_metrics': aggregate_metrics,
            'total_queries': len(queries),
            'evaluated_queries': len(query_results),
            'query_results': query_results
        }
        
        # Print summary
        print(f"\n✅ Hybrid Search evaluation complete!")
        print(f"   Evaluated: {results['evaluated_queries']}/{results['total_queries']} queries")
        print(f"\n📊 RESULTS:")
        print("=" * 60)
        for k in k_values:
            metrics = aggregate_metrics[f'recall@{k}']
            print(f"Recall@{k:2d}: {metrics['mean']:.4f} (median: {metrics['median']:.4f})")
        print("=" * 60)
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n💾 Results saved to: {output_file}")


async def main(
    data_dir: str,
    workspace: str,
    provider: str = 'openai',
    skip_indexing: bool = False,
    output_file: str = None
):
    """Main evaluation pipeline."""

    namespace = f"{workspace}-search-eval"

    print(f"🚀 Starting Evaluation of Hybrid Search vs Slack API")
    print(f"   Workspace: {workspace}")
    print(f"   Data directory: {data_dir}")
    print(f"   Namespace: {namespace}")
    print(f"   Embedding Provider: {provider}")
    
    # Load data
    loader = JSONLDataLoader(data_dir)
    documents = loader.load_documents()
    queries = loader.load_queries()
    qrels = loader.load_qrels()

    # Index documents (if not skipping)
    if not skip_indexing:
        indexer = TurboPufferIndexer(namespace, provider)
        await indexer.index_documents(documents, batch_size=100)
    else:
        print("\n⏭️  Skipping indexing (using existing index)")

    # Evaluate search
    evaluator = SearchEvaluator(namespace, provider)
    results = await evaluator.evaluate_hybrid(queries, qrels, rerank=True)
    
    # Save results
    if output_file is None:
        output_file = f"results/{namespace}_evaluation.json"
    evaluator.save_results(results, output_file)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare turbopuffer hybrid search performance with Slack API for a given workspace"
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing documents.jsonl, queries.jsonl, and qrels.jsonl"
    )
    parser.add_argument(
        "--workspace",
        choices=['modal', 'support-driven'],
        required=True,
        help="Workspace to evaluate (determines namespace)"
    )
    parser.add_argument(
        "--provider",
        choices=['openai', 'qwen'],
        default='openai',
        help="Embedding provider (default: openai)"
    )
    parser.add_argument(
        "--skip-indexing",
        action='store_true',
        help="Skip document indexing (use existing index)"
    )
    parser.add_argument(
        "--output",
        help="Output file for results (default: results/<namespace>_evaluation.json)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(
        args.data_dir,
        args.workspace,
        args.provider,
        args.skip_indexing,
        args.output
    ))