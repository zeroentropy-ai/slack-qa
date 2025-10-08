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
from pathlib import Path
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import statistics

from openai_tools import embed_openai, query_reduce_openai
from hybrid_search import TurbopufferHybridSearcher
from slack_search import SlackSearch


XOXC = {
    "support-driven": "xoxc-2469263068-9628809684802-9686151789920-db3b91155804aecf49f36726aa7b23ab690bac1cde57ff90f7d0859fe848edc0",
    "modal":          "xoxc-3052645262231-9641679200657-9613480439127-31e794a9afc781c3fc3cdcd019732e0fe4fd25af2b58e86ba83ec0285c3c6283",
}

COOKIES = {
    "support-driven": "b=.a61952af5dcace9baebd31847eaf197f; shown_ssb_redirect_page=1; tz=-420; ssb_instance_id=332ff020-d3b1-46d7-8be7-973d48eda65a; optimizelySession=1759438330883; utm=%7B%22utm_source%22%3A%22thehiveindex.com%22%7D; shown_download_ssb_modal=1; show_download_ssb_banner=1; no_download_ssb_banner=1; web_cache_last_updated652e8224e5adbb12232da00d72e1b32f=1759444360895; web_cache_last_updated0ad2d18e8d30e6d95e1c1d8604385e48=1759449180999; web_cache_last_updated58208ac66e071bd79a824469a9c06679=1759449373157; web_cache_last_updated04735373bcef794525852d1ec5c5c79a=1759449506346; d-s=1759876715; ec=enQtOTY0OTYwMjY0OTg0My1kOGEwNWQyNGM5OGQ3MzdmM2NjODA3ZGJmNDU5YmRiMzBlMjkxYWVjNTk1YmYxOTg3OWJiMmViOTQ3MDEyMThm; web_cache_last_updatedbc23576c2ec06665f1c40206504516b3=1759882038110; x=a61952af5dcace9baebd31847eaf197f.1759907564; lc=1759907612; OptanonConsent=isGpcEnabled=0&datestamp=Wed+Oct+08+2025+00%3A13%3A33+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=07597623-af2c-4821-bd16-7dcdeea0d673&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; d=xoxd-8P7QVc1SrHOz5oVTJcARdIr6LuMjuDM6j6AdIKQW96Ir8BeyrpUKTeR5YCFD9t3tsTurF%2FZvEhSMks%2BRgY1uqw30jFBJkiPpHrIE1nPmhbr5qUp1lRMjdioHFI%2FWnckFKp4VZGFUCBf2KyalVNgZllh9lY3f8G42%2F0q6Y0jjQeAh3nTvZkHfGtNruQ5MB76cdoUCHn7oAVt82M2ZbNwcFxmk",
    "modal":          "b=.a61952af5dcace9baebd31847eaf197f; shown_ssb_redirect_page=1; tz=-420; ssb_instance_id=332ff020-d3b1-46d7-8be7-973d48eda65a; optimizelySession=1759438330883; utm=%7B%22utm_source%22%3A%22thehiveindex.com%22%7D; shown_download_ssb_modal=1; show_download_ssb_banner=1; no_download_ssb_banner=1; web_cache_last_updated652e8224e5adbb12232da00d72e1b32f=1759444360895; web_cache_last_updated0ad2d18e8d30e6d95e1c1d8604385e48=1759449180999; web_cache_last_updated58208ac66e071bd79a824469a9c06679=1759449373157; web_cache_last_updated04735373bcef794525852d1ec5c5c79a=1759449506346; d-s=1759876715; ec=enQtOTY0OTYwMjY0OTg0My1kOGEwNWQyNGM5OGQ3MzdmM2NjODA3ZGJmNDU5YmRiMzBlMjkxYWVjNTk1YmYxOTg3OWJiMmViOTQ3MDEyMThm; web_cache_last_updatedbc23576c2ec06665f1c40206504516b3=1759882038110; x=a61952af5dcace9baebd31847eaf197f.1759907564; lc=1759907612; OptanonConsent=isGpcEnabled=0&datestamp=Wed+Oct+08+2025+00%3A13%3A33+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=07597623-af2c-4821-bd16-7dcdeea0d673&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; d=xoxd-8P7QVc1SrHOz5oVTJcARdIr6LuMjuDM6j6AdIKQW96Ir8BeyrpUKTeR5YCFD9t3tsTurF%2FZvEhSMks%2BRgY1uqw30jFBJkiPpHrIE1nPmhbr5qUp1lRMjdioHFI%2FWnckFKp4VZGFUCBf2KyalVNgZllh9lY3f8G42%2F0q6Y0jjQeAh3nTvZkHfGtNruQ5MB76cdoUCHn7oAVt82M2ZbNwcFxmk",
}

WORKSPACE_URLS = {
    "modal": "https://modallabscommunity.slack.com",
    "support-driven": "https://supportdriven.slack.com",
}

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


class SlackEvaluator:
    """Evaluates search performance using Recall@K metrics."""
    
    def __init__(self, workspace: str, documents: List[Document]):
        assert workspace in XOXC, f"Unknown workspace '{workspace}', missing XOXC token"
        assert workspace in COOKIES, f"Unknown workspace '{workspace}', missing cookies"
        assert workspace in WORKSPACE_URLS, f"Unknown workspace '{workspace}', missing workspace URL"
        self.slacksearch = SlackSearch(
            token=XOXC[workspace],
            auth_mode='browser',
            cookies=COOKIES[workspace],
            workspace_url=WORKSPACE_URLS[workspace],
        )
        self.documents = documents
        
    def build_relevance_map(self, qrels: List[QRel]) -> Dict[str, Set[str]]:
        """Build mapping of query_id -> set of relevant doc_ids."""
        relevance_map = defaultdict(set)
        
        for qrel in qrels:
            if qrel.relevance > 0:  # Only count positive relevance
                relevance_map[qrel.query_id].add(qrel.doc_id)
                
        return dict(relevance_map)
    
    def build_msg_to_doc_map(self, documents: List[Document]) -> Dict[str, str]:
        """Build mapping of Slack message_id -> doc_id."""
        msg_to_doc = {}
        
        for doc in documents:
            metadata = doc.metadata or {}
            msg_id = metadata.get('message_id') or metadata.get('msg_id') or metadata.get('slack_message_id')
            if msg_id:
                msg_to_doc[msg_id] = doc.doc_id
                
        print(f"Found mapping for {len(msg_to_doc)}/{len(documents)} message_ids to doc_ids")
        print(f"There are {len(set(doc.doc_id for doc in documents))} unique doc_ids in the mapping")
        return msg_to_doc
    
    async def search_query(self, query: str, top_k: int = 20) -> List[str]:
        """Perform hybrid search and return list of doc_ids."""
        try:
            results = await self.searcher.search_file(
                self.namespace,
                query,
                top_k
            )
            
            # Extract doc_ids from results
            doc_ids = []
            for result in results:
                # Adapt this based on your result format
                doc_id = result.get('id') or result.get('doc_id')
                if doc_id:
                    doc_ids.append(doc_id)
                    
            return doc_ids
            
        except Exception as e:
            print(f"Error searching query: {e}")
            return []

    async def search_slack(self, query: str, top_k: int = 20) -> List[str]:
        """Perform Slack search and return list of message IDs."""
        try:
            keywords = await query_reduce_openai(query)
            results = self.slacksearch.search(keywords, search_type="messages", count=top_k)
            print(f"Slack search for '{query[:30]}...' with keywords '{keywords}' returned {len(results.matches)} results")
            message_ids = []
            for match in results.matches:
                text, ts = match.get('text', ''), float(match.get('ts', '0'))
                mid = generate_deterministic_uuid(text, ts)
                print(f" ===> generated message ID = {mid} for ts={ts}, text={repr(text)[:100]}...")
                message_ids.append(mid)
            return message_ids
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
    
    async def evaluate_slack(
        self, 
        queries: List[Query], 
        qrels: List[QRel],
        k_values: List[int] = None
    ) -> Dict:
        """Evaluate search performance across all queries."""
        if k_values is None:
            k_values = [5, 10, 20]
            
        print(f"\n🔍 Evaluating Slack API search performance...")
        print(f"   Queries: {len(queries)}")
        print(f"   Metrics: Recall@{k_values}")
        
        # Build relevance map
        relevance_map = self.build_relevance_map(qrels)

        # Build message_id to doc_id map
        msg_to_doc_map = self.build_msg_to_doc_map(self.documents)
        
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
            
            # Perform Slack API keyword search
            slack_docs = await self.search_slack(query.text, max(k_values))

            # Convert back to doc_ids
            retrieved_docs = [msg_to_doc_map.get(mid, 'N/A') for mid in slack_docs]

            # print(f"Query {query.text[:50]}... | Retrieved {len(retrieved_docs)} docs, {len(relevant_docs)} relevant")
            print(f"   Retrieved doc IDs: {retrieved_docs}")
            
            # Calculate recall at different K values
            query_recall = {}
            for k in k_values:
                recall = self.calculate_recall_at_k(retrieved_docs, relevant_docs, k)
                recall_scores[k].append(recall)
                query_recall[f'recall@{k}'] = recall
            
            query_results.append({
                'query_id': query.query_id,
                'query_text': query.text,
                'relevant_docs_count': len(relevant_docs),
                'retrieved_docs': retrieved_docs[:max(k_values)],
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
        print(f"\n✅ Slack API Search evaluation complete!")
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
    output_file: str = None
):
    """Main evaluation pipeline."""

    namespace = f"{workspace}-slack-eval"

    print(f"🚀 Starting Evaluation of Search via Slack API")
    print(f"   Workspace: {workspace}")
    print(f"   Data directory: {data_dir}")
    # print(f"   Namespace: {namespace}")
    print(f"   Keyword Reduction Provider: {provider}")
    
    # Load data
    loader = JSONLDataLoader(data_dir)
    documents = loader.load_documents()
    queries = loader.load_queries()
    qrels = loader.load_qrels()

    # Evaluate search
    evaluator = SlackEvaluator(workspace, documents)
    results = await evaluator.evaluate_slack(queries, qrels)

    
    # Save results
    if output_file is None:
        output_file = f"results/{namespace}_evaluation.json"
    evaluator.save_results(results, output_file)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Slack API search performance with and without zerank"
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
        "--output",
        help="Output file for results (default: results/<namespace>_evaluation.json)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(
        args.data_dir,
        args.workspace,
        args.provider,
        args.output
    ))