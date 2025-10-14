import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import List, Dict, Any

from slack_search import SlackSearch


def load_data():
    """Load queries, qrels, documents, and mappings"""
    # Load queries
    queries = {}
    with open("./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/queries.jsonl") as f:
        for line in f:
            if "{" not in line:
                continue
            query = json.loads(line)
            queries[query["id"]] = query

    # Load qrels
    qrels_by_query_id = {}
    with open("./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/qrels.jsonl") as f:
        for line in f:
            if "{" not in line:
                continue
            qrel = json.loads(line)
            query_id = qrel["query_id"]
            if query_id not in qrels_by_query_id:
                qrels_by_query_id[query_id] = []
            qrels_by_query_id[query_id].append(qrel)

    # Load documents
    documents = {}
    message_id_to_document_id = {}
    with open("./synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/documents.jsonl") as f:
        for line in f:
            j = json.loads(line)
            documents[j["id"]] = j
            message_id = j["metadata"]["message_id"]
            if message_id not in message_id_to_document_id:
                message_id_to_document_id[message_id] = set()
            message_id_to_document_id[message_id].add(j["id"])

    # Load timestamp to message ID mapping
    with open("timestamp_to_message_id.json") as f:
        timestamp_to_message_id = json.load(f)

    return queries, qrels_by_query_id, documents, message_id_to_document_id, timestamp_to_message_id


def rrf(all_rankings: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion to combine multiple rankings"""
    doc_id_to_score = defaultdict(float)
    for ranking in all_rankings:
        for i, doc in enumerate(ranking):
            doc_id_to_score[doc] += 1 / (k + i + 1)
    doc_id_and_scores = list(doc_id_to_score.items())
    doc_id_and_scores.sort(key=lambda x: -x[1])
    return [doc_id for doc_id, score in doc_id_and_scores]


async def execute_searches(search_client: SlackSearch, search_terms: List[str]) -> List[Dict]:
    """Execute multiple searches and return results"""
    results = []
    for i, term in enumerate(search_terms):
        # Add 3 second delay between requests (except for the first one)
        if i > 0:
            await asyncio.sleep(3)
            
        try:
            result = await search_client.search_async(term, search_type="messages", count=100)
            results.append(asdict(result))
        except Exception as e:
            print(f"  ⚠️  Error searching '{term}': {e}")
            results.append({"matches": [], "total": 0})
    return results


def transform_results(
    slack_results: List[Dict],
    timestamp_to_message_id: Dict[str, str],
    message_id_to_document_id: Dict[str, set]
) -> List[str]:
    """Transform Slack results to ranked document IDs using RRF"""
    all_rankings = []

    for result in slack_results:
        ranking = []
        for match in result.get("matches", []):
            ts = match.get("ts", "")
            message_id = timestamp_to_message_id.get(ts, "")
            if message_id:
                ranking.append(message_id)
        all_rankings.append(ranking)

    # Apply RRF to get unified ranking of message IDs
    message_ids = rrf(all_rankings)

    # Convert message IDs to document IDs
    document_ids = []
    for mid in message_ids:
        for doc_id in message_id_to_document_id.get(mid, set()):
            if doc_id not in document_ids:
                document_ids.append(doc_id)

    return document_ids


def get_qrel_rank(document_ids: List[str], qrel_doc_ids: List[str]) -> int:
    """Get the rank of the first relevant document (1-indexed, 0 if not found)"""
    for i, doc_id in enumerate(document_ids):
        if doc_id in qrel_doc_ids:
            return i + 1
    return 0  # Not found


def display_query_info(query: Dict, target_content: str, step: int, previous_rank: int = None):
    """Display query and target document clearly"""
    print("\n" + "="*80)
    print(f"QUERY: {query['query']}")
    print(f"\nTARGET DOCUMENT (what we're looking for):")
    print("-" * 40)
    # Show first 500 chars of content, properly handling newlines
    content_preview = target_content[:500].replace('\n', ' ')
    if len(target_content) > 500:
        content_preview += "..."
    print(content_preview)
    print("-" * 40)

    if step > 0 and previous_rank:
        if previous_rank == 0:
            print(f"\nPrevious attempt: NOT FOUND in top results")
        else:
            print(f"\nPrevious attempt: Rank {previous_rank}")

    print(f"\nStep {step + 1}")
    print("="*80)


def save_results_for_inspection(document_ids: List[str], documents: Dict[str, Dict], qrel_doc_ids: List[str], query: Dict) -> str:
    """Save top results to temp file for inspection"""
    import tempfile
    import subprocess
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(f"Query: {query['query']}\n")
        f.write(f"Query ID: {query['id']}\n")
        f.write("=" * 80 + "\n\n")
        
        # Show all results
        for i, doc_id in enumerate(document_ids):
            doc = documents.get(doc_id, {})
            is_target = doc_id in qrel_doc_ids
            
            f.write(f"\n{'='*60}\n")
            f.write(f"RANK {i+1}: {'🎯 TARGET DOCUMENT' if is_target else ''}\n")
            f.write(f"Doc ID: {doc_id}\n")
            f.write("-" * 60 + "\n")
            
            content = doc.get("content", "")
            # Show first 500 chars of content
            if len(content) > 500:
                f.write(content[:500] + "...\n")
            else:
                f.write(content + "\n")
            
            # Also show metadata
            metadata = doc.get("metadata", {})
            f.write(f"\nMetadata: {json.dumps(metadata, indent=2)}\n")
        
        
        return f.name


async def manual_agent_loop(
    query: Dict,
    qrel_doc_ids: List[str],
    target_content: str,
    search_client: SlackSearch,
    timestamp_to_message_id: Dict[str, str],
    message_id_to_document_id: Dict[str, set],
    documents: Dict[str, Dict]
) -> Dict[str, Any]:
    """Run the manual agent loop for a single query"""
    MAX_STEPS = None  # No limit in manual mode
    steps = []
    found = False
    step = 0

    while True:
        # Display query info
        previous_rank = steps[-1]["qrel_rank"] if steps else None
        display_query_info(query, target_content, step, previous_rank)

        # Get user input
        print("\nEnter search terms as JSON list (e.g., [\"modal error\", \"connection issue\"])")
        print("Commands: 'skip' (skip query), 'quit' (exit), 'done' (mark complete)")
        user_input = input("> ").strip()

        if user_input.lower() == 'skip':
            return None
        if user_input.lower() == 'quit':
            sys.exit(0)
        if user_input.lower() == 'done':
            break

        # Parse search terms
        try:
            search_terms = json.loads(user_input)
            if not isinstance(search_terms, list):
                print("❌ Please provide a JSON list of search terms")
                continue
        except json.JSONDecodeError:
            print("❌ Invalid JSON. Please provide a list like [\"term1\", \"term2\"]")
            continue

        # Get optional reasoning
        reasoning = input("Reasoning (optional, press Enter to skip): ").strip()

        # Execute searches
        print(f"\n🔍 Searching for: {search_terms}")
        slack_results = await execute_searches(search_client, search_terms)

        # Transform results
        document_ids = transform_results(slack_results, timestamp_to_message_id, message_id_to_document_id)

        # Get rank of qrel document
        qrel_rank = get_qrel_rank(document_ids, qrel_doc_ids)
        
        # Calculate recall@20
        recall_at_20 = 1 if qrel_rank > 0 and qrel_rank <= 20 else 0

        # Record step
        steps.append({
            "search_calls": search_terms,
            "reasoning": reasoning,
            "qrel_rank": qrel_rank,
            "recall_at_20": recall_at_20
        })

        # Display result
        if qrel_rank == 0:
            print(f"❌ Target document NOT FOUND in top {len(document_ids)} results")
        elif qrel_rank == 1:
            print(f"✅ SUCCESS! Target document found at rank 1!")
            found = True
        elif qrel_rank <= 20:
            print(f"📊 Target document found at rank {qrel_rank} (within recall@20)")
        else:
            print(f"📊 Target document found at rank {qrel_rank} (outside recall@20)")
        
        # Save results for inspection
        if document_ids:
            results_file = save_results_for_inspection(document_ids, documents, qrel_doc_ids, query)
            print(f"\n📄 Results saved to: {results_file}")
            
            # Ask if user wants to view results
            view = input("View results in vim? (y/n): ").strip().lower()
            if view == 'y':
                import subprocess
                subprocess.call(['vim', results_file])
        
        # Auto-exit if found at rank 1
        if qrel_rank == 1:
            break
        
        step += 1

    # Calculate best recall@20 across all attempts
    best_recall_at_20 = max((step.get("recall_at_20", 0) for step in steps if "recall_at_20" in step), default=0)
    
    return {
        "query": query["query"],
        "query_id": query["id"],
        "steps": steps,
        "found": found,
        "best_recall_at_20": best_recall_at_20
    }


async def main():
    """Main function to run the manual agent simulator"""
    # Parse command line arguments
    if len(sys.argv) > 1:
        n_queries = int(sys.argv[1])
        offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    else:
        n_queries = int(input("How many queries to test? "))
        offset_input = input("Offset to start from (default 0): ").strip()
        offset = int(offset_input) if offset_input else 0

    print("Loading data...")
    queries, qrels_by_query_id, documents, message_id_to_document_id, timestamp_to_message_id = load_data()

    # Initialize Slack search client
    print("Initializing Slack search...")
    search_client = SlackSearch(
        token="xoxc-3052645262231-9689129827878-9704176242193-ede9e23190f6b136aceaab8e42bd414808ddf9032c0d33aa33bcb9ae4410a5d8",
        auth_mode='browser',
        cookies='b=.693510bbd0d5677b628fff03f268acf4; d-s=1759520448; utm=%7B%7D; x=693510bbd0d5677b628fff03f268acf4.1760409378; shown_ssb_redirect_page=1; shown_download_ssb_modal=1; show_download_ssb_banner=1; no_download_ssb_banner=1; tz=-420; web_cache_last_updated5f1c2806a2541abd794aa08422f95de2=1760409445559; lc=1760410519; d=xoxd-HhQFLWY%2B0vp3R1qj1eDXWlR3Jj4kuvHRdPdUHUs%2FYKbKTAKhPdpUM%2BuR5EKcQtJIw7nmeL57HKM%2F3aY6FwNEf%2Fb6hcGBjKcHrDrooNTZhEJ0GI92aIlwJktXkV9amDk02Y1HsfBBYD6vp7UPpxHwCa%2Fq6zyfupwaHEFOn0Y5Th6CkrM1GL4aGy1GjWol6auPBfkOiBe5OQM%2B5EsIJZyVOdnWR11I; web_cache_last_updated34a7ff4d0b036d3d72ee8717822ef770=1760411388730',
        workspace_url='https://modallabscommunity.slack.com'
    )

    # Get all queries in sequential order
    query_ids = sorted(list(queries.keys()))
    
    # Apply offset and limit
    start_idx = offset
    end_idx = min(offset + n_queries, len(query_ids))
    selected_query_ids = query_ids[start_idx:end_idx]
    
    if offset > 0:
        print(f"Starting from query {offset + 1} (skipping first {offset} queries)")

    # Process queries
    traces = []
    completed = 0

    for i, query_id in enumerate(selected_query_ids):
        query = queries[query_id]
        qrels = qrels_by_query_id.get(query_id, [])

        if not qrels:
            print(f"⚠️  No qrel found for query {query_id}, skipping...")
            continue

        # Get target document content
        qrel_doc_ids = [qrel["document_id"] for qrel in qrels]
        target_doc = documents.get(qrel_doc_ids[0], {})
        target_content = target_doc.get("content", "")

        print(f"\n\n{'='*80}")
        actual_index = i + offset + 1
        print(f"Query {actual_index}/{len(query_ids)} (Session: {i+1}/{len(selected_query_ids)}, Completed: {completed})")
        print(f"{'='*80}")

        # Run agent loop
        trace = await manual_agent_loop(
            query, qrel_doc_ids, target_content, search_client,
            timestamp_to_message_id, message_id_to_document_id, documents
        )

        if trace:  # None if skipped
            traces.append(trace)
            completed += 1

            # Save traces after each completion
            with open("manual_agent_traces.jsonl", "w") as f:
                for t in traces:
                    f.write(json.dumps(t) + "\n")

            if completed >= n_queries:
                break

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Started at offset: {offset}")
    print(f"To continue from where you left off, use: python {sys.argv[0]} {n_queries} {completed + offset}")
    print(f"Queries attempted: {completed}/{n_queries}")
    
    if traces:
        success_count = sum(1 for t in traces if t["found"])
        recall_20_count = sum(1 for t in traces if t.get("best_recall_at_20", 0) == 1)
        print(f"Success rate (rank 1): {success_count}/{len(traces)} ({100*success_count/len(traces):.1f}%)")
        print(f"Recall@20: {recall_20_count}/{len(traces)} ({100*recall_20_count/len(traces):.1f}%)")

        avg_steps = sum(len(t["steps"]) for t in traces) / len(traces)
        print(f"Average steps per query: {avg_steps:.1f}")

    print(f"\nTraces saved to: manual_agent_traces.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
