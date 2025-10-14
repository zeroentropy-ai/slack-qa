import asyncio
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from typing import List, Dict, Any
import openai
import difflib

from slack_search import SlackSearch
from ai import AIRerankModel, ai_rerank


# Rerank model configuration
RERANK_MODEL = AIRerankModel(
    company="zeroentropy",
    model="zerank-1",
)

# Agent configuration
SYSTEM_PROMPT = """Your job is to find a single Slack message that answers a query.

Given a user query and optionally the target document content, generate keyword-based search calls that would help find relevant Slack messages.

IMPORTANT: You can generate up to 10 search calls (generally 3-4 should be usual). These will be searched INDEPENDENTLY and then combined using Reciprocal Rank Fusion (RRF) to produce a final ranking. This means:
- Each query is searched separately. Presumably it does something similar to Bm25 (results contain all keywords in the search query)
- Results from all calls are merged using RRF
- Documents appearing in multiple search results will get ranked higher

Each search query should be 1-3 keywords that someone might use to find the information from the slack search box.

Think about:
- Key technical terms or product names mentioned
- Actions or problems described
- Specific error messages or codes
- Related concepts that might appear in the same conversation
- Different ways people might phrase the same concept

Output ONLY a JSON object with this format:
{"search": ["modal error", "connection timeout", "gpu memory", "modal timeout issue"]}  // Up to 4 calls
"""

USER_PROMPT_TEMPLATE = """User query: {query}

Target document:
{target_preview}

Previous attempts and their results:
{previous_attempts}

You have a maximum of 10 search attempts to get the target document to rank 1.

Generate up to 10 (but median of 3) new search queries. Remember: Multiple queries will be combined with RRF, so diverse queries covering different aspects can be very effective.

If previous attempts had high ranks (far from 1), try significantly different keywords or approaches.
If previous attempts had low ranks (close to 1), examine the top results shown above and make small refinements to push the target to rank 1.

Search attempts so far: {current_step}/10
"""


class RateLimiter:
    """Simple rate limiter to ensure we don't exceed 20 requests per minute"""
    def __init__(self, max_per_minute=20):
        self.max_per_minute = max_per_minute
        self.min_interval = 60.0 / max_per_minute  # seconds between requests
        self.last_request_time = 0

    async def wait_if_needed(self):
        """Wait if necessary to maintain rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            print(f"  ⏱️  Rate limiting: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()


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


async def execute_searches(rate_limiter: RateLimiter, search_client: SlackSearch, search_terms: List[str]) -> List[Dict]:
    """Execute multiple searches and return results"""
    results = []
    for i, term in enumerate(search_terms):
        # Add 3 second delay between requests (except for the first one)
        try:

            await rate_limiter.wait_if_needed()
            result = await search_client.search_async(term, search_type="messages", count=100)
            results.append(asdict(result))
        except Exception as e:
            print(f"  ⚠️  Error searching '{term}': {e}")
            results.append({"matches": [], "total": 0})
    return results


def transform_results(
    slack_results: List[Dict],
    timestamp_to_message_id: Dict[str, str],
    message_id_to_document_id: Dict[str, set],
    qrel_doc_ids: List[str] = None
) -> tuple[List[str], List[int]]:
    """Transform Slack results to ranked document IDs using RRF
    Returns: (document_ids, individual_query_ranks)
    """
    all_rankings = []
    individual_query_ranks = []

    for i, result in enumerate(slack_results):
        ranking = []
        matches = result.get("matches", [])
        print(f"    Search {i+1} returned {len(matches)} matches")
        
        # Build ranking for this query
        for match in matches:
            ts = match.get("ts", "")
            message_id = timestamp_to_message_id.get(ts, "")
            if message_id:
                ranking.append(message_id)
        all_rankings.append(ranking)
        
        # Find rank of target in this individual query if qrel_doc_ids provided
        query_rank = 0
        if qrel_doc_ids:
            # Convert message IDs to document IDs for this query
            query_doc_ids = []
            for mid in ranking:
                for doc_id in message_id_to_document_id.get(mid, set()):
                    if doc_id not in query_doc_ids:
                        query_doc_ids.append(doc_id)
            
            # Find rank of qrel document
            for rank_idx, doc_id in enumerate(query_doc_ids):
                if doc_id in qrel_doc_ids:
                    query_rank = rank_idx + 1
                    break
        
        individual_query_ranks.append(query_rank)

    # Apply RRF to get unified ranking of message IDs
    message_ids = rrf(all_rankings)

    # Convert message IDs to document IDs
    document_ids = []
    for mid in message_ids:
        for doc_id in message_id_to_document_id.get(mid, set()):
            if doc_id not in document_ids:
                document_ids.append(doc_id)

    return document_ids, individual_query_ranks


def get_qrel_rank(document_ids: List[str], qrel_doc_ids: List[str]) -> int:
    """Get the rank of the first relevant document (1-indexed, 0 if not found)"""
    for i, doc_id in enumerate(document_ids):
        if doc_id in qrel_doc_ids:
            return i + 1
    return 0  # Not found


async def generate_agent_action(
    query: str,
    target_preview: str,
    previous_attempts: str,
    current_step: int,
    openai_client: openai.AsyncOpenAI,
    rate_limiter: RateLimiter
) -> Dict[str, Any]:
    """Generate search queries using OpenAI"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    query=query,
                    target_preview=target_preview,
                    previous_attempts=previous_attempts,
                    current_step=current_step
                )}
            ],
            temperature=0.7,
            max_tokens=100
        )

        # Parse the response
        content = response.choices[0].message.content.strip()
        # Try to extract JSON object from the response
        if "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
            try:
                action = json.loads(json_str)
                # Ensure it's a dict, not a list
                if isinstance(action, dict):
                    return action
                else:
                    print(f"  ⚠️  AI returned a list instead of object: {action}")
                    return {"search": []}
            except json.JSONDecodeError as je:
                print(f"  ⚠️  JSON parsing error: {je}")
                print(f"  ⚠️  Failed to parse: {json_str}")
                return {"search": []}
        else:
            print(f"  ⚠️  No JSON object found in AI response: {content}")
            return {"search": []}

    except Exception as e:
        print(f"  ⚠️  Error generating queries: {e}")
        return {"search": []}


async def automated_agent_loop(
    query: Dict,
    qrel_doc_ids: List[str],
    target_content: str,
    search_client: SlackSearch,
    timestamp_to_message_id: Dict[str, str],
    message_id_to_document_id: Dict[str, set],
    documents: Dict[str, Dict],
    openai_client: openai.AsyncOpenAI,
    rate_limiter: RateLimiter,
    rerank_model: AIRerankModel
) -> Dict[str, Any]:
    """Run the automated agent loop for a single query"""
    MAX_STEPS = 10
    steps = []
    found = False
    step = 0

    # Prepare target preview
    target_preview = target_content

    while step < MAX_STEPS and not found:
        # Format previous attempts with results
        previous_attempts = ""
        for i, prev_step in enumerate(steps):
            search_calls = prev_step["search_calls"]
            rank = prev_step["qrel_rank"]
            individual_ranks = prev_step.get("individual_query_ranks", [])
            rank_str = f"rank {rank}" if rank > 0 else "NOT FOUND"
            previous_attempts += f"\nAttempt {i+1}: {json.dumps(search_calls)} -> Target document {rank_str} (RRF combined)\n"
            
            # Show individual query performance
            if individual_ranks and len(individual_ranks) == len(search_calls):
                previous_attempts += "Individual query results:\n"
                for j, (query, ind_rank) in enumerate(zip(search_calls, individual_ranks)):
                    ind_rank_str = f"rank {ind_rank}" if ind_rank > 0 else "NOT FOUND"
                    previous_attempts += f"  Query '{query}': {ind_rank_str}\n"
            
            # Show top 5 RRF results
            top_results = prev_step.get("top_results", [])
            if top_results:
                previous_attempts += "\nTop 5 results after RRF + reranking:\n"
                for j, doc_id in enumerate(top_results):
                    doc = documents.get(doc_id, {})
                    content = doc.get("content", "")[:150].replace('\n', ' ')
                    is_target = doc_id in qrel_doc_ids
                    previous_attempts += f"  {j+1}.{' [TARGET]' if is_target else ''} {content}...\n"

        if not previous_attempts:
            previous_attempts = "No previous attempts yet."

        # Generate agent action (this may wait for rate limiting)
        print("  🤖 Deciding next action...")
        action = await generate_agent_action(
            query["query"],
            target_preview,
            previous_attempts,
            step + 1,  # Use current step count (only searches)
            openai_client,
            rate_limiter
        )
        
        # Ensure action is a dict
        if not isinstance(action, dict):
            print(f"  ⚠️  Invalid action format, skipping...")
            continue
        
        # Handle search action
        search_queries = action.get("search", [])
        if not search_queries:
            print("  ⚠️  No search queries generated, skipping...")
            continue
        
        # Action will be performed - show step number
        step += 1
        print(f"\n  Step {step}/{MAX_STEPS}")
        
        print(f"  🔍 Searching for: {search_queries}")

        # Execute searches
        slack_results = await execute_searches(rate_limiter, search_client, search_queries)

        # Transform results
        document_ids, individual_query_ranks = transform_results(
            slack_results, timestamp_to_message_id, message_id_to_document_id, qrel_doc_ids
        )
        
        # Rerank all documents if we have results
        if document_ids:
            # Get document texts for reranking
            texts_to_rerank = []
            for doc_id in document_ids:
                doc = documents.get(doc_id, {})
                content = doc.get("content", "")
                texts_to_rerank.append(content)
            
            # Rerank using the model
            print(f"  🔄 Reranking {len(document_ids)} results...")
            rerank_scores = await ai_rerank(
                model=rerank_model,
                query=query["query"],
                texts=texts_to_rerank,
            )
            
            # Sort documents by rerank scores
            doc_score_pairs = list(zip(document_ids, rerank_scores))
            doc_score_pairs.sort(key=lambda x: -x[1])  # Sort by score descending
            
            # Update document_ids with reranked order
            document_ids = [doc_id for doc_id, _ in doc_score_pairs]

        # Get rank of qrel document
        qrel_rank = get_qrel_rank(document_ids, qrel_doc_ids)
        
        # Calculate recall@20
        recall_at_20 = 1 if qrel_rank > 0 and qrel_rank <= 20 else 0

        # Record step
        steps.append({
            "search_calls": search_queries,
            "individual_query_ranks": individual_query_ranks,
            "qrel_rank": qrel_rank,
            "recall_at_20": recall_at_20,
            "top_results": document_ids[:5]  # Store top 5 for showing to LLM
        })

        # Display result
        if qrel_rank == 0:
            print(f"  ❌ Target document NOT FOUND in top {len(document_ids)} results")
        elif qrel_rank == 1:
            print(f"  ✅ SUCCESS! Target document found at rank 1!")
            found = True
            break
        elif qrel_rank <= 20:
            print(f"  📊 Target document found at rank {qrel_rank} (within recall@20)")
        else:
            print(f"  📊 Target document found at rank {qrel_rank} (outside recall@20)")
        
        # Display individual query performance
        if individual_query_ranks and len(individual_query_ranks) == len(search_queries):
            print(f"  📋 Individual query ranks: {dict(zip(search_queries, individual_query_ranks))}")

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
    """Main function to run the automated agent"""
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

    # Initialize clients
    print("Initializing clients...")
    search_client = SlackSearch(
        token="xoxc-3052645262231-9689129827878-9704176242193-ede9e23190f6b136aceaab8e42bd414808ddf9032c0d33aa33bcb9ae4410a5d8",
        auth_mode='browser',
        cookies='b=.693510bbd0d5677b628fff03f268acf4; d-s=1759520448; utm=%7B%7D; x=693510bbd0d5677b628fff03f268acf4.1760409378; shown_ssb_redirect_page=1; shown_download_ssb_modal=1; show_download_ssb_banner=1; no_download_ssb_banner=1; tz=-420; web_cache_last_updated5f1c2806a2541abd794aa08422f95de2=1760409445559; lc=1760410519; d=xoxd-HhQFLWY%2B0vp3R1qj1eDXWlR3Jj4kuvHRdPdUHUs%2FYKbKTAKhPdpUM%2BuR5EKcQtJIw7nmeL57HKM%2F3aY6FwNEf%2Fb6hcGBjKcHrDrooNTZhEJ0GI92aIlwJktXkV9amDk02Y1HsfBBYD6vp7UPpxHwCa%2Fq6zyfupwaHEFOn0Y5Th6CkrM1GL4aGy1GjWol6auPBfkOiBe5OQM%2B5EsIJZyVOdnWR11I; web_cache_last_updated34a7ff4d0b036d3d72ee8717822ef770=1760411388730',
        workspace_url='https://modallabscommunity.slack.com'
    )

    openai_client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(max_per_minute=18)  # Leave some buffer

    # Get all queries in sequential order
    query_ids = sorted(list(queries.keys()))

    # Apply offset and limit
    start_idx = offset
    end_idx = min(offset + n_queries, len(query_ids))
    selected_query_ids = query_ids[start_idx:end_idx]

    if offset > 0:
        print(f"Starting from query {offset + 1} (skipping first {offset} queries)")

    # Load existing traces if any
    traces_dict = {}
    try:
        with open("automated_agent_traces.json", "r") as f:
            traces_dict = json.load(f)
        print(f"Loaded {len(traces_dict)} existing traces")
    except FileNotFoundError:
        print("No existing traces found, starting fresh")
    
    # Process queries
    completed_in_session = 0

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

        print(f"\n{'='*80}")
        actual_index = i + offset + 1
        print(f"Query {actual_index}/{len(query_ids)} (Session: {i+1}/{len(selected_query_ids)})")
        print(f"Query: {query['query']}")
        print(f"{'='*80}")

        # Run agent loop
        trace = await automated_agent_loop(
            query, qrel_doc_ids, target_content, search_client,
            timestamp_to_message_id, message_id_to_document_id, documents,
            openai_client, rate_limiter, RERANK_MODEL
        )

        # Update traces dict
        traces_dict[query_id] = trace
        completed_in_session += 1
        
        # Save traces after each completion
        with open("automated_agent_traces.json", "w") as f:
            json.dump(traces_dict, f, indent=2)

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Started at offset: {offset}")
    print(f"To continue from where you left off, use: python {sys.argv[0]} {n_queries} {completed_in_session + offset}")
    print(f"Completed in this session: {completed_in_session}")
    print(f"Total traces in file: {len(traces_dict)}")
    
    # Calculate stats for queries attempted in this session
    session_query_ids = selected_query_ids[:completed_in_session]
    session_traces = [traces_dict[qid] for qid in session_query_ids if qid in traces_dict]
    
    if session_traces:
        success_count = sum(1 for t in session_traces if t["found"])
        recall_20_count = sum(1 for t in session_traces if t.get("best_recall_at_20", 0) == 1)
        
        print(f"\nSession stats (queries attempted: {len(session_traces)}):")
        print(f"  Success rate (rank 1): {success_count}/{len(session_traces)} ({100*success_count/len(session_traces):.1f}%)")
        print(f"  Recall@20: {recall_20_count}/{len(session_traces)} ({100*recall_20_count/len(session_traces):.1f}%)")
        
        avg_steps = sum(len(t["steps"]) for t in session_traces) / len(session_traces)
        print(f"  Average steps per query: {avg_steps:.1f}")
    
    # Overall stats for all traces in file
    all_traces = list(traces_dict.values())
    if all_traces and len(all_traces) > len(session_traces):
        print(f"\nOverall stats (all {len(all_traces)} queries in file):")
        success_count_all = sum(1 for t in all_traces if t["found"])
        recall_20_count_all = sum(1 for t in all_traces if t.get("best_recall_at_20", 0) == 1)
        
        print(f"  Success rate (rank 1): {success_count_all}/{len(all_traces)} ({100*success_count_all/len(all_traces):.1f}%)")
        print(f"  Recall@20: {recall_20_count_all}/{len(all_traces)} ({100*recall_20_count_all/len(all_traces):.1f}%)")
    
    print(f"\nTraces saved to: automated_agent_traces.json")


if __name__ == "__main__":
    asyncio.run(main())
