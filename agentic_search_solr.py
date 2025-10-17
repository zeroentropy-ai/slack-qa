import asyncio
import json
import sys
import time
import os
from collections import defaultdict
from typing import List, Dict, Any
import openai
from tqdm.asyncio import tqdm

# Import Solr masked search instead of Slack API
sys.path.append('mock-slack')
from masked_solr_library import masked_solr_search
from ai import AIRerankModel, ai_rerank


# Rerank model configuration
RERANK_MODEL = AIRerankModel(
    company="zeroentropy",
    model="zerank-1",
)

# Solr configuration
SOLR_COLLECTION = "training-slack"

# Agent configuration
SYSTEM_PROMPT = """Your job is to find a set of 5-7 shortest possible keyword search queries that lead to a target document being found in the top 20 results. Each keyword search must use diverse search terms.

As input you will be given a question by the user. The question is in a conversational form, something they would ask a chat bot, but the search queries
you generate will be run on a search server which is running the following algorithm.

If search has N words:

1. it runs lucene algorithm looking for all documents which contain all N words
2. if no matches were found, it decrements N and repeats from step 1, else returns the matches.

Some things to note:

1. **Specificity trades off with recall**: using salient words (and only those), technical terms, proper nouns, ids and numbers in the question will be VERY good since it is more likely to be in the target document. However too many specific words or restrictions in one query will lead to no results.

2. **Length of query**: if the query is short with words that are not specific, it may surface too many results causing the target document to go beyond top 20. A good strategy is to try queries of 1, 2, and 3 keywords in one go, and see which ones hit, and keep the shortest one that worked and any other searches that did better for the next step. Note that if a 2 word query matches but ranks > 20, adding a 3rd specific term might push it to to a better rank. Do not use filler words like of, is, the, in, etc. These will be disregarded by the search, we are only interested in looking for uncommon words.

3. **Order doesn't matter**: Lucene searches for documents containing all terms regardless of order or proximity. Don't worry about word order - focus on selecting the right terms.

4. **Think like the document author**: Use terminology that would likely appear IN the target document, not just ABOUT the topic. If searching for a Python tutorial, "python tutorial beginner" is better than "learning programming basics".

EXAMPLE Question: "How can the Deflection Gap concept help in evaluating knowledge failures and improving knowledge quality, trust, and coverage?"

you should start by searching ["deflection gap", "knowledge quality"] since "deflection gap" seems to refer to have a specific meaning in this case (see title case) and it's not likely there will be too many documents mentioning it.  "knowledge quality" seems to be the general concept being searched for, so it is a good starting point. In the next step if "knowledge quality" is not found, you can search for other peripheral terms like "knowledge failure". If there are too many results and hence the target document is not being found, then think about adding some more words like "failure" -- which in this case is the main property the user is looking for.

You can generate up to 10 calls per step, and they will all be tried together.

A trace of previous search attempts and resulting ranks of target will be given  so that you can improve on your next set of queries.. A null rank means the document was not found.
Output ONLY a JSON object with this format:
{"thinking": "your thinking process to make the choices", "search": ["pytorch error", "connection timeout", "gpu memory", "pytorch timeout issue"]}
"""

USER_PROMPT_TEMPLATE = """User query: {query}

Previous attempts and their results:
{previous_attempts}
"""


class RateLimiter:
    """Simple rate limiter - simplified for Solr usage"""
    def __init__(self, max_per_minute=60):
        self.max_per_minute = max_per_minute
        self.min_interval = 60.0 / max_per_minute  # seconds between requests
        self.last_request_time = 0

    async def wait_if_needed(self):
        """Wait if necessary to maintain rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()


def safe_save_json(data: Dict, filename: str):
    """Safely save JSON data to file using atomic write"""
    temp_filename = f"{filename}.tmp"
    try:
        # Write to temporary file
        with open(temp_filename, "w") as f:
            json.dump(data, f, indent=2)
        
        # Atomically rename temp file to target file
        os.replace(temp_filename, filename)
    except Exception as e:
        # Clean up temp file if something went wrong
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise e


def load_data():
    """Load queries and documents from training data"""
    # Load queries and target documents from training_data_step_0.json
    queries = {}
    documents = {}
    target_doc_by_query = {}
    
    with open("./mock-slack/training_data_step_0.json") as f:
        training_data = json.load(f)
    
    for item in training_data:
        query_id = item["query_id"]
        document_id = item["document_id"]
        
        # Create query object with target document ID
        queries[query_id] = {
            "id": query_id,
            "query": item["question"],
            "target_document_id": document_id
        }
        
        # Create document object
        documents[document_id] = {
            "id": document_id,
            "content": item["content"],
            "metadata": item.get("metadata", {})
        }
        
        # Store target document mapping
        target_doc_by_query[query_id] = document_id

    return queries, documents, target_doc_by_query


def rrf(all_rankings: List[List[str]], k: int = 60) -> List[str]:
    """Reciprocal Rank Fusion to combine multiple rankings"""
    doc_id_to_score = defaultdict(float)
    for ranking in all_rankings:
        for i, doc in enumerate(ranking):
            doc_id_to_score[doc] += 1 / (k + i + 1)
    doc_id_and_scores = list(doc_id_to_score.items())
    doc_id_and_scores.sort(key=lambda x: -x[1])
    return [doc_id for doc_id, score in doc_id_and_scores]


async def execute_searches(rate_limiter: RateLimiter, search_terms: List[str], cache: Dict[str, List[str]] = None) -> List[Dict]:
    """Execute multiple searches using Solr masked search with caching"""
    if cache is None:
        cache = {}
    
    results = []
    for i, term in enumerate(search_terms):
        # Check cache first
        if term in cache:
            # Reconstruct result dict from cached document IDs
            cached_doc_ids = cache[term]
            result_dict = {
                "search_results": [{"id": doc_id} for doc_id in cached_doc_ids],
                "keywords_matched": len(term.split()),  # Estimate based on query length
                "total": len(cached_doc_ids)
            }
            results.append(result_dict)
            continue
        
        try:
            await rate_limiter.wait_if_needed()
            
            # Use Solr masked search instead of Slack API
            docs, keywords_matched = masked_solr_search(term, SOLR_COLLECTION)
            
            # Extract document IDs for caching
            doc_ids = [doc.get("id", "") for doc in docs if doc.get("id")]
            
            # Store only document IDs in cache
            cache[term] = doc_ids
            
            # Convert Solr results to format compatible with existing code
            result_dict = {
                "search_results": docs,
                "keywords_matched": keywords_matched,
                "total": len(docs)
            }
            
            results.append(result_dict)
            
        except Exception as e:
            result_dict = {"search_results": [], "keywords_matched": 0, "total": 0}
            cache[term] = []  # Cache empty result
            results.append(result_dict)
    
    return results


def transform_results(
    solr_results: List[Dict],
    target_doc_id: str = None
) -> tuple[List[str], List[int], List[List[str]]]:
    """Transform Solr results to ranked document IDs using RRF
    Returns: (document_ids, individual_query_ranks, individual_query_results)
    """
    all_rankings = []
    individual_query_ranks = []
    individual_query_results = []  # Top 10 doc IDs for each query

    for i, result in enumerate(solr_results):
        ranking = []
        search_results = result.get("search_results", [])
        
        # Build ranking for this query - extract document IDs from Solr results
        for doc in search_results:
            doc_id = doc.get("id", "")
            if doc_id:
                ranking.append(doc_id)
        
        all_rankings.append(ranking)
        
        # Get top 10 doc IDs for this query
        query_doc_ids = ranking[:10]
        individual_query_results.append(query_doc_ids)
        
        # Find rank of target document in this individual query
        query_rank = None
        if target_doc_id:
            for rank_idx, doc_id in enumerate(ranking):
                if doc_id == target_doc_id:
                    query_rank = rank_idx + 1
                    break
        
        individual_query_ranks.append(query_rank)

    # Apply RRF to get unified ranking of document IDs
    document_ids = rrf(all_rankings)

    return document_ids, individual_query_ranks, individual_query_results


def get_target_rank(document_ids: List[str], target_doc_id: str):
    """Get the rank of the target document (1-indexed, None if not found)"""
    for i, doc_id in enumerate(document_ids):
        if doc_id == target_doc_id:
            return i + 1
    return None  # Not found


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
            temperature=0.7
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
    target_doc_id: str,
    target_content: str,
    documents: Dict[str, Dict],
    openai_client: openai.AsyncOpenAI,
    rate_limiter: RateLimiter,
    rerank_model: AIRerankModel,
    search_cache: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Run the automated agent loop for a single query"""
    MAX_STEPS = 10
    steps = []
    found = False
    step = 0

    # Prepare target preview
    target_preview = target_content

    while step < MAX_STEPS:
        # Format previous attempts with results
        previous_attempts = ""
        for i, prev_step in enumerate(steps):
            search_calls = prev_step["search_calls"]
            rank = prev_step["target_rank"]
            individual_ranks = prev_step.get("individual_query_ranks", [])
            
            # Create JSON format showing query -> rank mapping
            query_rank_map = {}
            for query, ind_rank in zip(search_calls, individual_ranks):
                query_rank_map[query] = ind_rank
            
            rank_str = f"rank {rank}" if rank is not None else "NOT FOUND"
            previous_attempts += f"\nStep {i+1}: {json.dumps(query_rank_map)}\n"
            
            # Show top 10 results from RRF combination for context
            individual_results = prev_step.get("individual_query_results", [])
            if individual_results:
                # Get unique top documents from all queries for this step
                seen_docs = set()
                top_docs = []
                for query_results in individual_results:
                    for doc_id in query_results[:5]:  # Top 5 from each query
                        if doc_id not in seen_docs:
                            seen_docs.add(doc_id)
                            top_docs.append(doc_id)
                            if len(top_docs) >= 10:  # Limit to top 10 overall
                                break
                    if len(top_docs) >= 10:
                        break
                
                if top_docs:
                    previous_attempts += "  Top combined results:\n"
                    for k, doc_id in enumerate(top_docs[:10]):
                        doc = documents.get(doc_id, {})
                        content = doc.get("content", "")[:100].replace('\n', ' ')
                        is_target = doc_id == target_doc_id
                        previous_attempts += f"    {k+1}.{' [TARGET]' if is_target else ''} {content}...\n"

        if not previous_attempts:
            previous_attempts = "No previous attempts yet."

        # Generate agent action (this may wait for rate limiting)
        action = await generate_agent_action(
            query["query"] if isinstance(query, dict) else query,
            target_preview,
            previous_attempts,
            step + 1,  # Use current step count (only searches)
            openai_client,
            rate_limiter
        )
        
        # Ensure action is a dict
        if not isinstance(action, dict):
            continue
        
        # Handle search action
        search_queries = action.get("search", [])
        if not search_queries:
            continue
        
        # Action will be performed - show step number
        step += 1

        # Execute searches using Solr
        solr_results = await execute_searches(rate_limiter, search_queries, search_cache)

        # Transform results
        document_ids, individual_query_ranks, individual_query_results = transform_results(
            solr_results, target_doc_id
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
            try:
                rerank_scores = await ai_rerank(
                    model=rerank_model,
                    query=query["query"] if isinstance(query, dict) else query,
                    texts=texts_to_rerank,
                )
                
                # Sort documents by rerank scores
                doc_score_pairs = list(zip(document_ids, rerank_scores))
                doc_score_pairs.sort(key=lambda x: -x[1])  # Sort by score descending
                
                # Update document_ids with reranked order
                document_ids = [doc_id for doc_id, _ in doc_score_pairs]
                
            except Exception as e:
                # Continue with RRF results if reranking fails
                pass

        # Get rank of target document
        target_rank = get_target_rank(document_ids, target_doc_id)
        
        # Calculate recall@20
        recall_at_20 = 1 if target_rank is not None and target_rank <= 20 else 0

        # Record step
        steps.append({
            "search_calls": search_queries,
            "individual_query_ranks": individual_query_ranks,
            "individual_query_results": individual_query_results,  # Top 10 from each query
            "target_rank": target_rank,
            "recall_at_20": recall_at_20
        })

        # Track found status
        if target_rank == 1 and not found:
            found = True

    # Calculate best recall@20 across all attempts
    best_recall_at_20 = max((step.get("recall_at_20", 0) for step in steps if "recall_at_20" in step), default=0)
    
    return {
        "query": query["query"] if isinstance(query, dict) else query,
        "query_id": query["id"] if isinstance(query, dict) else "unknown",
        "steps": steps,
        "found": found,
        "best_recall_at_20": best_recall_at_20
    }


async def process_single_query(
    semaphore: asyncio.Semaphore,
    query_id: str,
    query: Dict,
    target_doc_id: str,
    target_content: str,
    documents: Dict[str, Dict],
    openai_client: openai.AsyncOpenAI,
    rate_limiter: RateLimiter,
    rerank_model: AIRerankModel,
    search_cache: Dict[str, List[str]],
    pbar: tqdm
) -> tuple[str, Dict[str, Any]]:
    """Process a single query with semaphore control"""
    async with semaphore:
        # Run agent loop (silently)
        trace = await automated_agent_loop(
            query, target_doc_id, target_content, documents,
            openai_client, rate_limiter, rerank_model, search_cache
        )
        
        # Update progress bar
        pbar.update(1)
        
        return query_id, trace


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

    queries, documents, target_doc_by_query = load_data()
    openai_client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(max_per_minute=1200)

    # Get all queries in sequential order
    query_ids = sorted(list(queries.keys()))

    # Apply offset and limit
    start_idx = offset
    end_idx = min(offset + n_queries, len(query_ids))
    selected_query_ids = query_ids[start_idx:end_idx]

    # Load existing traces if any
    traces_dict = {}
    try:
        with open("automated_agent_traces_solr.json", "r") as f:
            traces_dict = json.load(f)
    except FileNotFoundError:
        pass
    
    # Create search cache
    search_cache = {}
    
    # Prepare tasks for parallel processing
    semaphore = asyncio.Semaphore(32)  # Limit to 32 concurrent operations
    tasks = []
    valid_queries = []
    
    for i, query_id in enumerate(selected_query_ids):
        query = queries[query_id]
        target_doc_id = target_doc_by_query.get(query_id)

        if not target_doc_id:
            continue

        # Get target document content
        target_doc = documents.get(target_doc_id, {})
        target_content = target_doc.get("content", "")
        
        valid_queries.append((query_id, query, target_doc_id, target_content))
    
    # Create progress bar
    pbar = tqdm(total=len(valid_queries), desc="Processing queries")
    
    for query_id, query, target_doc_id, target_content in valid_queries:
        task = process_single_query(
            semaphore, query_id, query, target_doc_id, target_content, documents,
            openai_client, rate_limiter, RERANK_MODEL, search_cache, pbar
        )
        tasks.append(task)
    
    # Process all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    pbar.close()
    
    # Process results
    completed_in_session = 0
    session_success_count = 0
    session_recall_20_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            continue
            
        query_id, trace = result
        traces_dict[query_id] = trace
        completed_in_session += 1
        
        # Update running statistics
        if trace.get("found", False):
            session_success_count += 1
        if trace.get("best_recall_at_20", 0) == 1:
            session_recall_20_count += 1
    
    # Save all traces after processing
    safe_save_json(traces_dict, "automated_agent_traces_solr.json")

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
    
    print(f"\nTraces saved to: automated_agent_traces_solr.json")
    
    # Print cache statistics
    if search_cache:
        cache_size = len(search_cache)
        print(f"\nSearch cache: {cache_size} unique queries cached")


if __name__ == "__main__":
    asyncio.run(main())
