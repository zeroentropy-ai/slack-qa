#!/usr/bin/env python3
"""
Masked Solr search library - progressive keyword matching from all keywords down to single keywords
"""
import json
import requests
from typing import List, Dict, Any, Set, Tuple
from itertools import combinations

SOLR_BASE_URL = "http://localhost:8983/solr"

# Common English stop words that Slack likely filters out
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 
    'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'said', 
    'each', 'which', 'their', 'time', 'if', 'up', 'out', 'many', 'then', 'them', 
    'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'two', 
    'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 
    'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 'come', 
    'made', 'may', 'part', 'vs'
}

def remove_stop_words(words: List[str]) -> List[str]:
    """Remove stop words from a list of words"""
    return [word for word in words if word.lower() not in STOP_WORDS]

def escape_solr_special_chars(word: str) -> str:
    """
    Escape special Solr characters by wrapping the entire term in quotes.
    This handles cases like seedstring() or other terms with special characters.
    """
    # Characters that need escaping in Solr
    special_chars = set('+-&|!(){}[]^"~*?:\\/')
    
    # If the word contains any special characters, wrap in quotes
    if any(char in special_chars for char in word):
        # Escape any existing quotes and wrap in quotes
        escaped = word.replace('"', '\\"')
        return f'"{escaped}"'
    
    return word

def build_combination_query(keywords: List[str], k: int) -> str:
    """
    Build query for all combinations of k keywords from the list
    Returns OR of all k-combinations where each combination is ANDed
    """
    if k == 0 or k > len(keywords):
        return ""
    
    # Escape special characters in keywords
    escaped_keywords = [escape_solr_special_chars(word) for word in keywords]
    
    if k == 1:
        # Simple OR for single keywords
        return " OR ".join([f"content:{word}" for word in escaped_keywords])
    
    # Generate all k-combinations and AND them, then OR all combinations
    combinations_list = list(combinations(escaped_keywords, k))
    and_queries = []
    
    for combo in combinations_list:
        and_query = " AND ".join([f"content:{word}" for word in combo])
        and_queries.append(f"({and_query})")
    
    return " OR ".join(and_queries)

def masked_solr_search(search_query: str, collection: str = "slack", rows: int = 100) -> Tuple[List[Dict], int]:
    """
    Perform masked Solr search - try all keywords, then k-1, k-2, etc.
    
    Args:
        search_query: The search query string
        collection: Solr collection name (default "slack")
        rows: Maximum number of results to return
    
    Returns:
        Tuple of (documents_list, keywords_matched)
        keywords_matched indicates how many keywords were required for the match
    """
    keywords = search_query.strip().split()
    # Remove stop words
    keywords = remove_stop_words(keywords)
    
    if not keywords:
        return [], 0
    
    # Return empty result for queries with >6 words
    if len(keywords) > 6:
        return [], 0
    
    # Try from all keywords down to 1 keyword
    for k in range(len(keywords), 0, -1):
        solr_query = build_combination_query(keywords, k)
        
        params = {
            'q': solr_query,
            'rows': rows,
            'fl': 'id,score',
            'sort': 'score desc'
        }
        
        try:
            response = requests.get(f"{SOLR_BASE_URL}/{collection}/select", params=params)
            response.raise_for_status()
            
            solr_result = response.json()
            docs = solr_result['response']['docs']
            
            # If we found results, return them with the keyword count
            if docs:
                return docs, k
                
        except Exception as e:
            print(f"Error in Solr search for k={k}: {e}")
            continue
    
    # No results found even with single keywords
    return [], 0

def batch_masked_search_with_ranks(queries: List[Dict[str, str]], collection: str = "slack", rows: int = 100) -> List[Dict[str, Any]]:
    """
    Perform batch masked search and return rank of target document for each query
    
    Args:
        queries: List of dicts with "search_query" and "target_document_id" keys
        collection: Solr collection name (default "slack")
        rows: Number of results to fetch (default 100 to capture most ranks)
    
    Returns:
        List of dicts with original fields plus "masked_target_rank" and "keywords_matched"
        masked_target_rank is null if document not found in results, otherwise 1-based rank
        keywords_matched indicates minimum number of keywords needed to find results
    """
    results = []
    
    for query_info in queries:
        search_query = query_info["search_query"]
        target_doc_id = query_info["target_document_id"]
        
        # Perform masked Solr search
        docs, keywords_matched = masked_solr_search(search_query, collection, rows)
        
        # Find rank of target document
        masked_target_rank = None
        for i, doc in enumerate(docs, 1):
            if doc['id'] == target_doc_id:
                masked_target_rank = i
                break
        
        # Copy input object and add results
        result = query_info.copy()
        result["masked_target_rank"] = masked_target_rank
        result["keywords_matched"] = keywords_matched
        results.append(result)
    
    return results

def single_masked_search_with_rank(search_query: str, target_document_id: str, collection: str = "slack", rows: int = 100) -> Dict[str, Any]:
    """
    Perform single masked search and return rank of target document
    
    Args:
        search_query: The search query string
        target_document_id: ID of document to find rank for
        collection: Solr collection name (default "slack")
        rows: Number of results to fetch
    
    Returns:
        Dict with "search_query", "target_document_id", "masked_target_rank", and "keywords_matched"
    """
    return batch_masked_search_with_ranks([{
        "search_query": search_query,
        "target_document_id": target_document_id
    }], collection, rows)[0]

if __name__ == "__main__":
    # Test the masked search functionality
    test_queries = [
        {"search_query": "modal deploy gpu instance", "target_document_id": "d_test123"},
        {"search_query": "billing spike", "target_document_id": "d_54e2f7618f"},
    ]
    
    print("Testing masked Solr search:")
    print("=" * 50)
    
    for query_info in test_queries:
        result = single_masked_search_with_rank(
            query_info["search_query"], 
            query_info["target_document_id"]
        )
        
        print(f"Query: '{result['search_query']}'")
        print(f"Keywords matched: {result['keywords_matched']}")
        print(f"Target rank: {result['masked_target_rank']}")
        print("-" * 30)
