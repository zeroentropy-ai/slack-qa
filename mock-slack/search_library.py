#!/usr/bin/env python3
"""
Search library for mock Slack API using Solr
"""
import json
import requests
from typing import List, Dict, Any
from urllib.parse import quote

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

def build_solr_query(search_query: str) -> str:
    """
    Build Solr query from search string using OR for multiple words, removing stop words
    """
    words = search_query.strip().split()
    # Remove stop words
    words = remove_stop_words(words)
    
    if not words:  # If all words were stop words
        return ""  # Return empty query
    
    # Escape special characters in words
    escaped_words = [escape_solr_special_chars(word) for word in words]
    
    if len(escaped_words) == 1:
        return f"content:{escaped_words[0]}"
    else:
        return " OR ".join([f"content:{word}" for word in escaped_words])

def batch_search_with_ranks(queries: List[Dict[str, str]], collection: str = "slack", rows: int = 100) -> List[Dict[str, Any]]:
    """
    Perform batch search and return rank of target document for each query
    
    Args:
        queries: List of dicts with "search_query" and "target_document_id" keys
        collection: Solr collection name (default "slack")
        rows: Number of results to fetch (default 100 to capture most ranks)
    
    Returns:
        List of dicts with "search_query", "target_document_id", and "target_rank" keys
        target_rank is null if document not found in results, otherwise 1-based rank
    """
    results = []
    
    for query_info in queries:
        search_query = query_info["search_query"]
        target_doc_id = query_info["target_document_id"]
        
        # Build Solr query
        solr_query = build_solr_query(search_query)
        
        # If query is empty (all stop words), return no results
        if not solr_query:
            result = query_info.copy()
            result["target_rank"] = None
            results.append(result)
            continue
        
        # Make request to Solr
        params = {
            'q': solr_query,
            'rows': rows,
            'fl': 'id',
            'sort': 'score desc'
        }
        
        try:
            response = requests.get(f"{SOLR_BASE_URL}/{collection}/select", params=params)
            response.raise_for_status()
            
            solr_result = response.json()
            docs = solr_result['response']['docs']
            
            # Find rank of target document
            target_rank = None
            for i, doc in enumerate(docs, 1):
                if doc['id'] == target_doc_id:
                    target_rank = i
                    break
            
            # Copy input object and add target_rank
            result = query_info.copy()
            result["target_rank"] = target_rank
            results.append(result)
            
        except Exception as e:
            # Copy input object and add target_rank
            result = query_info.copy()
            result["target_rank"] = None
            results.append(result)
    
    return results

def single_search_with_rank(search_query: str, target_document_id: str, collection: str = "slack", rows: int = 100) -> Dict[str, Any]:
    """
    Perform single search and return rank of target document
    
    Args:
        search_query: The search query string
        target_document_id: ID of document to find rank for
        collection: Solr collection name (default "slack")
        rows: Number of results to fetch
    
    Returns:
        Dict with "search_query", "target_document_id", and "target_rank" keys
    """
    return batch_search_with_ranks([{
        "search_query": search_query,
        "target_document_id": target_document_id
    }], collection, rows)[0]