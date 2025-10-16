#!/usr/bin/env python3
"""
Test script for masked Solr search with content display
"""
import requests
from masked_solr_library import masked_solr_search, build_combination_query

SOLR_URL = "http://localhost:8983/solr/slack"

def test_query_with_content(search_query: str, max_results: int = 5):
    """Test a search query and display document contents"""
    keywords = search_query.strip().split()
    print(f'Testing query: "{search_query}" ({len(keywords)} keywords)')
    print('=' * 60)
    
    # Try each level and show what we get
    for k in range(len(keywords), 0, -1):
        solr_query = build_combination_query(keywords, k)
        print(f'\nTrying {k} keywords: {solr_query[:100]}...')
        
        params = {
            'q': solr_query,
            'rows': max_results,
            'fl': 'id,content,score',
            'sort': 'score desc'
        }
        
        try:
            response = requests.get(f"{SOLR_URL}/select", params=params)
            response.raise_for_status()
            
            solr_result = response.json()
            docs = solr_result['response']['docs']
            
            if docs:
                print(f'Found {len(docs)} results with {k} keywords:')
                for i, doc in enumerate(docs, 1):
                    content = doc.get('content', [''])[0] if isinstance(doc.get('content', ''), list) else doc.get('content', '')
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    print(f'  {i}. Score: {doc.get("score", 0):.2f}')
                    print(f'     ID: {doc["id"]}')
                    print(f'     Content: {content_preview}')
                    print()
                return docs, k
            else:
                print(f'No results with {k} keywords')
        
        except Exception as e:
            print(f'Error with {k} keywords: {e}')
    
    print('No results found at any level')
    return [], 0

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "modal deploy gpu llm instance"
    
    test_query_with_content(query)