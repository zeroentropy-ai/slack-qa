"""
Slack Search API Wrapper
========================
Simple wrapper around Slack's search API for ML dataset generation.

Usage:
    from slack_search import SlackSearch
    
    search = SlackSearch(token="xoxe.xoxp-...")
    
    # Search everything
    results = search.search("machine learning", search_type="all")
    
    # Search only messages
    messages = search.search("error logs", search_type="messages")
    
    # Search with filters
    results = search.search("in:#general from:@john bug", search_type="messages")

Required scope: search:read
"""

import requests
import time
from typing import Dict, List, Optional, Literal, Iterator
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Wrapper for search results with consistent interface"""
    query: str
    search_type: str
    total: int
    matches: List[Dict]
    messages: List[Dict] = None
    files: List[Dict] = None
    raw_response: Dict = None


class SlackSearch:
    """
    Simple wrapper around Slack's search API.
    
    Args:
        token: Slack user token with search:read scope
        rate_limit_delay: Delay between API calls in seconds (default: 3s for Tier 2)
    """
    
    def __init__(self, token: str, rate_limit_delay: float = 3.0):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
        self.base_url = "https://slack.com/api"
        self.rate_limit_delay = rate_limit_delay
        self._last_call_time = 0
    
    def _rate_limit(self):
        """Simple rate limiting"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_call_time = time.time()
    
    def _api_call(self, endpoint: str, params: Dict) -> Dict:
        """Make API call with rate limiting"""
        self._rate_limit()
        
        response = requests.get(
            f"{self.base_url}/{endpoint}",
            headers=self.headers,
            params=params
        )
        
        data = response.json()
        
        if not data.get('ok'):
            error = data.get('error', 'unknown_error')
            raise Exception(f"Slack API error: {error}")
        
        return data
    
    def search(
        self,
        query: str,
        search_type: Literal["all", "messages", "files"] = "all",
        count: int = 100,
        page: int = 1,
        sort: Literal["score", "timestamp"] = "score",
        sort_dir: Literal["asc", "desc"] = "desc",
        highlight: bool = False
    ) -> SearchResult:
        """
        Search Slack workspace.
        
        Args:
            query: Search query (supports Slack search syntax like 'in:#channel from:@user')
            search_type: Type of search - "all", "messages", or "files"
            count: Number of results per page (max 100)
            page: Page number (max 100)
            sort: Sort by "score" or "timestamp"
            sort_dir: Sort direction "asc" or "desc"
            highlight: Enable highlighting markers in results
            
        Returns:
            SearchResult object with matches
            
        Examples:
            # Basic search
            results = search.search("error")
            
            # Search in specific channel
            results = search.search("in:#engineering bug")
            
            # Search from specific user
            results = search.search("from:@john deployment")
            
            # Search with date range
            results = search.search("error after:2024-01-01 before:2024-01-31")
            
            # Search only messages
            results = search.search("meeting notes", search_type="messages")
        """
        
        params = {
            'query': query,
            'count': min(count, 100),
            'page': min(page, 100),
            'sort': sort,
            'sort_dir': sort_dir,
            'highlight': highlight
        }
        
        # Call appropriate endpoint
        endpoint_map = {
            'all': 'search.all',
            'messages': 'search.messages',
            'files': 'search.files'
        }
        
        endpoint = endpoint_map.get(search_type, 'search.all')
        data = self._api_call(endpoint, params)
        
        # Extract results based on search type
        result = SearchResult(
            query=query,
            search_type=search_type,
            total=0,
            matches=[],
            raw_response=data
        )
        
        if search_type == "all":
            result.messages = data.get('messages', {}).get('matches', [])
            result.files = data.get('files', {}).get('matches', [])
            result.total = (
                data.get('messages', {}).get('total', 0) + 
                data.get('files', {}).get('total', 0)
            )
            result.matches = result.messages + result.files
            
        elif search_type == "messages":
            result.messages = data.get('messages', {}).get('matches', [])
            result.total = data.get('messages', {}).get('total', 0)
            result.matches = result.messages
            
        elif search_type == "files":
            result.files = data.get('files', {}).get('matches', [])
            result.total = data.get('files', {}).get('total', 0)
            result.matches = result.files
        
        return result
    
    def search_all_pages(
        self,
        query: str,
        search_type: Literal["all", "messages", "files"] = "all",
        max_results: Optional[int] = None,
        **kwargs
    ) -> Iterator[Dict]:
        """
        Search and automatically paginate through all results.
        
        Args:
            query: Search query
            search_type: Type of search
            max_results: Maximum total results to return (None for all)
            **kwargs: Additional arguments passed to search()
            
        Yields:
            Individual result dictionaries (messages or files)
            
        Example:
            for result in search.search_all_pages("error", search_type="messages"):
                print(result['text'])
        """
        
        page = 1
        total_yielded = 0
        
        while True:
            result = self.search(query, search_type=search_type, page=page, **kwargs)
            
            if not result.matches:
                break
            
            for match in result.matches:
                yield match
                total_yielded += 1
                
                if max_results and total_yielded >= max_results:
                    return
            
            # Check if we've reached the last page
            if search_type == "messages":
                pagination = result.raw_response.get('messages', {}).get('pagination', {})
            elif search_type == "files":
                pagination = result.raw_response.get('files', {}).get('pagination', {})
            else:
                # For 'all', check messages pagination (could also check files)
                pagination = result.raw_response.get('messages', {}).get('pagination', {})
            
            if pagination.get('page') >= pagination.get('page_count', 1):
                break
            
            page += 1
            
            # Safety check
            if page > 100:
                print("Warning: Reached max page limit (100)")
                break


# Convenience functions for quick access
def search_messages(token: str, query: str, **kwargs) -> SearchResult:
    """Quick function to search messages"""
    searcher = SlackSearch(token)
    return searcher.search(query, search_type="messages", **kwargs)


def search_files(token: str, query: str, **kwargs) -> SearchResult:
    """Quick function to search files"""
    searcher = SlackSearch(token)
    return searcher.search(query, search_type="files", **kwargs)


def search_all(token: str, query: str, **kwargs) -> SearchResult:
    """Quick function to search everything"""
    searcher = SlackSearch(token)
    return searcher.search(query, search_type="all", **kwargs)
