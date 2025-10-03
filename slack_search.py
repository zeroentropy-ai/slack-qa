"""
Slack Search API Wrapper
========================
Simple wrapper around Slack's search API for ML dataset generation.

Usage:
    from slack_search import SlackSearch
    
    # Browser mode (recommended)
    search = SlackSearch(
        token="xoxc-...",
        auth_mode='browser',
        cookies="full cookie string from browser",
        workspace_url="https://yourworkspace.slack.com"
    )
    
    # App mode (legacy)
    search = SlackSearch(token="xoxp-...", auth_mode='app')
    
    # Search everything
    results = search.search("machine learning", search_type="all")
    
    # Search only messages
    messages = search.search("error logs", search_type="messages")
    
    # Search with filters
    results = search.search("in:#general from:@john bug", search_type="messages")

Required scope (app mode): search:read
Browser mode: Requires xoxc token and full cookies from browser session
"""

import requests
import time
from typing import Dict, List, Optional, Literal, Iterator
from dataclasses import dataclass
from requests_toolbelt.multipart.encoder import MultipartEncoder


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
        token: Either xoxp (app) token or xoxc (browser) token
        auth_mode: 'app' for OAuth app tokens, 'browser' for session tokens (default: 'browser')
        cookies: Full cookie string from browser (required for browser mode)
        workspace_url: Workspace URL like 'https://workspace.slack.com' (required for browser mode)
        rate_limit_delay: Delay between API calls in seconds (default: 3s for Tier 2)
    """
    
    def __init__(
        self,
        token: str,
        auth_mode: Literal['app', 'browser'] = 'browser',
        cookies: Optional[str] = None,
        workspace_url: Optional[str] = None,
        rate_limit_delay: float = 3.0
    ):
        self.token = token
        self.auth_mode = auth_mode
        self.cookies = cookies
        self.rate_limit_delay = rate_limit_delay
        self._last_call_time = 0
        
        if auth_mode == 'app':
            self.headers = {"Authorization": f"Bearer {token}"}
            self.base_url = "https://slack.com/api"
        elif auth_mode == 'browser':
            if not cookies or not workspace_url:
                raise ValueError("Browser mode requires both cookies and workspace_url")
            self.base_url = f"{workspace_url}/api"
            # Extract xoxd token from cookies if present
            self.xoxd_token = self._extract_xoxd_from_cookies(cookies)
        else:
            raise ValueError("auth_mode must be 'app' or 'browser'")
    
    def _extract_xoxd_from_cookies(self, cookies: str) -> Optional[str]:
        """Extract xoxd token from cookie string"""
        for cookie in cookies.split('; '):
            if cookie.startswith('d='):
                # URL decode the xoxd token
                import urllib.parse
                return urllib.parse.unquote(cookie[2:])
        return None
    
    def _rate_limit(self):
        """Simple rate limiting"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_call_time = time.time()
    
    def _api_call(self, endpoint: str, params: Dict) -> Dict:
        """Make API call with rate limiting and appropriate authentication"""
        self._rate_limit()
        
        if self.auth_mode == 'app':
            return self._api_call_app(endpoint, params)
        else:
            return self._api_call_browser(endpoint, params)
    
    def _api_call_app(self, endpoint: str, params: Dict) -> Dict:
        """Make API call with OAuth app token"""
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
    
    def _api_call_browser(self, endpoint: str, params: Dict) -> Dict:
        """Make API call with browser session tokens"""
        # Prepare multipart form data
        fields = {'token': self.token}
        if params:
            fields.update(params)
        
        multipart_data = MultipartEncoder(fields=fields)
        
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": self.cookies,
            "Content-Type": multipart_data.content_type,
            "Origin": "https://app.slack.com",
            "Referer": "https://app.slack.com/",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        }
        
        # Query parameters
        query_params = {
            "_x_id": f"search-{int(time.time() * 1000)}",
            "_x_version_ts": str(int(time.time())),
            "_x_frontend_build_type": "current",
            "_x_desktop_ia": "4",
            "_x_gantry": "true",
            "fp": "a2"
        }
        
        response = requests.post(
            f"{self.base_url}/{endpoint}",
            headers=headers,
            params=query_params,
            data=multipart_data
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
            'count': str(min(count, 100)),
            'page': str(min(page, 100)),
            'sort': sort,
            'sort_dir': sort_dir,
            'highlight': str(highlight).lower()
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
def search_messages(
    query: str,
    token: str = None,
    auth_mode: str = 'browser',
    cookies: str = None,
    workspace_url: str = None,
    **kwargs
) -> SearchResult:
    """Quick function to search messages"""
    searcher = SlackSearch(
        token=token,
        auth_mode=auth_mode,
        cookies=cookies,
        workspace_url=workspace_url
    )
    return searcher.search(query, search_type="messages", **kwargs)


def search_files(
    query: str,
    token: str = None,
    auth_mode: str = 'browser',
    cookies: str = None,
    workspace_url: str = None,
    **kwargs
) -> SearchResult:
    """Quick function to search files"""
    searcher = SlackSearch(
        token=token,
        auth_mode=auth_mode,
        cookies=cookies,
        workspace_url=workspace_url
    )
    return searcher.search(query, search_type="files", **kwargs)


def search_all(
    query: str,
    token: str = None,
    auth_mode: str = 'browser',
    cookies: str = None,
    workspace_url: str = None,
    **kwargs
) -> SearchResult:
    """Quick function to search everything"""
    searcher = SlackSearch(
        token=token,
        auth_mode=auth_mode,
        cookies=cookies,
        workspace_url=workspace_url
    )
    return searcher.search(query, search_type="all", **kwargs)


# Example usage
if __name__ == "__main__":
    # Browser mode example
    XOXC_TOKEN = "xoxc-..."
    FULL_COOKIES = "utm=%7B%7D; x=..."
    WORKSPACE_URL = "https://yourworkspace.slack.com"
    
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    
    # Search for messages
    results = search.search("machine learning", search_type="messages")
    print(f"Found {results.total} results")
    for msg in results.matches[:5]:
        print(f"- {msg.get('text', '')[:100]}")
    
    # Paginate through all results
    print("\nAll results:")
    for i, msg in enumerate(search.search_all_pages("error", search_type="messages", max_results=10)):
        print(f"{i+1}. {msg.get('text', '')[:80]}")