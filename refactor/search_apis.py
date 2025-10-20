#!/usr/bin/env python3
# Concrete implementations of SearchAPI

import asyncio
import tomllib
from pathlib import Path
import httpx
from .types import SearchAPI, Document, ErrorReport
from .utils import get_logger, ROOT

logger = get_logger()

def _load_slack_tokens() -> dict[str, dict[str, str]]:
    """Load Slack tokens from slack_tokens.toml"""
    tokens_path = Path(ROOT) / "refactor" / "slack_tokens.toml"
    try:
        with open(tokens_path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        logger.error(f"slack_tokens.toml not found at {tokens_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading slack_tokens.toml: {e}")
        return {}


class SlackSearch(SearchAPI):
    """Slack search API with token rotation and rate limiting"""

    def __init__(self, workspace_name: str = "modal_community"):
        super().__init__(source_name="slack")
        self.workspace_name = workspace_name
        self.tokens = _load_slack_tokens()
        self.current_token_index = 0
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0.0

        if workspace_name not in self.tokens:
            raise ValueError(f"Workspace '{workspace_name}' not found in slack_tokens.toml")

    def _get_current_token(self) -> dict[str, str]:
        """Get current token configuration"""
        return self.tokens[self.workspace_name]

    def _rotate_token(self) -> None:
        """Rotate to next available token (placeholder for future multi-token support)"""
        # For now, just log the rotation attempt
        logger.warning("Token rotation requested - implement multi-token support if needed")

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = asyncio.get_event_loop().time()

    async def run_search(self, query: str) -> list[Document] | ErrorReport:
        """Search Slack with a single query"""
        await self._rate_limit()

        token_config = self._get_current_token()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{token_config['workspace_url']}/api/search.messages",
                    headers={
                        "Authorization": f"Bearer {token_config['xoxc']}",
                        "Cookie": token_config["cookie"],
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data={
                        "token": token_config["xoxc"],
                        "query": query,
                        "count": 1000,  # Maximum allowed by Slack API
                        "sort": "score",
                        "sort_dir": "desc"
                    },
                    timeout=30.0
                )

                if response.status_code == 429:
                    # Rate limited - try rotating token
                    self._rotate_token()
                    return ErrorReport(message=f"Rate limited for query: {query}")

                if response.status_code != 200:
                    return ErrorReport(message=f"Slack API error {response.status_code}: {response.text}")

                data = response.json()

                if not data.get("ok", False):
                    error_msg = data.get("error", "Unknown error")
                    if error_msg == "invalid_auth":
                        self._rotate_token()
                    return ErrorReport(message=f"Slack API error: {error_msg}")

                # Convert Slack messages to Document format
                documents: list[Document] = []
                messages = data.get("messages", {}).get("matches", [])

                for msg in messages:
                    doc_id = f"slack_{msg.get('ts', '')}"
                    content = msg.get("text", "")
                    metadata = {
                        "user": msg.get("user", ""),
                        "channel": msg.get("channel", {}).get("name", ""),
                        "timestamp": msg.get("ts", ""),
                        "type": msg.get("type", "message"),
                        "source": "slack"
                    }

                    documents.append(Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata
                    ))

                return documents

        except Exception as e:
            return ErrorReport(message=f"Slack search error: {str(e)}")


class SolrSearch(SearchAPI):
    """Apache Solr search API"""

    def __init__(self, collection: str = "slack", base_url: str = "http://localhost:8983/solr"):
        super().__init__(source_name="solr")
        self.collection = collection
        self.base_url = base_url

    async def run_search(self, query: str) -> list[Document] | ErrorReport:
        """Search Solr with a single query"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/{self.collection}/select",
                    params={
                        "q": query,
                        "rows": 1000,  # High limit for comprehensive results
                        "wt": "json",
                        "fl": "id,content,*",
                        "sort": "score desc"
                    },
                    timeout=30.0
                )

                if response.status_code != 200:
                    return ErrorReport(message=f"Solr error {response.status_code}: {response.text}")

                data = response.json()
                docs = data.get("response", {}).get("docs", [])

                documents: list[Document] = []
                for doc in docs:
                    doc_id = doc.get("id", "")
                    content = doc.get("content", "")

                    # Extract metadata (all fields except id and content)
                    metadata = {k: v for k, v in doc.items() if k not in ["id", "content"]}
                    metadata["source"] = "solr"

                    documents.append(Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata
                    ))

                return documents

        except Exception as e:
            return ErrorReport(message=f"Solr search error: {str(e)}")


# Common English stop words that search engines typically filter out
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

# TODO replace this with a simpler bm25
class MaskedSolrSearch(SearchAPI):
    """Masked Solr search with progressive keyword matching"""

    def __init__(self, collection: str = "slack", base_url: str = "http://localhost:8983/solr"):
        super().__init__(source_name="masked_solr")
        self.collection = collection
        self.base_url = base_url

    def _remove_stop_words(self, words: list[str]) -> list[str]:
        """Remove stop words from a list of words"""
        return [word for word in words if word.lower() not in STOP_WORDS]

    def _escape_solr_chars(self, word: str) -> str:
        """Escape special Solr characters"""
        special_chars = set('+-&|!(){}[]^"~*?:\\/')
        if any(char in special_chars for char in word):
            escaped = word.replace('"', '\\"')
            return f'"{escaped}"'
        return word

    async def run_search(self, query: str) -> list[Document] | ErrorReport:
        """
        Search with progressive keyword matching:
        1. Try all keywords (AND)
        2. If no results, try N-1 keywords
        3. Continue until results found or single keyword
        """
        words = query.strip().split()
        keywords = self._remove_stop_words(words)

        if len(keywords) > 6:
            return []

        if len(keywords) == 0:
            return ErrorReport(message="No valid keywords after stop word removal")

        # Try progressively smaller keyword sets
        for num_keywords in range(len(keywords), 0, -1):
            escaped_keywords = [self._escape_solr_chars(kw) for kw in keywords[:num_keywords]]
            solr_query = " AND ".join(escaped_keywords)

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/{self.collection}/select",
                        params={
                            "q": solr_query,
                            "rows": 1000,  # High limit for comprehensive results
                            "wt": "json",
                            "fl": "id,content,*",
                            "sort": "score desc"
                        },
                        timeout=30.0
                    )

                    if response.status_code != 200:
                        continue  # Try with fewer keywords

                    data = response.json()
                    docs = data.get("response", {}).get("docs", [])

                    if len(docs) > 0:
                        # Found results, convert to Document format
                        documents: list[Document] = []
                        for doc in docs:
                            doc_id = doc.get("id", "")
                            content = doc.get("content", "")

                            metadata = {k: v for k, v in doc.items() if k not in ["id", "content"]}
                            metadata["source"] = "masked_solr"
                            metadata["keywords_matched"] = num_keywords

                            documents.append(Document(
                                id=doc_id,
                                content=content,
                                metadata=metadata
                            ))

                        return documents

            except Exception as e:
                logger.error(f"Error in masked Solr search with {num_keywords} keywords: {e}")
                continue

        # No results found with any keyword combination
        return []
