"""
Slack Search API - Usage Examples
==================================
Examples for using slack_search.py in ML dataset generation pipelines.
"""

from slack_search import SlackSearch, search_messages

# Replace with your actual tokens and workspace info
# Browser mode (recommended)
XOXC_TOKEN = "xoxc-..."
FULL_COOKIES = "utm=%7B%7D; x=..."
WORKSPACE_URL = "https://yourworkspace.slack.com"

# Legacy app mode (if you have it)
XOXP_TOKEN = "xoxp-..."


# Example 1: Simple message search (browser mode)
def example_simple_search():
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    results = search.search("machine learning", search_type="messages")
    
    print(f"Found {len(results.matches)} messages")
    for message in results.matches[:5]:  # Show first 5
        print(f"- {message.get('text', '')[:100]}")
        print(f"  User: {message.get('username', 'N/A')}")
        print(f"  Channel: {message.get('channel', {}).get('name', 'N/A')}")
        print()


# Example 2: Advanced query with filters
def example_advanced_search():
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    
    results = search.search(
        query="error in:#engineering from:@john after:2024-01-01",
        search_type="messages",
        sort="timestamp",
        sort_dir="desc"
    )
    
    print(f"Found {results.total} messages")
    for message in results.matches[:5]:
        ts = message.get('ts', 'N/A')
        text = message.get('text', '')[:80]
        print(f"{ts}: {text}")


# Example 3: Get all results (auto-pagination)
def example_pagination():
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    
    dataset = []
    
    for message in search.search_all_pages("deployment", max_results=100, search_type="messages"):
        # Process each message for ML dataset
        dataset.append({
            'text': message.get('text', ''),
            'timestamp': message.get('ts', ''),
            'user': message.get('username', ''),
            'channel': message.get('channel', {}).get('name', '')
        })
    
    print(f"Collected {len(dataset)} messages")
    return dataset


# Example 4: Quick convenience functions
def example_convenience():
    results = search_messages(
        query="bug report",
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    print(f"Found {results.total} bug reports")


# Example 5: Search files
def example_file_search():
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    
    results = search.search("diagram", search_type="files", count=10)
    
    print(f"Found {results.total} files")
    for file in results.files[:10]:
        name = file.get('name', 'N/A')
        filetype = file.get('filetype', 'N/A')
        print(f"  - {name} ({filetype})")


# Example 6: Build training dataset
def build_ml_dataset(queries: list):
    """Example of building an ML dataset from multiple queries"""
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    
    dataset = []
    
    for query in queries:
        print(f"Searching for: {query}")
        for message in search.search_all_pages(query, max_results=500, search_type="messages"):
            dataset.append({
                'query': query,
                'text': message.get('text', ''),
                'user': message.get('username', ''),
                'channel': message.get('channel', {}).get('name', ''),
                'timestamp': message.get('ts', ''),
                'permalink': message.get('permalink', '')
            })
    
    return dataset


# Example 7: Boolean search logic
def boolean_example():
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    
    query = "key XOR Amazon"
    results = search.search(query=query, search_type="messages")
    print(f"'key XOR Amazon': {len(results.matches)} results")
    
    query = "key"
    results = search.search(query=query, search_type="messages")
    print(f"'key': {len(results.matches)} results")

    query = "Amazon"
    results = search.search(query=query, search_type="messages")
    print(f"'Amazon': {len(results.matches)} results")


# Example 8: Legacy app mode (if you still have app tokens)
def example_app_mode():
    """Example using legacy app token authentication"""
    search = SlackSearch(
        token=XOXP_TOKEN,
        auth_mode='app'
    )
    
    results = search.search("test", search_type="messages")
    print(f"App mode: Found {results.total} messages")


# Example 9: Search with different sort orders
def example_sort_options():
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    
    # Most relevant first
    results = search.search("error", sort="score", sort_dir="desc", search_type="messages")
    print(f"By relevance: {len(results.matches)} messages")
    
    # Most recent first
    results = search.search("error", sort="timestamp", sort_dir="desc", search_type="messages")
    print(f"By recency: {len(results.matches)} messages")
    
    # Oldest first
    results = search.search("error", sort="timestamp", sort_dir="asc", search_type="messages")
    print(f"Oldest first: {len(results.matches)} messages")


# Example 10: Channel-specific searches
def example_channel_searches():
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    
    # Search in specific channel
    results = search.search("in:#general announcement", search_type="messages")
    print(f"In #general: {len(results.matches)} announcements")
    
    # Search in multiple channels
    results = search.search("in:#engineering,#product bug", search_type="messages")
    print(f"In eng/product: {len(results.matches)} bug mentions")
    
    # Exclude certain channels
    results = search.search("-in:#random meeting", search_type="messages")
    print(f"Excluding #random: {len(results.matches)} meeting mentions")


# Example 11: Date range searches
def example_date_searches():
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    
    # Messages from last month
    results = search.search("after:2024-01-01 before:2024-02-01 deploy", search_type="messages")
    print(f"January deploys: {len(results.matches)}")
    
    # Messages from specific user in date range
    results = search.search("from:@john after:2024-01-01 milestone", search_type="messages")
    print(f"John's milestone mentions: {len(results.matches)}")


# Example 12: Export search results to JSON
def example_export_results():
    import json
    
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )
    
    results = search.search("machine learning", search_type="messages", count=50)
    
    # Export to JSON
    export_data = {
        'query': results.query,
        'total': results.total,
        'result_count': len(results.matches),
        'messages': [
            {
                'text': msg.get('text', ''),
                'user': msg.get('username', ''),
                'channel': msg.get('channel', {}).get('name', ''),
                'timestamp': msg.get('ts', ''),
                'permalink': msg.get('permalink', '')
            }
            for msg in results.matches
        ]
    }
    
    with open('search_results.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported {len(results.matches)} results to search_results.json")


if __name__ == "__main__":
    print("=== Slack Search API Examples ===\n")
    
    print("=== Example 1: Simple Search ===")
    example_simple_search()
    
    print("\n=== Example 2: Advanced Search ===")
    example_advanced_search()
    
    print("\n=== Example 3: Pagination ===")
    example_pagination()
    
    print("\n=== Example 4: Convenience Function ===")
    example_convenience()
    
    print("\n=== Example 5: File Search ===")
    example_file_search()
    
    print("\n=== Example 6: Build ML Dataset ===")
    queries = ["bug", "feature request", "error"]
    dataset = build_ml_dataset(queries)
    print(f"Total dataset size: {len(dataset)} messages")
    
    print("\n=== Example 7: Boolean Logic ===")
    boolean_example()
    
    print("\n=== Example 9: Sort Options ===")
    example_sort_options()
    
    print("\n=== Example 10: Channel Searches ===")
    example_channel_searches()
    
    print("\n=== Example 11: Date Searches ===")
    example_date_searches()
    
    print("\n=== Example 12: Export Results ===")
    example_export_results()