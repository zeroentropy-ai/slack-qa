"""
Slack Search API - Usage Examples
==================================
Examples for using slack_search.py in ML dataset generation pipelines.
"""

from slack_search import SlackSearch, search_messages

# Replace with your actual token
TOKEN = "xoxp-7878678554402-9489359346982-9626258575062-38d6e8a9ad40801d66a86792dc450868"


# Example 1: Simple message search
def example_simple_search():
    search = SlackSearch(token=TOKEN)
    results = search.search("machine learning", search_type="messages")
    
    for message in results.matches:
        print(message['text'])
        print(message['user'])
        print(message['channel']['name'])


# Example 2: Advanced query with filters
def example_advanced_search():
    search = SlackSearch(token=TOKEN)
    results = search.search(
        query="error in:#engineering from:@john after:2024-01-01",
        search_type="messages",
        sort="timestamp",
        sort_dir="desc"
    )
    
    print(f"Found {results.total} messages")
    for message in results.matches:
        print(f"{message['ts']}: {message['text']}")


# Example 3: Get all results (auto-pagination)
def example_pagination():
    search = SlackSearch(token=TOKEN)
    dataset = []
    
    for message in search.search_all_pages("deployment", max_results=1000):
        # Process each message for ML dataset
        dataset.append({
            'text': message['text'],
            'timestamp': message['ts'],
            'user': message['user']
        })
    
    print(f"Collected {len(dataset)} messages")
    return dataset


# Example 4: Quick convenience functions
def example_convenience():
    results = search_messages(token=TOKEN, query="bug report")
    print(f"Found {results.total} bug reports")


# Example 5: Search files
def example_file_search():
    search = SlackSearch(token=TOKEN)
    results = search.search("diagram", search_type="files", count=10)
    
    print(f"Found {results.total} files")
    for file in results.files:
        print(f"  - {file.get('name')} ({file.get('filetype')})")


# Example 6: Build training dataset
def build_ml_dataset(queries: list):
    """Example of building an ML dataset from multiple queries"""
    search = SlackSearch(token=TOKEN)
    dataset = []
    
    for query in queries:
        print(f"Searching for: {query}")
        for message in search.search_all_pages(query, max_results=500, search_type="messages"):
            dataset.append({
                'query': query,
                'text': message.get('text', ''),
                'user': message.get('user', ''),
                'channel': message.get('channel', {}).get('name', ''),
                'timestamp': message.get('ts', ''),
                'permalink': message.get('permalink', '')
            })
    
    return dataset

def boolean_example():
    search = SlackSearch(token=TOKEN)
    query = "key XOR Amazon"
    results = search.search(query=query, search_type="messages")
    
    print(len(results.matches))
    
    query = "key"
    results = search.search(query=query, search_type="messages")
    
    print(len(results.matches))

    query = "Amazon"
    results = search.search(query=query, search_type="messages")
    
    print(len(results.matches))


if __name__ == "__main__":
    # Run examples
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
    
    """print("\n=== Example 6: Build ML Dataset ===")
    queries = ["bug", "feature request", "error"]
    dataset = build_ml_dataset(queries)
    print(f"Total dataset size: {len(dataset)} messages")"""

    print("\n=== Example 7: Boolean Logic ===")
    boolean_example()