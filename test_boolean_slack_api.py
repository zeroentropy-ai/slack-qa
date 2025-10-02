"""
Slack Search Boolean Syntax Tester
===================================
Tests various boolean operators to see what Slack actually supports.
"""

from slack_search import SlackSearch
import time

TOKEN = "xoxp-7878678554402-9489359346982-9626258575062-38d6e8a9ad40801d66a86792dc450868"


# Test queries - modify these base terms to match content in your workspace
TERM_A = "hello"  # Common term that should have results
TERM_B = "index"    # Another common term
TERM_C = "document"    # Third term


def test_search(search, query, description):
    """Test a search query and print results"""
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"Query: '{query}'")
    print('-'*60)
    
    try:
        results = search.search(query, search_type="messages", count=5)
        print(f"✅ Success! Found {results.total} results")
        
        if results.matches:
            print(f"First result: {results.matches[0].get('text', '')[:100]}...")
        
        return True, results.total
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, 0


def run_battery():
    search = SlackSearch(token=TOKEN)
    results = []
    
    # Baseline tests
    tests = [
        # Basic searches
        (f"{TERM_A}", "Baseline: Single term"),
        (f"{TERM_B}", "Baseline: Second term"),
        (f"{TERM_A} {TERM_B}", "Two terms (implicit AND?)"),
        
        # NOT operator (we know - works)
        (f"{TERM_A} -{TERM_B}", "NOT with dash (known working)"),
        (f"-{TERM_B}", "NOT alone with dash"),
        
        # AND operator attempts
        (f"{TERM_A} AND {TERM_B}", "AND (uppercase)"),
        (f"{TERM_A} and {TERM_B}", "and (lowercase)"),
        (f"{TERM_A} && {TERM_B}", "AND with &&"),
        (f"{TERM_A} & {TERM_B}", "AND with &"),
        (f"{TERM_A} + {TERM_B}", "AND with +"),
        
        # OR operator attempts
        (f"{TERM_A} OR {TERM_B}", "OR (uppercase)"),
        (f"{TERM_A} or {TERM_B}", "or (lowercase)"),
        (f"{TERM_A} || {TERM_B}", "OR with ||"),
        (f"{TERM_A} | {TERM_B}", "OR with |"),
        
        # XOR attempts
        (f"{TERM_A} XOR {TERM_B}", "XOR (uppercase)"),
        (f"{TERM_A} xor {TERM_B}", "xor (lowercase)"),
        (f"{TERM_A} ^ {TERM_B}", "XOR with ^"),
        
        # NOT operator alternatives
        (f"{TERM_A} NOT {TERM_B}", "NOT (uppercase)"),
        (f"{TERM_A} not {TERM_B}", "not (lowercase)"),
        (f"{TERM_A} !{TERM_B}", "NOT with !"),
        
        # Grouping attempts
        (f"({TERM_A} {TERM_B})", "Parentheses grouping"),
        (f"({TERM_A} OR {TERM_B}) {TERM_C}", "Parentheses with OR"),
        (f'"{TERM_A} {TERM_B}"', "Exact phrase (quotes)"),
        
        # Wildcards
        (f"{TERM_A}*", "Wildcard with *"),
        (f"*{TERM_A}", "Wildcard prefix"),
        (f"{TERM_A}?", "Wildcard with ?"),
        
        # Special combinations
        (f"{TERM_A} -{TERM_B} -{TERM_C}", "Multiple NOT operators"),
        (f"{TERM_A} {TERM_B} {TERM_C}", "Three terms"),
        
        # Slack-specific operators (known to work)
        (f"in:#general {TERM_A}", "in: operator (channel)"),
        (f"from:@user {TERM_A}", "from: operator"),
        (f"has:link {TERM_A}", "has: operator (link)"),
        (f"has:star {TERM_A}", "has: operator (star)"),
        (f"has:pin {TERM_A}", "has: operator (pin)"),
        (f"has:reaction {TERM_A}", "has: operator (reaction)"),
        (f"before:2024-01-01 {TERM_A}", "before: date operator"),
        (f"after:2024-01-01 {TERM_A}", "after: date operator"),
        (f"during:january {TERM_A}", "during: operator"),
        (f"on:2024-01-01 {TERM_A}", "on: date operator"),
        
        # Case sensitivity
        (f"{TERM_A.upper()}", "UPPERCASE term"),
        (f"{TERM_A.lower()}", "lowercase term"),
        (f"{TERM_A.capitalize()}", "Capitalized term"),
    ]
    
    print("Starting Slack Search Boolean Syntax Battery Test")
    print(f"Using terms: '{TERM_A}', '{TERM_B}', '{TERM_C}'")
    print(f"Total tests: {len(tests)}")
    
    for query, description in tests:
        success, count = test_search(search, query, description)
        results.append({
            'query': query,
            'description': description,
            'success': success,
            'count': count
        })
        time.sleep(1)  # Be nice to the API
    
    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n✅ Successful queries: {len(successful)}/{len(results)}")
    print(f"❌ Failed queries: {len(failed)}/{len(results)}")
    
    if successful:
        print("\n" + "-"*60)
        print("Working Operators:")
        print("-"*60)
        for r in successful:
            if r['count'] > 0:
                print(f"  ✓ {r['description']}")
                print(f"    Query: '{r['query']}'")
                print(f"    Results: {r['count']}")
    
    if failed:
        print("\n" + "-"*60)
        print("Failed/Unsupported Operators:")
        print("-"*60)
        for r in failed:
            print(f"  ✗ {r['description']}: '{r['query']}'")
    
    return results


if __name__ == "__main__":
    # Update these terms to match content in your workspace
    print(f"\n⚠️  Make sure to update TERM_A, TERM_B, TERM_C to match content in your workspace!")
    print(f"Current terms: '{TERM_A}', '{TERM_B}', '{TERM_C}'")
    
    input("\nPress Enter to start the battery test...")
    
    results = run_battery()
    
    # Save results to file
    import json
    with open('slack_search_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n\n📄 Results saved to 'slack_search_test_results.json'")