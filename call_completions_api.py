#!/usr/bin/env python3
"""
Call localhost:8000/v1/completions with few-shot prompt for Slack query generation
"""

import json
import sys
import requests
from typing import List, Optional


def load_prompt_template(prompt_file: str = "few_shot_prompt.txt") -> str:
    """Load the few-shot prompt template"""
    try:
        with open(prompt_file, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: {prompt_file} not found")
        sys.exit(1)


def generate_search_queries(
    user_query: str,
    prompt_template: str,
    api_url: str = "http://localhost:8000/v1/completions",
    max_tokens: int = 100,
    temperature: float = 0.0,
    model: str = "default"
) -> Optional[List[str]]:
    """Generate search queries using the completions API"""
    
    # Build the complete prompt
    # Remove any trailing <user> tag and add the new query
    if prompt_template.endswith("<user>"):
        prompt = prompt_template + user_query + "</user>\n<assistant>"
    else:
        # Handle the case where there might be extra content
        prompt = prompt_template.rstrip() + f"\n\n<user>{user_query}</user>\n<assistant>"
    
    # Prepare the request
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["</assistant>", "<user>"],
        "model": model
    }
    
    try:
        # Make the API request
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        completion_text = result["choices"][0]["text"].strip()
        
        # Parse the JSON array from the completion
        try:
            search_queries = json.loads(completion_text)
            if isinstance(search_queries, list):
                return search_queries
            else:
                print(f"Error: Expected JSON array but got {type(search_queries)}")
                return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw completion: {completion_text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing API response: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'No response'}")
        return None


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python call_completions_api.py <query> [api_url] [prompt_file]")
        print("\nExample:")
        print('  python call_completions_api.py "how do i fix modal container shutdowns"')
        sys.exit(1)
    
    user_query = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/v1/completions"
    prompt_file = sys.argv[3] if len(sys.argv) > 3 else "few_shot_prompt.txt"
    
    # Load the prompt template
    print(f"Loading prompt template from: {prompt_file}")
    prompt_template = load_prompt_template(prompt_file)
    
    # Generate search queries
    print(f"\nGenerating search queries for: {user_query}")
    print(f"Calling API at: {api_url}")
    
    search_queries = generate_search_queries(
        user_query,
        prompt_template,
        api_url=api_url,
        temperature=0.0,  # Use deterministic generation
        max_tokens=100
    )
    
    if search_queries:
        print(f"\nGenerated {len(search_queries)} search queries:")
        for i, query in enumerate(search_queries, 1):
            print(f"  {i}. {query}")
    else:
        print("\nFailed to generate search queries")
        sys.exit(1)


def batch_process(
    queries_file: str,
    output_file: str,
    prompt_file: str = "few_shot_prompt.txt",
    api_url: str = "http://localhost:8000/v1/completions"
):
    """Process multiple queries from a file"""
    
    # Load prompt template
    prompt_template = load_prompt_template(prompt_file)
    
    # Load queries
    try:
        with open(queries_file, 'r') as f:
            queries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading queries file: {e}")
        return
    
    results = []
    total = len(queries)
    
    print(f"Processing {total} queries...")
    
    for i, query_data in enumerate(queries):
        query_id = query_data.get("id", f"query_{i}")
        user_query = query_data.get("query", "")
        
        print(f"\n[{i+1}/{total}] Processing: {user_query[:80]}...")
        
        search_queries = generate_search_queries(
            user_query,
            prompt_template,
            api_url=api_url
        )
        
        if search_queries:
            print(f"  ✓ Generated {len(search_queries)} queries")
            results.append({
                "query_id": query_id,
                "query": user_query,
                "generated_searches": search_queries
            })
        else:
            print(f"  ✗ Failed to generate queries")
            results.append({
                "query_id": query_id,
                "query": user_query,
                "generated_searches": [],
                "error": "Failed to generate"
            })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    print(f"Successfully processed: {sum(1 for r in results if r.get('generated_searches'))}/{total}")


if __name__ == "__main__":
    # Check if batch mode
    if len(sys.argv) >= 2 and sys.argv[1] == "--batch":
        if len(sys.argv) < 4:
            print("Usage: python call_completions_api.py --batch <queries_file> <output_file> [api_url] [prompt_file]")
            sys.exit(1)
        
        queries_file = sys.argv[2]
        output_file = sys.argv[3]
        api_url = sys.argv[4] if len(sys.argv) > 4 else "http://localhost:8000/v1/completions"
        prompt_file = sys.argv[5] if len(sys.argv) > 5 else "few_shot_prompt.txt"
        
        batch_process(queries_file, output_file, prompt_file, api_url)
    else:
        main()