#!/usr/bin/env python3
import json
import sys
import requests

def create_prompt(question: str) -> str:
    """Create the proper prompt format matching training template"""
    return f"""<system>
For the given user query output a JSON list of keyword-based search queries that
can be sent to lucene to obtain the documents that answer the query.
</system>

<question>
{question}
</question>

OUTPUT:
"""

if len(sys.argv) != 2:
    print("Usage: python call_completions_api.py <question>")
    sys.exit(1)

question = sys.argv[1]
prompt = create_prompt(question)

print(f"Question: {question}")
print("Generating search queries...\n")

# Make API request
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.3,
        "stop": ["<|end_of_text|>"]
    }
)

# Print the result
if response.status_code == 200:
    result = response.json()
    generated_text = result["choices"][0]["text"].strip()
    
    print("Generated search queries:")
    print(generated_text)
    
    # Try to parse as JSON to validate format
    try:
        # Extract just the JSON part if there's extra text
        if generated_text.startswith('[') and ']' in generated_text:
            json_end = generated_text.find(']') + 1
            json_part = generated_text[:json_end]
            queries = json.loads(json_part)
            print(f"\nParsed {len(queries)} search queries:")
            for i, query in enumerate(queries, 1):
                print(f"  {i}. {query}")
        else:
            print("\nNote: Response doesn't appear to be valid JSON format")
    except json.JSONDecodeError:
        print("\nNote: Could not parse response as JSON")
else:
    print(f"Error: {response.status_code} - {response.text}")
