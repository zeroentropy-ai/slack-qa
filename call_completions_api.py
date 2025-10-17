#!/usr/bin/env python3
import json
import sys
import requests

def create_prompt(question: str) -> str:
    """Create the proper prompt format matching training template"""
    return f"""<system>
Generate search queries for the given question. Return a JSON array of 5-7
shortest possible keyword search queries that will help find the target document
in the top 20 results using diverse search terms.

Search Algorithm Context:
- Uses Lucene-style search: finds documents containing ALL keywords in each query
- If no matches found with N words, decrements to N-1 words and tries again
- Order doesn't matter - focus on selecting the right terms

Key Guidelines:
1. **Specificity vs Recall**: Use salient words, technical terms, proper nouns,
IDs and numbers from the question. Too many specific words may return no results.
2. **Query Length**: Try 1-3 keyword combinations. Short, non-specific queries may
surface too many results. Adding a specific 3rd term can improve ranking.
3. **Think like document author**: Use terminology that would appear IN the target
document, not just ABOUT the topic.
4. **Avoid filler words**: Skip "of", "is", "the", "in" etc. - focus on uncommon,
meaningful words.

Example:
Question: "How can the Deflection Gap concept help in evaluating knowledge

failures and improving knowledge quality, trust, and coverage?"
Output: ["deflection gap", "knowledge quality"]
Reasoning: "deflection gap" appears to be a specific concept (title case) unlikely
to appear in many documents. "knowledge quality" is the general concept being
searched for.

Your output MUST BE valid JSON array of strings.
<system>

<user>Question: How can the Deflection Gap concept help in evaluating knowledge failures and improving knowledge quality, trust, and coverage?</user>

OUTPUT:
["deflection gap", "knowledge quality"]

<user> Question: What does the `--write-context` flag do in run_butler.py's main function? </user>

OUTPUT:
["run_butler.py limited SQL operations", "run_butler.py limited SQL functionality", "run_butler.py limited SQL", "run_butler.py limited SQL main function", "run_butler.py SQL limited functionality"]

<user>Question: {question}</user>

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
