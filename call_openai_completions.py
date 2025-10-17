#!/usr/bin/env python3
"""
Call OpenAI completions API directly with the same prompt format as call_completions_api.py
This allows comparing finetuned model vs unfinetuned OpenAI models.
"""
import json
import sys
import os
import openai

def create_prompt(question: str) -> str:
    """Create the exact same prompt format as call_completions_api.py"""
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
<system>

<user>Question: {question}</user>

OUTPUT:
"""

def main():
    if len(sys.argv) != 2:
        print("Usage: python call_openai_completions.py <question>")
        print("Example: python call_openai_completions.py 'How do I fix CSS alignment issues?'")
        sys.exit(1)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    question = sys.argv[1]
    prompt = create_prompt(question)

    print(f"Question: {question}")
    print("Calling OpenAI API...\n")

    try:
        # Initialize OpenAI client
        client = openai.OpenAI()
        
        # Make API request using the legacy completions endpoint
        # Note: Using gpt-3.5-turbo-instruct as it supports the completions format
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",  # Best model that supports completions API
            prompt=prompt,
            max_tokens=150,
            temperature=0.3,
            stop=["<|end_of_text|>"]
        )

        # Extract the result
        generated_text = response.choices[0].text.strip()
        
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
            
    except openai.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()