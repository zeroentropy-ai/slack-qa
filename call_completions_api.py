#!/usr/bin/env python3
import json
import sys
import requests

if len(sys.argv) != 2:
    print("Usage: python call_completions_api.py <query>")
    sys.exit(1)

# Load prompt template
with open("few_shot_prompt.txt", 'r') as f:
    prompt = f.read()

# Add the user query
prompt = prompt.rstrip() + f"\n\n<user>{sys.argv[1]}</user>\n<assistant>"

# Make API request
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.0,
        "stop": ["</assistant>", "<user>"],
        "model": "Qwen3/Qwen3-8B"
    }
)

# Print the result
result = response.json()
print(result["choices"][0]["text"].strip())