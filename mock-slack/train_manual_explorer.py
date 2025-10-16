#!/usr/bin/env python3
"""
Simple manual search explorer for training data
"""
import json
import random
import requests
import os
from collections import defaultdict
from masked_solr_library import masked_solr_search
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

SOLR_BASE_URL = "http://localhost:8983/solr"
COLLECTION_NAME = "train_data"

def load_random_document():
    """Load a random document from training collection"""
    response = requests.get(f"{SOLR_BASE_URL}/{COLLECTION_NAME}/select", params={
        "q": "*:*",
        "rows": 1,
        "start": random.randint(0, 100000),
        "fl": "*"
    })
    
    if response.status_code == 200:
        result = response.json()
        docs = result["response"]["docs"]
        return docs[0] if docs else {}
    return {}

def generate_questions(doc):
    """Generate 5-10 suggested questions using ChatGPT"""
    content = doc.get('content', '')
    title = doc.get('title', '')
    
    prompt = f"""Given this document, generate 5-10 natural questions that a user might ask an AI assistant that could be answered by this document. Make the questions varied and realistic.

Document title: {title}
Document content: {content[:1000]}

Generate questions as a JSON list of strings."""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        questions_text = response.choices[0].message.content.strip()
        # Try to extract JSON
        start = questions_text.find('[')
        end = questions_text.rfind(']') + 1
        if start >= 0 and end > start:
            questions_json = questions_text[start:end]
            questions = json.loads(questions_json)
            return questions if isinstance(questions, list) else []
    except Exception as e:
        print(f"Error generating questions: {e}")
    
    return []

def show_document(doc):
    """Display document"""
    print(f"ID: {doc.get('id')}")
    print(f"Dataset: {doc.get('dataset')}")
    print(f"Title: {doc.get('title', 'No title')}")
    print("-" * 40)
    content = doc.get('content', '')
    if len(content) > 500:
        print(f"{content[:500]}...")
    else:
        print(content)
    print("-" * 40)
    
    # Show suggested questions
    print("Suggested questions:")
    questions = generate_questions(doc)
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    print("-" * 40)

def rrf(rankings, k=60):
    """Reciprocal Rank Fusion"""
    scores = defaultdict(float)
    for ranking in rankings:
        for i, doc_id in enumerate(ranking):
            scores[doc_id] += 1 / (k + i + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

def test_queries(queries, target_doc_id):
    """Test multiple queries and return ranks + RRF rank"""
    results = []
    rankings = []
    
    for query in queries:
        docs, keywords_matched = masked_solr_search(query, COLLECTION_NAME, 100)
        doc_ids = [doc['id'] for doc in docs]
        rankings.append(doc_ids)
        
        # Find target rank
        rank = None
        for i, doc_id in enumerate(doc_ids, 1):
            if doc_id == target_doc_id:
                rank = i
                break
        
        results.append({
            "query": query,
            "rank": rank,
            "keywords_matched": keywords_matched
        })
    
    # Calculate RRF rank
    if rankings:
        fused_ids = rrf(rankings)
        rrf_rank = None
        for i, doc_id in enumerate(fused_ids, 1):
            if doc_id == target_doc_id:
                rrf_rank = i
                break
    else:
        rrf_rank = None
    
    return results, rrf_rank

def main():
    print("Training Manual Explorer")
    
    while True:
        # Get random document
        doc = load_random_document()
        if not doc:
            print("Failed to load document")
            continue
        
        show_document(doc)
        
        cmd = input("Commands: 's' skip, 'q' quit, or press Enter to continue: ").strip()
        if cmd == 'q':
            break
        elif cmd == 's':
            continue
        
        # Get user question
        question = input("Enter your question: ").strip()
        if not question:
            continue
        
        # Get search queries
        print("Enter search queries as JSON list:")
        queries_input = input("> ").strip()
        try:
            queries = json.loads(queries_input)
            if not isinstance(queries, list):
                print("Must be a list")
                continue
        except:
            print("Invalid JSON")
            continue
        
        # Test queries
        results, rrf_rank = test_queries(queries, doc['id'])
        
        # Show results
        print("\nResults:")
        for r in results:
            print(f"  '{r['query']}' -> rank {r['rank']} (k={r['keywords_matched']})")
        print(f"RRF rank: {rrf_rank}")
        
        # Save option
        save = input("Save? (y/n): ").strip().lower()
        if save == 'y':
            entry = {
                "question": question,
                "document_id": doc['id'],
                "search_queries": queries,
                "individual_ranks": [r['rank'] for r in results],
                "rrf_rank": rrf_rank,
                "document_content": doc.get('content', ''),
                "dataset": doc.get('dataset', '')
            }
            
            with open("manual_training_data.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            print("Saved")

if __name__ == "__main__":
    main()