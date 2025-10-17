#!/usr/bin/env python3
"""
Generate Slack search queries from training questions.
Takes training_data_step_0.json and generates multiple search queries for each question.
Outputs in the same format as step_0.json.
"""
import json
import os
import time
from typing import List, Dict, Any
from tqdm import tqdm
import openai

# Set up OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_training_data(file_path: str = "training_data_step_0.json") -> List[Dict[str, Any]]:
    """Load training questions from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_search_queries(question: str, document_content: str) -> List[str]:

    prompt = f"""Given a user question, generate 10-20 *diverse* Slack search queries that someone might use to find this information. The queries should not have repeated keywords beyond

The search queries have success with the following tactics:
- Look for the salient words and terms in the question and use them
- We need to get diverse queries. Don't make many queries share the same keywords beyond 2 keywords!
- Throw in extra queries which have the expansion of acronyms if used in the question, and acronyms of words given in the question.
- Short keyword phrases (1-4 words typically)
- What someone would actually type in Slack search
- Diverse approaches to finding the same information
- salient terms, error messages, ids, key concepts from the document

User Question: {question}

Target document for reference: {document_content}

Prepare the query *solely* by looking at the question and not the target document.

Generate search queries as a JSON list of strings. Focus on different ways someone might search for this information:"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=500
        )
        
        queries_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON
        start = queries_text.find('[')
        end = queries_text.rfind(']') + 1
        if start >= 0 and end > start:
            queries_json = queries_text[start:end]
            queries = json.loads(queries_json)
            
            # Clean and validate queries
            clean_queries = []
            for query in queries:
                if isinstance(query, str) and len(query.strip()) > 0:
                    # Clean the query
                    clean_query = query.strip().lower()
                    # Remove quotes if present
                    if clean_query.startswith('"') and clean_query.endswith('"'):
                        clean_query = clean_query[1:-1]
                    if clean_query != "":  # Reasonable length limit
                        clean_queries.append(clean_query)
            
            return clean_queries[:20]  # Limit to 20 queries
        
    except Exception as e:
        print(f"Error generating search queries: {e}")
    
    return []

def generate_search_training_data(
    input_file: str = "training_data_step_0.json",
    output_file: str = "search_queries_step_0.json",
    max_questions: int = 500
) -> List[Dict[str, Any]]:
    """
    Generate search query training data in step_0.json format.
    
    For each question, generates multiple search queries and creates entries like:
    {
        "search_query": "modal error",
        "target_document_id": "doc_123", 
        "query_id": "uuid-here",
        "original_user_query": "How do I fix Modal errors?"
    }
    """
    
    print(f"Loading training data from {input_file}...")
    training_data = load_training_data(input_file)
    
    print(f"Loaded {len(training_data):,} training questions")
    
    # Limit to first max_questions for faster iteration
    if len(training_data) > max_questions:
        training_data = training_data[:max_questions]
        print(f"Processing first {max_questions} questions for fast iteration")
    
    search_training_data = []
    
    for entry in tqdm(training_data, desc="Generating search queries"):
        question = entry['question']
        document_id = entry['document_id']
        query_id = entry['query_id']
        content = entry['content']
        
        # Generate search queries for this question
        search_queries = generate_search_queries(question, content)
        
        # Create entries in step_0 format
        for search_query in search_queries:
            search_entry = {
                "search_query": search_query,
                "target_document_id": document_id,
                "query_id": query_id,
                "original_user_query": question
            }
            search_training_data.append(search_entry)
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    print(f"\nGenerated {len(search_training_data):,} search query entries")
    print(f"Average {len(search_training_data)/len(training_data):.1f} search queries per question")
    
    # Save search training data
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(search_training_data, f, indent=2)
    
    print(f"✅ Search query training data saved")
    
    # Show some examples
    print(f"\nExample search query entries:")
    for i, entry in enumerate(search_training_data[:5], 1):
        print(f"\n{i}. Search Query: '{entry['search_query']}'")
        print(f"   Original Question: {entry['original_user_query']}")
        print(f"   Target Document: {entry['target_document_id']}")
    
    # Show statistics by query_id
    query_counts = {}
    for entry in search_training_data:
        qid = entry['query_id']
        query_counts[qid] = query_counts.get(qid, 0) + 1
    
    print(f"\nSearch queries per question stats:")
    counts = list(query_counts.values())
    print(f"  Min: {min(counts)}, Max: {max(counts)}, Average: {sum(counts)/len(counts):.1f}")
    
    return search_training_data

def main():
    print("🔍 Search Query Generator")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        return
    
    # Check if input file exists
    input_file = "training_data_step_0.json"
    if not os.path.exists(input_file):
        print(f"❌ Input file {input_file} not found!")
        print("Please run generate_training_questions.py first to create the training data.")
        return
    
    try:
        search_data = generate_search_training_data(max_questions=500)
        
        print(f"\n✅ Successfully generated search query training data!")
        
    except KeyboardInterrupt:
        print("\n❌ Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")

if __name__ == "__main__":
    main()
