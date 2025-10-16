#!/usr/bin/env python3
"""
Generate training questions from documents by:
1. Randomly selecting documents from traingen_documents.json
2. Evaluating if document is suitable for question generation
3. Generating a relevant question using OpenAI
4. Saving to training_data_step_0.json
"""
import json
import random
import uuid
import os
import time
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import openai

# Set up OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_documents(file_path: str = "traingen_documents.json") -> List[Dict[str, Any]]:
    """Load documents from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def is_suitable_for_question(content: str, dataset: str) -> bool:
    """
    Evaluate if document content is suitable for generating a training question.
    Uses OpenAI to determine if the content contains valuable information that 
    would warrant someone asking a question about it.
    """
    
    # Skip very short content
    if len(content.strip()) < 50:
        return False
    
    prompt = f"""Evaluate if this document content contains valuable information that someone would want to ask a question about. Consider:

1. Does it contain useful technical information, explanations, or solutions?
2. Would someone search for this information to solve a problem or learn something?
3. Is it substantive enough to warrant a meaningful question?

Answer only "YES" or "NO".

Document content: {content[:800]}
Dataset: {dataset}"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        
        answer = response.choices[0].message.content.strip().upper()
        return answer == "YES"
        
    except Exception as e:
        print(f"Error evaluating suitability: {e}")
        # Default to conservative approach - only accept if clearly valuable
        return len(content) > 200 and any(keyword in content.lower() for keyword in 
            ['how', 'why', 'what', 'error', 'problem', 'solution', 'tutorial', 'guide', 'explain'])

def generate_question(content: str, title: str, dataset: str) -> Optional[str]:
    """
    Generate a natural question that this document could answer.
    The question should be something a user might realistically ask.
    """
    
    prompt = f"""Generate a natural, realistic question that someone might ask an AI assistant, where this document would be a good answer.

Requirements:
- The question should be specific and actionable
- It should sound like something a real user would ask
- Don't just rephrase the content - ask what someone would want to KNOW
- Make it a question someone would search for or ask a chatbot

Document title: {title}
Document content: {content[:1000]}
Dataset: {dataset}

Generate only the question, nothing else."""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        
        question = response.choices[0].message.content.strip()
        
        # Clean up the question
        if not question.endswith('?'):
            question += '?'
        
        return question
        
    except Exception as e:
        print(f"Error generating question: {e}")
        return None

def generate_training_data(
    input_file: str = "traingen_documents.json",
    output_file: str = "training_data_step_0.json", 
    target_count: int = 1000,
    sample_size: int = 5000
) -> List[Dict[str, Any]]:
    """
    Generate training data by sampling documents and creating questions.
    
    Args:
        input_file: Input document file
        output_file: Output training data file
        target_count: Target number of training examples
        sample_size: Number of documents to sample for evaluation
    """
    
    print(f"Loading documents from {input_file}...")
    documents = load_documents(input_file)
    
    print(f"Loaded {len(documents):,} documents")
    print(f"Sampling {sample_size:,} documents for evaluation...")
    
    # Randomly sample documents
    sampled_docs = random.sample(documents, min(sample_size, len(documents)))
    
    training_data = []
    suitable_count = 0
    
    for doc in tqdm(sampled_docs, desc="Processing documents"):
        # Check if document is suitable
        is_suitable = is_suitable_for_question(
            doc['content'], 
            doc['metadata']['dataset']
        )
        
        if is_suitable:
            suitable_count += 1
            
            # Generate question
            question = generate_question(
                doc['content'],
                doc['metadata'].get('title', ''),
                doc['metadata']['dataset']
            )
            
            if question:
                training_entry = {
                    "document_id": doc['id'],
                    "content": doc['content'],
                    "query_id": str(uuid.uuid4()),
                    "question": question,
                    "metadata": {
                        "dataset": doc['metadata']['dataset'],
                        "title": doc['metadata'].get('title', ''),
                        "original_query_context": doc['metadata'].get('query_context', '')
                    }
                }
                
                training_data.append(training_entry)
                
                # Stop if we reach target count
                if len(training_data) >= target_count:
                    break
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
    
    print(f"\nResults:")
    print(f"Documents processed: {len(sampled_docs):,}")
    print(f"Suitable for questions: {suitable_count:,} ({100*suitable_count/len(sampled_docs):.1f}%)")
    print(f"Questions generated: {len(training_data):,}")
    
    # Save training data
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"✅ Training data saved with {len(training_data)} examples")
    
    # Show some examples
    print(f"\nExample training entries:")
    for i, entry in enumerate(training_data[:3], 1):
        print(f"\n{i}. Question: {entry['question']}")
        print(f"   Document preview: {entry['content'][:150]}...")
        print(f"   Dataset: {entry['metadata']['dataset']}")
    
    return training_data

def main():
    print("🤖 Training Question Generator")
    print("=" * 50)
    
    # Configuration
    target_count = int(input("Target number of training examples (default 1000): ") or "1000")
    sample_size = int(input("Documents to sample for evaluation (default 5000): ") or "5000")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        return
    
    try:
        training_data = generate_training_data(
            target_count=target_count,
            sample_size=sample_size
        )
        
        print(f"\n✅ Successfully generated {len(training_data)} training examples!")
        
    except KeyboardInterrupt:
        print("\n❌ Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")

if __name__ == "__main__":
    main()
