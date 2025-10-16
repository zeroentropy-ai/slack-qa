#!/usr/bin/env python3
"""
Manual search explorer for developing few-shot examples
Allows interactive testing of search queries against random documents
"""
import json
import random
from test_masked_search import test_query_with_content

def load_documents():
    """Load the training documents"""
    with open("traingen_documents.json", 'r') as f:
        documents = json.load(f)
    return documents

def show_document(doc):
    """Display a document for manual search testing"""
    print("=" * 80)
    print(f"DOCUMENT ID: {doc['id']}")
    print(f"DATASET: {doc['metadata']['dataset']}")
    print(f"TITLE: {doc['metadata'].get('title', 'No title')}")
    print(f"ORIGINAL QUERY CONTEXT: {doc['metadata']['query_context']}")
    print("-" * 80)
    print("CONTENT:")
    content = doc['content']
    if len(content) > 1000:
        print(f"{content[:1000]}... [TRUNCATED - Full length: {len(content)} chars]")
    else:
        print(content)
    print("=" * 80)

def manual_search_session():
    """Interactive manual search session"""
    documents = load_documents()
    print(f"Loaded {len(documents):,} documents for manual testing")
    
    examples = []  # Store successful search examples
    
    while True:
        print(f"\n{'='*60}")
        print("MANUAL SEARCH EXPLORER")
        print("Commands: 'new' for new random document, 'search <query>' to test, 'save' to save examples, 'quit' to exit")
        print(f"Current successful examples: {len(examples)}")
        
        cmd = input("\nCommand: ").strip()
        
        if cmd == 'quit':
            break
        elif cmd == 'new':
            # Show a random document
            current_doc = random.choice(documents)
            show_document(current_doc)
            print(f"\n🎯 YOUR GOAL: Find search queries that would retrieve this document")
            print(f"Think about what a user might ask that this document answers...")
            
        elif cmd.startswith('search '):
            query = cmd[7:].strip()
            if not query:
                print("Please provide a search query")
                continue
            
            if 'current_doc' not in locals():
                print("Please select a document first with 'new'")
                continue
                
            print(f"\n🔍 Testing query: \"{query}\"")
            print(f"🎯 Looking for document: {current_doc['id']}")
            
            # Test the query
            docs, keywords_matched = test_query_with_content(query, max_results=20)
            
            # Check if target document was found
            found_rank = None
            for i, result_doc in enumerate(docs, 1):
                if result_doc['id'] == current_doc['id']:
                    found_rank = i
                    break
            
            if found_rank:
                print(f"✅ SUCCESS! Found target document at rank {found_rank}")
                print(f"Keywords matched: {keywords_matched}")
                
                # Ask if user wants to save this example
                save = input("Save this as a successful example? (y/n): ").strip().lower()
                if save == 'y':
                    # Generate a realistic user question for this document
                    print(f"\nNow create a realistic user question that this document answers:")
                    print(f"Document content preview: {current_doc['content'][:200]}...")
                    user_question = input("User question: ").strip()
                    
                    if user_question:
                        examples.append({
                            "user_question": user_question,
                            "target_document_id": current_doc['id'],
                            "successful_search_query": query,
                            "rank_found": found_rank,
                            "keywords_matched": keywords_matched,
                            "document_preview": current_doc['content'][:300],
                            "dataset": current_doc['metadata']['dataset']
                        })
                        print(f"✅ Saved example #{len(examples)}")
            else:
                print(f"❌ Target document not found in top 20 results")
                if keywords_matched > 0:
                    print(f"Search did return {len(docs)} results using {keywords_matched} keywords")
                else:
                    print("Search returned no results")
        
        elif cmd == 'save':
            if examples:
                filename = f"manual_search_examples_{len(examples)}.json"
                with open(filename, 'w') as f:
                    json.dump(examples, f, indent=2)
                print(f"💾 Saved {len(examples)} examples to {filename}")
            else:
                print("No examples to save")
        
        elif cmd == 'examples':
            if examples:
                print(f"\n📋 CURRENT EXAMPLES ({len(examples)}):")
                for i, ex in enumerate(examples, 1):
                    print(f"{i}. Question: \"{ex['user_question']}\"")
                    print(f"   Query: \"{ex['successful_search_query']}\" (rank {ex['rank_found']})")
                    print(f"   Dataset: {ex['dataset']}")
                    print()
            else:
                print("No examples saved yet")
        
        else:
            print("Unknown command. Use 'new', 'search <query>', 'save', 'examples', or 'quit'")

if __name__ == "__main__":
    print("🔍 MANUAL SEARCH EXPLORER")
    print("This tool helps you develop few-shot examples for LLM prompt engineering")
    print("You'll test search queries manually to understand what works")
    print()
    
    manual_search_session()