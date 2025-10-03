"""
Convert validated_query_chunk_pairs.jsonl to BEIR format
=========================================================
Converts synthetic query-chunk pairs into documents.jsonl, queries.jsonl, and qrels.jsonl
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict
import os


def find_input_file():
    """Find the validated_query_chunk_pairs.jsonl file"""
    print("Searching for validated_query_chunk_pairs.jsonl...")
    
    # Search in current directory and subdirectories
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == 'validated_query_chunk_pairs.jsonl':
                filepath = os.path.join(root, file)
                print(f"Found: {filepath}")
                return filepath
    
    return None


def generate_id(text: str, prefix: str = "") -> str:
    """Generate a short unique ID from text using hash"""
    hash_obj = hashlib.md5(text.encode('utf-8'))
    short_hash = hash_obj.hexdigest()[:10]
    return f"{prefix}{short_hash}" if prefix else short_hash


def convert_to_beir_format(input_file: str, output_dir: str):
    """
    Convert validated_query_chunk_pairs.jsonl to BEIR format
    
    Args:
        input_file: Path to validated_query_chunk_pairs.jsonl
        output_dir: Directory to write output files
    """
    input_path = Path(input_file)
    
    # Check if file exists
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        print(f"❌ Absolute path: {input_path.absolute()}")
        print(f"\nCurrent working directory: {os.getcwd()}")
        
        # Try to find the file
        found_file = find_input_file()
        if found_file:
            print(f"\n💡 Try running with: python {__file__} {found_file}")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collections to track unique documents and queries
    documents: Dict[str, Dict] = {}  # doc_id -> document
    queries: Dict[str, Dict] = {}    # query_id -> query
    qrels: list = []                 # list of query-doc relevance pairs
    
    # Track which chunk_ids we've seen to ensure unique documents
    seen_chunks: Set[str] = set()
    
    # Track query text to ID mapping (same query text = same ID)
    query_text_to_id: Dict[str, str] = {}
    
    print(f"Reading from: {input_path}")
    print(f"Output directory: {output_path}")
    
    # Read input file
    line_count = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            data = json.loads(line.strip())
            
            query_text = data['query']
            chunk_id = data['chunk_id']
            chunk_content = data['chunk_content']
            metadata = data.get('metadata', {})
            persona = data.get('persona', '')
            
            # Generate or retrieve query ID
            if query_text in query_text_to_id:
                query_id = query_text_to_id[query_text]
            else:
                query_id = generate_id(query_text, prefix="q_")
                query_text_to_id[query_text] = query_id
                
                # Add to queries collection
                queries[query_id] = {
                    "id": query_id,
                    "query": query_text,
                    "metadata": {
                        "persona": persona
                    }
                }
            
            # Generate document ID from chunk_id (ensures uniqueness)
            doc_id = generate_id(chunk_id, prefix="d_")
            
            # Add document if not already present
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                documents[doc_id] = {
                    "id": doc_id,
                    "content": chunk_content,
                    "metadata": metadata,
                    "scores": {}
                }
            
            # Add qrel (query-document relevance)
            qrels.append({
                "query_id": query_id,
                "document_id": doc_id,
                "score": 1.0  # All pairs are relevant in synthetic data
            })
    
    print(f"\nProcessed {line_count} lines")
    print(f"Unique documents: {len(documents)}")
    print(f"Unique queries: {len(queries)}")
    print(f"Total qrels: {len(qrels)}")
    
    # Write documents.jsonl
    documents_file = output_path / "documents.jsonl"
    with open(documents_file, 'w', encoding='utf-8') as f:
        for doc in documents.values():
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print(f"\nWrote {len(documents)} documents to {documents_file}")
    
    # Write queries.jsonl
    queries_file = output_path / "queries.jsonl"
    with open(queries_file, 'w', encoding='utf-8') as f:
        for query in queries.values():
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    print(f"Wrote {len(queries)} queries to {queries_file}")
    
    # Write qrels.jsonl
    qrels_file = output_path / "qrels.jsonl"
    with open(qrels_file, 'w', encoding='utf-8') as f:
        for qrel in qrels:
            f.write(json.dumps(qrel, ensure_ascii=False) + '\n')
    print(f"Wrote {len(qrels)} qrels to {qrels_file}")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels: {len(qrels)}")
    print(f"Avg queries per document: {len(qrels) / len(documents):.2f}")
    print(f"Avg documents per query: {len(qrels) / len(queries):.2f}")
    
    # Analyze persona distribution
    persona_counts = defaultdict(int)
    for query in queries.values():
        persona = query['metadata'].get('persona', 'unknown')
        persona_counts[persona] += 1
    
    print("\n=== Persona Distribution ===")
    for persona, count in sorted(persona_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{persona}: {count} queries ({count/len(queries)*100:.1f}%)")
    
    # Analyze chunk types
    chunk_type_counts = defaultdict(int)
    for doc in documents.values():
        chunk_type = doc['metadata'].get('chunk_type', 'unknown')
        chunk_type_counts[chunk_type] += 1
    
    print("\n=== Chunk Type Distribution ===")
    for chunk_type, count in sorted(chunk_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{chunk_type}: {count} documents ({count/len(documents)*100:.1f}%)")
    
    # Show sample entries
    print("\n=== Sample Document ===")
    sample_doc = list(documents.values())[0]
    print(json.dumps(sample_doc, indent=2, ensure_ascii=False)[:500] + "...")
    
    print("\n=== Sample Query ===")
    sample_query = list(queries.values())[0]
    print(json.dumps(sample_query, indent=2, ensure_ascii=False)[:500] + "...")
    
    print("\n=== Sample Qrel ===")
    sample_qrel = qrels[0]
    print(json.dumps(sample_qrel, indent=2, ensure_ascii=False))


def verify_output(output_dir: str):
    """Verify the generated files are valid"""
    output_path = Path(output_dir)
    
    print("\n=== Verification ===")
    
    # Check files exist
    required_files = ["documents.jsonl", "queries.jsonl", "qrels.jsonl"]
    for filename in required_files:
        filepath = output_path / filename
        if not filepath.exists():
            print(f"❌ Missing file: {filename}")
            return False
        print(f"✓ Found {filename}")
    
    # Verify each file is valid JSONL
    for filename in required_files:
        filepath = output_path / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    json.loads(line.strip())
                    line_count += 1
                print(f"✓ {filename}: {line_count} valid JSON lines")
        except json.JSONDecodeError as e:
            print(f"❌ {filename}: Invalid JSON on line {line_count + 1}: {e}")
            return False
    
    # Verify qrels reference valid queries and documents
    documents_file = output_path / "documents.jsonl"
    queries_file = output_path / "queries.jsonl"
    qrels_file = output_path / "qrels.jsonl"
    
    # Load document IDs
    doc_ids = set()
    with open(documents_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_ids.add(doc['id'])
    
    # Load query IDs
    query_ids = set()
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line.strip())
            query_ids.add(query['id'])
    
    # Check qrels
    invalid_qrels = 0
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            qrel = json.loads(line.strip())
            if qrel['query_id'] not in query_ids:
                print(f"❌ Invalid query_id in qrel: {qrel['query_id']}")
                invalid_qrels += 1
            if qrel['document_id'] not in doc_ids:
                print(f"❌ Invalid document_id in qrel: {qrel['document_id']}")
                invalid_qrels += 1
    
    if invalid_qrels == 0:
        print(f"✓ All qrels reference valid queries and documents")
    else:
        print(f"❌ Found {invalid_qrels} invalid qrels")
        return False
    
    print("\n✅ All verifications passed!")
    return True


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Auto-generate output dir if not provided
        if output_dir is None:
            input_path = Path(input_file)
            output_dir = str(input_path.parent / "beir_format")
    else:
        # Try to find the file automatically
        print("No input file specified. Searching for validated_query_chunk_pairs.jsonl...")
        input_file = find_input_file()
        
        if input_file is None:
            print("\n❌ Could not find validated_query_chunk_pairs.jsonl")
            print("\nUsage:")
            print("  python validated_query_chunk_pairs_to_proper_json_format.py <input_file> [output_dir]")
            print("\nExample:")
            print("  python validated_query_chunk_pairs_to_proper_json_format.py \\")
            print("      synthetic_data/workspace_name/validated_query_chunk_pairs.jsonl \\")
            print("      synthetic_data/workspace_name/beir_format")
            sys.exit(1)
        
        # Auto-generate output dir
        input_path = Path(input_file)
        output_dir = str(input_path.parent / "beir_format")
        print(f"✓ Found input file: {input_file}")
        print(f"✓ Output directory: {output_dir}\n")
    
    # Convert
    convert_to_beir_format(input_file, output_dir)
    
    # Verify
    if Path(output_dir).exists():
        verify_output(output_dir)
        
        print("\n✅ Conversion complete!")
        print(f"Files written to: {output_dir}")