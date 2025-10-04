"""
Convert validated_query_chunk_pairs.jsonl to BEIR format (Multi-Workspace)
===========================================================================
Processes each workspace independently to create documents.jsonl, queries.jsonl, and qrels.jsonl

Documents are sourced from chunks/ subdirectory (all chunks)
Queries and qrels are sourced from validated_query_chunk_pairs.jsonl
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict
import os


def generate_id(text: str, prefix: str = "") -> str:
    """Generate a short unique ID from text using hash"""
    hash_obj = hashlib.md5(text.encode('utf-8'))
    short_hash = hash_obj.hexdigest()[:10]
    return f"{prefix}{short_hash}" if prefix else short_hash


def find_workspaces(base_dir: str = "synthetic_data") -> List[Tuple[str, Path, Path]]:
    """
    Find all workspaces with both chunks/ and validated_query_chunk_pairs.jsonl
    
    Returns:
        List of tuples: (workspace_name, chunks_dir, validated_pairs_file)
    """
    workspaces = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return workspaces
    
    print(f"🔍 Searching for workspaces in {base_dir}...\n")
    
    # Look for workspace directories
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        
        workspace_name = item.name
        chunks_dir = item / "chunks"
        validated_file = item / "validated_query_chunk_pairs.jsonl"
        
        # Check if workspace has required files
        has_chunks = chunks_dir.exists() and chunks_dir.is_dir()
        has_validated = validated_file.exists() and validated_file.is_file()
        
        if has_chunks and has_validated:
            workspaces.append((workspace_name, chunks_dir, validated_file))
            print(f"✓ Found workspace: {workspace_name}")
            print(f"  - Chunks: {chunks_dir}")
            print(f"  - Validated pairs: {validated_file}")
        else:
            missing = []
            if not has_chunks:
                missing.append("chunks/")
            if not has_validated:
                missing.append("validated_query_chunk_pairs.jsonl")
            print(f"⚠ Skipping {workspace_name} (missing: {', '.join(missing)})")
        print()
    
    return workspaces


def load_all_chunks(chunks_dir: Path) -> Dict[str, Dict]:
    """
    Load all chunks from the chunks directory
    
    Args:
        chunks_dir: Path to chunks directory
        
    Returns:
        Dictionary mapping chunk_id to chunk data
    """
    chunks = {}
    
    # Expected chunk files
    chunk_files = [
        "individual_message.jsonl",
        "sliding_window.jsonl",
        "thread.jsonl"
    ]
    
    print(f"  📄 Loading chunks from {chunks_dir}...")
    
    for filename in chunk_files:
        filepath = chunks_dir / filename
        
        if not filepath.exists():
            print(f"    ⚠ File not found: {filename}")
            continue
        
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                chunk_id = data['chunk_id']
                
                if chunk_id in chunks:
                    print(f"    ⚠ Duplicate chunk_id: {chunk_id}")
                
                chunks[chunk_id] = {
                    'chunk_id': chunk_id,
                    'content': data['content'],
                    'metadata': data.get('metadata', {})
                }
                count += 1
        
        print(f"    ✓ Loaded {count} chunks from {filename}")
    
    return chunks


def process_workspace(workspace_name: str, chunks_dir: Path, validated_file: Path, output_base: str = "synthetic_data"):
    """
    Process a single workspace to generate BEIR format files
    
    Args:
        workspace_name: Name of the workspace
        chunks_dir: Path to chunks directory
        validated_file: Path to validated_query_chunk_pairs.jsonl
        output_base: Base directory for output
    """
    print(f"\n{'='*80}")
    print(f"Processing workspace: {workspace_name}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path(output_base) / workspace_name / "beir_format"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load ALL chunks as documents
    print("Step 1: Loading all chunks as documents...")
    all_chunks = load_all_chunks(chunks_dir)
    print(f"  ✓ Total chunks loaded: {len(all_chunks)}\n")
    
    # Step 2: Create documents from all chunks
    print("Step 2: Creating documents...")
    documents = {}
    for chunk_id, chunk_data in all_chunks.items():
        doc_id = generate_id(chunk_id, prefix="d_")
        documents[doc_id] = {
            "id": doc_id,
            "content": chunk_data['content'],
            "metadata": chunk_data['metadata'],
            "scores": {}
        }
    print(f"  ✓ Created {len(documents)} documents\n")
    
    # Create mapping from chunk_id to doc_id for qrels
    chunk_id_to_doc_id = {
        chunk_id: generate_id(chunk_id, prefix="d_")
        for chunk_id in all_chunks.keys()
    }
    
    # Step 3: Load queries and create qrels from validated pairs
    print("Step 3: Loading queries and creating qrels...")
    queries = {}
    qrels = []
    query_text_to_id = {}
    
    missing_chunks = set()
    line_count = 0
    
    with open(validated_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            data = json.loads(line.strip())
            
            query_text = data['query']
            chunk_id = data['chunk_id']
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
            
            # Get document ID for this chunk
            if chunk_id not in chunk_id_to_doc_id:
                missing_chunks.add(chunk_id)
                continue
            
            doc_id = chunk_id_to_doc_id[chunk_id]
            
            # Add qrel (query-document relevance)
            qrels.append({
                "query_id": query_id,
                "document_id": doc_id,
                "score": 1.0
            })
    
    print(f"  ✓ Processed {line_count} query-chunk pairs")
    print(f"  ✓ Created {len(queries)} unique queries")
    print(f"  ✓ Created {len(qrels)} qrels")
    
    if missing_chunks:
        print(f"  ⚠ Warning: {len(missing_chunks)} chunk_ids from validated pairs not found in chunks/")
        print(f"    First few missing: {list(missing_chunks)[:5]}")
    print()
    
    # Step 4: Write output files
    print("Step 4: Writing output files...")
    
    # Write documents.jsonl
    documents_file = output_dir / "documents.jsonl"
    with open(documents_file, 'w', encoding='utf-8') as f:
        for doc in documents.values():
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print(f"  ✓ Wrote {len(documents)} documents to {documents_file}")
    
    # Write queries.jsonl
    queries_file = output_dir / "queries.jsonl"
    with open(queries_file, 'w', encoding='utf-8') as f:
        for query in queries.values():
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    print(f"  ✓ Wrote {len(queries)} queries to {queries_file}")
    
    # Write qrels.jsonl
    qrels_file = output_dir / "qrels.jsonl"
    with open(qrels_file, 'w', encoding='utf-8') as f:
        for qrel in qrels:
            f.write(json.dumps(qrel, ensure_ascii=False) + '\n')
    print(f"  ✓ Wrote {len(qrels)} qrels to {qrels_file}")
    
    # Step 5: Print statistics
    print(f"\n{'='*80}")
    print(f"Statistics for {workspace_name}")
    print(f"{'='*80}")
    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels: {len(qrels)}")
    
    if len(queries) > 0:
        print(f"Avg documents per query: {len(qrels) / len(queries):.2f}")
    if len(documents) > 0:
        print(f"Avg queries per document: {len(qrels) / len(documents):.2f}")
    
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
    if documents:
        print("\n=== Sample Document ===")
        sample_doc = list(documents.values())[0]
        sample_str = json.dumps(sample_doc, indent=2, ensure_ascii=False)
        print(sample_str[:500] + ("..." if len(sample_str) > 500 else ""))
    
    if queries:
        print("\n=== Sample Query ===")
        sample_query = list(queries.values())[0]
        sample_str = json.dumps(sample_query, indent=2, ensure_ascii=False)
        print(sample_str[:500] + ("..." if len(sample_str) > 500 else ""))
    
    if qrels:
        print("\n=== Sample Qrel ===")
        sample_qrel = qrels[0]
        print(json.dumps(sample_qrel, indent=2, ensure_ascii=False))
    
    return len(documents), len(queries), len(qrels)


def verify_workspace_output(workspace_name: str, output_dir: Path):
    """Verify the generated files are valid for a workspace"""
    print(f"\n{'='*80}")
    print(f"Verification for {workspace_name}")
    print(f"{'='*80}")
    
    # Check files exist
    required_files = ["documents.jsonl", "queries.jsonl", "qrels.jsonl"]
    for filename in required_files:
        filepath = output_dir / filename
        if not filepath.exists():
            print(f"❌ Missing file: {filename}")
            return False
        print(f"✓ Found {filename}")
    
    # Verify each file is valid JSONL
    for filename in required_files:
        filepath = output_dir / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    json.loads(line.strip())
                    line_count += 1
                print(f"✓ {filename}: {line_count} valid JSON lines")
        except json.JSONDecodeError as e:
            print(f"❌ {filename}: Invalid JSON: {e}")
            return False
    
    # Verify qrels reference valid queries and documents
    documents_file = output_dir / "documents.jsonl"
    queries_file = output_dir / "queries.jsonl"
    qrels_file = output_dir / "qrels.jsonl"
    
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
    
    # Get base directory from command line or use default
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data"
    
    print(f"{'='*80}")
    print(f"BEIR Format Converter - Multi-Workspace")
    print(f"{'='*80}\n")
    
    # Find all workspaces
    workspaces = find_workspaces(base_dir)
    
    if not workspaces:
        print("\n❌ No valid workspaces found!")
        print("\nA valid workspace must have:")
        print("  1. A chunks/ subdirectory with chunk files")
        print("  2. A validated_query_chunk_pairs.jsonl file")
        print("\nUsage:")
        print("  python validated_query_chunk_pairs_to_proper_json_format.py [base_dir]")
        print("\nExample:")
        print("  python validated_query_chunk_pairs_to_proper_json_format.py synthetic_data")
        sys.exit(1)
    
    print(f"{'='*80}")
    print(f"Found {len(workspaces)} valid workspace(s)")
    print(f"{'='*80}\n")
    
    # Process each workspace
    results = []
    for workspace_name, chunks_dir, validated_file in workspaces:
        try:
            doc_count, query_count, qrel_count = process_workspace(
                workspace_name,
                chunks_dir,
                validated_file,
                base_dir
            )
            results.append((workspace_name, doc_count, query_count, qrel_count, True))
            
            # Verify output
            output_dir = Path(base_dir) / workspace_name / "beir_format"
            verify_workspace_output(workspace_name, output_dir)
            
        except Exception as e:
            print(f"\n❌ Error processing {workspace_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((workspace_name, 0, 0, 0, False))
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    successful = sum(1 for _, _, _, _, success in results if success)
    print(f"Processed {len(workspaces)} workspace(s): {successful} successful, {len(workspaces) - successful} failed\n")
    
    print(f"{'Workspace':<50} {'Docs':<8} {'Queries':<8} {'Qrels':<8} {'Status'}")
    print(f"{'-'*50} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    
    for workspace, docs, queries, qrels, success in results:
        status = "✓" if success else "✗"
        workspace_short = workspace[:47] + "..." if len(workspace) > 50 else workspace
        print(f"{workspace_short:<50} {docs:<8} {queries:<8} {qrels:<8} {status}")
    
    print(f"\n{'='*80}")
    print(f"✅ Conversion complete!")
    print(f"{'='*80}")