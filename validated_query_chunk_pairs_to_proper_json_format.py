"""
Convert validated_query_chunk_pairs.jsonl to BEIR format (Multi-Workspace, Multi-Strategy)
===========================================================================================
Processes each workspace independently to create separate BEIR datasets for each chunking strategy.

For each workspace and chunking strategy, creates:
- beir_format_sliding_window/
- beir_format_individual_message/
- beir_format_thread/

Each containing: documents.jsonl, queries.jsonl, and qrels.jsonl
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


def load_chunks_by_strategy(chunks_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Load all chunks from the chunks directory, organized by chunking strategy
    
    Args:
        chunks_dir: Path to chunks directory
        
    Returns:
        Dictionary mapping chunk_type -> chunk_id -> chunk data
    """
    chunks_by_strategy = {
        "sliding_window": {},
        "individual_message": {},
        "thread": {}
    }
    
    # Map filenames to chunk types
    file_to_strategy = {
        "sliding_window.jsonl": "sliding_window",
        "individual_message.jsonl": "individual_message",
        "thread.jsonl": "thread"
    }
    
    print(f"  📄 Loading chunks from {chunks_dir}...")
    
    for filename, strategy in file_to_strategy.items():
        filepath = chunks_dir / filename
        
        if not filepath.exists():
            print(f"    ⚠ File not found: {filename}")
            continue
        
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                chunk_id = data['chunk_id']
                
                if chunk_id in chunks_by_strategy[strategy]:
                    print(f"    ⚠ Duplicate chunk_id in {strategy}: {chunk_id}")
                
                chunks_by_strategy[strategy][chunk_id] = {
                    'chunk_id': chunk_id,
                    'content': data['content'],
                    'metadata': data.get('metadata', {})
                }
                count += 1
        
        print(f"    ✓ Loaded {count} chunks from {filename} ({strategy})")
    
    return chunks_by_strategy


def process_workspace(workspace_name: str, chunks_dir: Path, validated_file: Path, output_base: str = "synthetic_data"):
    """
    Process a single workspace to generate BEIR format files for each chunking strategy
    
    Args:
        workspace_name: Name of the workspace
        chunks_dir: Path to chunks directory
        validated_file: Path to validated_query_chunk_pairs.jsonl
        output_base: Base directory for output
    """
    print(f"\n{'='*80}")
    print(f"Processing workspace: {workspace_name}")
    print(f"{'='*80}\n")
    
    # Step 1: Load chunks organized by strategy
    print("Step 1: Loading chunks by strategy...")
    chunks_by_strategy = load_chunks_by_strategy(chunks_dir)
    
    total_chunks = sum(len(chunks) for chunks in chunks_by_strategy.values())
    print(f"  ✓ Total chunks loaded: {total_chunks}")
    for strategy, chunks in chunks_by_strategy.items():
        print(f"    - {strategy}: {len(chunks)} chunks")
    print()
    
    # Step 2: Load validated pairs and organize by strategy
    print("Step 2: Loading validated query-chunk pairs...")
    pairs_by_strategy = {
        "sliding_window": [],
        "individual_message": [],
        "thread": []
    }
    
    missing_chunks = set()
    unknown_strategy = []
    line_count = 0
    
    with open(validated_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            data = json.loads(line.strip())
            
            query_text = data['query']
            chunk_id = data['chunk_id']
            persona = data.get('persona', '')
            
            # Find which strategy this chunk belongs to
            found_strategy = None
            for strategy, chunks in chunks_by_strategy.items():
                if chunk_id in chunks:
                    found_strategy = strategy
                    break
            
            if found_strategy is None:
                missing_chunks.add(chunk_id)
                continue
            
            # Add to appropriate strategy
            pairs_by_strategy[found_strategy].append({
                'query': query_text,
                'chunk_id': chunk_id,
                'persona': persona
            })
    
    print(f"  ✓ Processed {line_count} query-chunk pairs")
    for strategy, pairs in pairs_by_strategy.items():
        print(f"    - {strategy}: {len(pairs)} pairs")
    
    if missing_chunks:
        print(f"  ⚠ Warning: {len(missing_chunks)} chunk_ids not found in any strategy")
        print(f"    First few missing: {list(missing_chunks)[:5]}")
    print()
    
    # Step 3: Process each strategy separately
    results = {}
    
    for strategy in ["sliding_window", "individual_message", "thread"]:
        print(f"\n{'='*80}")
        print(f"Processing strategy: {strategy}")
        print(f"{'='*80}\n")
        
        chunks = chunks_by_strategy[strategy]
        pairs = pairs_by_strategy[strategy]
        
        if not chunks:
            print(f"  ⚠ No chunks for {strategy}, skipping...")
            results[strategy] = (0, 0, 0)
            continue
        
        # Create output directory for this strategy
        output_dir = Path(output_base) / workspace_name / f"beir_format_{strategy}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create documents from chunks (all chunks for this strategy)
        print(f"  Creating documents for {strategy}...")
        documents = {}
        chunk_id_to_doc_id = {}
        
        for chunk_id, chunk_data in chunks.items():
            doc_id = generate_id(chunk_id, prefix="d_")
            documents[doc_id] = {
                "id": doc_id,
                "content": chunk_data['content'],
                "metadata": chunk_data['metadata'],
                "scores": {}
            }
            chunk_id_to_doc_id[chunk_id] = doc_id
        
        print(f"  ✓ Created {len(documents)} documents for {strategy}")
        
        # Create queries and qrels from pairs
        print(f"  Creating queries and qrels for {strategy}...")
        queries = {}
        qrels = []
        query_text_to_id = {}
        
        for pair in pairs:
            query_text = pair['query']
            chunk_id = pair['chunk_id']
            persona = pair['persona']
            
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
            doc_id = chunk_id_to_doc_id[chunk_id]
            
            # Add qrel (query-document relevance)
            qrels.append({
                "query_id": query_id,
                "document_id": doc_id,
                "score": 1.0
            })
        
        print(f"  ✓ Created {len(queries)} unique queries for {strategy}")
        print(f"  ✓ Created {len(qrels)} qrels for {strategy}")
        
        # Write output files for this strategy
        print(f"  Writing output files for {strategy}...")
        
        # Write documents.jsonl
        documents_file = output_dir / "documents.jsonl"
        with open(documents_file, 'w', encoding='utf-8') as f:
            for doc in documents.values():
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        print(f"    ✓ Wrote {len(documents)} documents to {documents_file}")
        
        # Write queries.jsonl
        queries_file = output_dir / "queries.jsonl"
        with open(queries_file, 'w', encoding='utf-8') as f:
            for query in queries.values():
                f.write(json.dumps(query, ensure_ascii=False) + '\n')
        print(f"    ✓ Wrote {len(queries)} queries to {queries_file}")
        
        # Write qrels.jsonl
        qrels_file = output_dir / "qrels.jsonl"
        with open(qrels_file, 'w', encoding='utf-8') as f:
            for qrel in qrels:
                f.write(json.dumps(qrel, ensure_ascii=False) + '\n')
        print(f"    ✓ Wrote {len(qrels)} qrels to {qrels_file}")
        
        results[strategy] = (len(documents), len(queries), len(qrels))
        
        # Print statistics for this strategy
        print(f"\n  === Statistics for {strategy} ===")
        print(f"  Documents: {len(documents)}")
        print(f"  Queries: {len(queries)}")
        print(f"  Qrels: {len(qrels)}")
        
        if len(queries) > 0:
            print(f"  Avg documents per query: {len(qrels) / len(queries):.2f}")
        if len(documents) > 0:
            print(f"  Avg queries per document: {len(qrels) / len(documents):.2f}")
        
        # Analyze persona distribution
        persona_counts = defaultdict(int)
        for query in queries.values():
            persona = query['metadata'].get('persona', 'unknown')
            persona_counts[persona] += 1
        
        if persona_counts:
            print(f"\n  === Persona Distribution for {strategy} ===")
            for persona, count in sorted(persona_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {persona}: {count} queries ({count/len(queries)*100:.1f}%)")
        
        # Show sample entries
        if documents:
            print(f"\n  === Sample Document for {strategy} ===")
            sample_doc = list(documents.values())[0]
            sample_str = json.dumps(sample_doc, indent=2, ensure_ascii=False)
            print("  " + sample_str[:500].replace("\n", "\n  ") + ("..." if len(sample_str) > 500 else ""))
        
        if queries:
            print(f"\n  === Sample Query for {strategy} ===")
            sample_query = list(queries.values())[0]
            print("  " + json.dumps(sample_query, indent=2, ensure_ascii=False).replace("\n", "\n  "))
        
        if qrels:
            print(f"\n  === Sample Qrel for {strategy} ===")
            sample_qrel = qrels[0]
            print("  " + json.dumps(sample_qrel, indent=2, ensure_ascii=False).replace("\n", "\n  "))
    
    return results


def verify_workspace_output(workspace_name: str, output_base: str):
    """Verify the generated files are valid for a workspace (all strategies)"""
    print(f"\n{'='*80}")
    print(f"Verification for {workspace_name}")
    print(f"{'='*80}")
    
    all_valid = True
    
    for strategy in ["sliding_window", "individual_message", "thread"]:
        print(f"\n  --- Verifying {strategy} ---")
        
        output_dir = Path(output_base) / workspace_name / f"beir_format_{strategy}"
        
        if not output_dir.exists():
            print(f"  ⚠ Directory not found: {output_dir}")
            continue
        
        # Check files exist
        required_files = ["documents.jsonl", "queries.jsonl", "qrels.jsonl"]
        strategy_valid = True
        
        for filename in required_files:
            filepath = output_dir / filename
            if not filepath.exists():
                print(f"  ❌ Missing file: {filename}")
                strategy_valid = False
                all_valid = False
                continue
            print(f"  ✓ Found {filename}")
        
        if not strategy_valid:
            continue
        
        # Verify each file is valid JSONL
        for filename in required_files:
            filepath = output_dir / filename
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    line_count = 0
                    for line in f:
                        json.loads(line.strip())
                        line_count += 1
                    print(f"  ✓ {filename}: {line_count} valid JSON lines")
            except json.JSONDecodeError as e:
                print(f"  ❌ {filename}: Invalid JSON: {e}")
                strategy_valid = False
                all_valid = False
        
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
                    print(f"  ❌ Invalid query_id in qrel: {qrel['query_id']}")
                    invalid_qrels += 1
                if qrel['document_id'] not in doc_ids:
                    print(f"  ❌ Invalid document_id in qrel: {qrel['document_id']}")
                    invalid_qrels += 1
        
        if invalid_qrels == 0:
            print(f"  ✓ All qrels reference valid queries and documents")
        else:
            print(f"  ❌ Found {invalid_qrels} invalid qrels")
            strategy_valid = False
            all_valid = False
        
        if strategy_valid:
            print(f"  ✅ {strategy} passed all verifications")
    
    if all_valid:
        print(f"\n✅ All strategies passed verification for {workspace_name}!")
    else:
        print(f"\n⚠ Some strategies failed verification for {workspace_name}")
    
    return all_valid


if __name__ == "__main__":
    import sys
    
    # Get base directory from command line or use default
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data"
    
    print(f"{'='*80}")
    print(f"BEIR Format Converter - Multi-Workspace, Multi-Strategy")
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
    all_results = []
    for workspace_name, chunks_dir, validated_file in workspaces:
        try:
            strategy_results = process_workspace(
                workspace_name,
                chunks_dir,
                validated_file,
                base_dir
            )
            all_results.append((workspace_name, strategy_results, True))
            
            # Verify output
            verify_workspace_output(workspace_name, base_dir)
            
        except Exception as e:
            print(f"\n❌ Error processing {workspace_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append((workspace_name, {}, False))
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    successful = sum(1 for _, _, success in all_results if success)
    print(f"Processed {len(workspaces)} workspace(s): {successful} successful, {len(workspaces) - successful} failed\n")
    
    for workspace, strategy_results, success in all_results:
        status = "✓" if success else "✗"
        print(f"\n{workspace} {status}")
        
        if success and strategy_results:
            print(f"{'  Strategy':<30} {'Docs':<8} {'Queries':<8} {'Qrels':<8}")
            print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
            
            for strategy in ["sliding_window", "individual_message", "thread"]:
                if strategy in strategy_results:
                    docs, queries, qrels = strategy_results[strategy]
                    print(f"  {strategy:<30} {docs:<8} {queries:<8} {qrels:<8}")
    
    print(f"\n{'='*80}")
    print(f"✅ Conversion complete!")
    print(f"✅ Each chunking strategy has its own isolated BEIR dataset")
    print(f"{'='*80}")