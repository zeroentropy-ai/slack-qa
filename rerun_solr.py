#!/usr/bin/env python3
"""
Force re-run all improved Solr queries (useful after stop word list changes)
"""

import json
from pathlib import Path
import sys
import asyncio
sys.path.append('mock-slack')
from solr_search import solr_with_termfreq
from typing import Dict

async def search_solr_improved(search_query: str, workspace: str) -> Dict:
    """Search Solr using improved search with camelCase and stemming"""
    try:
        # Map workspace names to collection names
        collection_map = {
            "Manifest": "training-slack",
        }
        
        collection = collection_map.get(workspace, "training-slack")
        
        # Fetch 100 documents as requested
        solr_response = await solr_with_termfreq(search_query, collection, rows=1000, cond="OR")
        
        # Extract data from Solr response
        response_data = solr_response.get('response', {})
        docs = response_data.get('docs', [])
        num_found = response_data.get('numFound', 0)
        
        return {
            "query": search_query,
            "total_results": num_found,  # Use actual total from Solr
            "documents": docs,  # Return all 100 docs
            "workspace": workspace
        }
    except Exception as e:
        print(f"Error searching Solr for '{search_query}': {e}")
        return {
            "query": search_query,
            "error": str(e),
            "total_results": 0,
            "documents": []
        }

async def main():
    comparison_dir = Path("comparison")
    
    # Load workspace mapping from training data
    workspace_map = {}
    try:
        with open("training_data_step_0.json", 'r') as f:
            training_data = json.load(f)
        
        # Create mapping from query_id to workspace
        for item in training_data:
            query_id = item['query_id']
            # Extract workspace from document_id prefix (e.g., "Manifest_C06QS2DRBNJ_..." -> "Manifest")
            workspace = item['document_id'].split('_')[0]
            workspace_map[query_id] = workspace
    except Exception as e:
        print(f"Error loading workspace mapping: {e}")
        workspace_map = {}
    
    print(f"Loaded workspace mapping for {len(workspace_map)} queries")
    
    processed = 0
    errors = 0
    
    # Walk through all comparison directories
    for query_dir in comparison_dir.iterdir():
        if not query_dir.is_dir() or query_dir.name.startswith('.'):
            continue
            
        query_id = query_dir.name
        workspace = workspace_map.get(query_id, "training-slack")
        
        # Check each search query subdirectory
        for search_dir in query_dir.iterdir():
            if not search_dir.is_dir():
                continue
                
            # Force re-run for all queries
            solr_improved_file = search_dir / "solr_with_freqs.json"
            
            try:
                # Get the search query from directory name
                search_query = search_dir.name.replace('_', ' ')
                if workspace != "Manifest":
                    continue
                
                # Run improved Solr search
                print(f"Re-running improved Solr for: {search_query} (workspace: {workspace})")
                improved_result = await search_solr_improved(search_query, workspace)
                
                # Save improved Solr result
                with open(solr_improved_file, 'w') as f:
                    json.dump(improved_result, f, indent=2)
                
                processed += 1
                
                if processed % 100 == 0:
                    print(f"Processed {processed} queries...")
                    
            except Exception as e:
                print(f"Error processing {search_dir}: {e}")
                errors += 1
    
    print(f"\nCompleted re-running all improved Solr queries")
    print(f"Total processed: {processed}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    asyncio.run(main())
