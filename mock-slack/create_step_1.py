#!/usr/bin/env python3
"""
Create step_1.json by processing all queries through Solr
"""
import json
import time
from tqdm import tqdm
from search_library import batch_search_with_ranks

def main():
    # Load step 0
    with open('step_0.json', 'r') as f:
        queries = json.load(f)

    print(f'Processing {len(queries)} search queries with Solr...')
    start_time = time.time()

    # Process individually with progress bar
    all_results = []
    
    for i, query in enumerate(tqdm(queries, desc="Processing queries")):
        result = batch_search_with_ranks([query])[0]
        all_results.append(result)
        
        # Save progress every 100 queries
        if (i + 1) % 100 == 0:
            with open('step_1_progress.json', 'w') as f:
                json.dump(all_results, f, indent=2)

    # Save final results
    with open('step_1.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - start_time
    print(f'\nCompleted in {elapsed:.1f}s! Saved {len(all_results)} results to step_1.json')

    # Show final stats
    found_count = sum(1 for r in all_results if r['target_rank'] is not None)
    recall_20_count = sum(1 for r in all_results if r['target_rank'] is not None and r['target_rank'] <= 20)

    print(f'Final Solr results: {found_count}/{len(all_results)} found ({found_count/len(all_results)*100:.1f}%)')
    print(f'Final Recall@20: {recall_20_count}/{len(all_results)} ({recall_20_count/len(all_results)*100:.1f}%)')

if __name__ == "__main__":
    main()