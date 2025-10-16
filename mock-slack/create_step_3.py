#!/usr/bin/env python3
"""
Create step_3.json by adding masked Solr ranks to step_2.json data
"""
import json
import time
from tqdm import tqdm
from masked_solr_library import batch_masked_search_with_ranks

def main():
    # Load step 2
    with open('step_2.json', 'r') as f:
        step_2_data = json.load(f)

    print(f'Processing {len(step_2_data)} search queries with masked Solr...')
    start_time = time.time()

    # Process in batches for progress tracking
    batch_size = 50
    all_results = []
    batches = [step_2_data[i:i+batch_size] for i in range(0, len(step_2_data), batch_size)]

    for batch_num, batch in enumerate(tqdm(batches, desc="Processing batches"), 1):
        batch_results = batch_masked_search_with_ranks(batch)
        all_results.extend(batch_results)
        
        # Save progress every 10 batches
        if batch_num % 10 == 0:
            with open('step_3_progress.json', 'w') as f:
                json.dump(all_results, f, indent=2)

    # Save final results
    with open('step_3.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - start_time
    print(f'\nCompleted in {elapsed:.1f}s! Saved {len(all_results)} results to step_3.json')

    # Show stats comparing all three search methods
    solr_found = sum(1 for r in all_results if r['target_rank'] is not None)
    slack_found = sum(1 for r in all_results if r['slack_target_rank'] is not None)
    masked_found = sum(1 for r in all_results if r['masked_target_rank'] is not None)
    
    solr_recall20 = sum(1 for r in all_results if r['target_rank'] is not None and r['target_rank'] <= 20)
    slack_recall20 = sum(1 for r in all_results if r['slack_target_rank'] is not None and r['slack_target_rank'] <= 20)
    masked_recall20 = sum(1 for r in all_results if r['masked_target_rank'] is not None and r['masked_target_rank'] <= 20)

    total = len(all_results)
    
    print(f'\nComparison of search methods:')
    print(f'{"Method":<12} {"Found":<15} {"Recall@20":<15}')
    print('-' * 45)
    print(f'{"Solr":<12} {solr_found}/{total} ({solr_found/total*100:.1f}%){"":<2} {solr_recall20}/{total} ({solr_recall20/total*100:.1f}%)')
    print(f'{"Slack":<12} {slack_found}/{total} ({slack_found/total*100:.1f}%){"":<2} {slack_recall20}/{total} ({slack_recall20/total*100:.1f}%)')
    print(f'{"Masked":<12} {masked_found}/{total} ({masked_found/total*100:.1f}%){"":<2} {masked_recall20}/{total} ({masked_recall20/total*100:.1f}%)')
    
    # Show keyword matching distribution for masked search
    keyword_counts = {}
    for r in all_results:
        if r['masked_target_rank'] is not None:
            k = r['keywords_matched']
            keyword_counts[k] = keyword_counts.get(k, 0) + 1
    
    if keyword_counts:
        print(f'\nMasked search keyword matching distribution:')
        for k in sorted(keyword_counts.keys()):
            print(f'  {k} keywords: {keyword_counts[k]} queries')

if __name__ == "__main__":
    main()