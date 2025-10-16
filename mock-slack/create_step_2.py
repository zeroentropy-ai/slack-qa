#!/usr/bin/env python3
"""
Create step_2.json by adding Slack search ranks from existing trace data
"""
import json
from tqdm import tqdm

def extract_slack_ranks_from_agent_traces():
    """Extract Slack search results from automated_agent_traces.json"""
    with open('../automated_agent_traces.json', 'r') as f:
        agent_traces = json.load(f)
    
    slack_ranks = {}  # (query_id, search_query) -> rank
    
    for query_id, trace in agent_traces.items():
        for step in trace['steps']:
            search_calls = step['search_calls']
            individual_ranks = step.get('individual_query_ranks', [])
            
            # Match search calls with their ranks
            for i, search_query in enumerate(search_calls):
                if i < len(individual_ranks):
                    rank = individual_ranks[i]
                    # Convert 0 to None for consistency
                    slack_rank = rank if rank > 0 else None
                    slack_ranks[(query_id, search_query)] = slack_rank
    
    return slack_ranks

def extract_slack_ranks_from_few_shot():
    """Extract Slack search results from few_shot_evaluation_results.json"""
    with open('../few_shot_evaluation_results.json', 'r') as f:
        few_shot_results = json.load(f)
    
    slack_ranks = {}  # (query_id, search_query) -> rank
    
    for result in few_shot_results:
        query_id = result['query_id']
        search_queries = result['search_queries']
        individual_ranks = result.get('individual_query_ranks', [])
        
        # Match search queries with their ranks
        for i, search_query in enumerate(search_queries):
            if i < len(individual_ranks):
                rank = individual_ranks[i]
                # Convert 0 to None for consistency
                slack_rank = rank if rank > 0 else None
                slack_ranks[(query_id, search_query)] = slack_rank
    
    return slack_ranks

def main():
    # Load step 1 data
    with open('step_1.json', 'r') as f:
        step_1_data = json.load(f)
    
    print(f'Loading step_1.json with {len(step_1_data)} entries...')
    
    # Extract Slack ranks from both sources
    print('Extracting Slack ranks from agent traces...')
    agent_slack_ranks = extract_slack_ranks_from_agent_traces()
    
    print('Extracting Slack ranks from few-shot evaluation...')
    few_shot_slack_ranks = extract_slack_ranks_from_few_shot()
    
    # Combine both sources (few-shot takes priority if both exist)
    all_slack_ranks = {**agent_slack_ranks, **few_shot_slack_ranks}
    
    print(f'Found {len(all_slack_ranks)} unique (query_id, search_query) combinations with Slack ranks')
    
    # Add Slack ranks to step 1 data
    step_2_data = []
    found_count = 0
    
    for entry in tqdm(step_1_data, desc="Adding Slack ranks"):
        # Copy the entry
        new_entry = entry.copy()
        
        # Look up Slack rank
        key = (entry['query_id'], entry['search_query'])
        if key in all_slack_ranks:
            new_entry['slack_target_rank'] = all_slack_ranks[key]
            found_count += 1
        else:
            new_entry['slack_target_rank'] = None
        
        step_2_data.append(new_entry)
    
    # Save step 2
    with open('step_2.json', 'w') as f:
        json.dump(step_2_data, f, indent=2)
    
    print(f'Saved {len(step_2_data)} entries to step_2.json')
    print(f'Found Slack ranks for {found_count}/{len(step_2_data)} entries ({found_count/len(step_2_data)*100:.1f}%)')
    
    # Show stats
    solr_found = sum(1 for r in step_2_data if r['target_rank'] is not None)
    slack_found = sum(1 for r in step_2_data if r['slack_target_rank'] is not None)
    both_found = sum(1 for r in step_2_data if r['target_rank'] is not None and r['slack_target_rank'] is not None)
    
    print(f'\nStats:')
    print(f'Solr found: {solr_found}/{len(step_2_data)} ({solr_found/len(step_2_data)*100:.1f}%)')
    print(f'Slack found: {slack_found}/{len(step_2_data)} ({slack_found/len(step_2_data)*100:.1f}%)')
    print(f'Both found: {both_found}/{len(step_2_data)} ({both_found/len(step_2_data)*100:.1f}%)')

if __name__ == "__main__":
    main()