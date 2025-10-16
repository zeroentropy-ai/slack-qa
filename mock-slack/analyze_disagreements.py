#!/usr/bin/env python3
"""
Analyze the 259 disagreement cases between Slack and Masked Solr
"""
import json
from collections import Counter, defaultdict

def analyze_disagreements():
    # Load step_3.json
    with open('step_3.json', 'r') as f:
        data = json.load(f)
    
    print("ANALYZING 259 DISAGREEMENT CASES")
    print("=" * 60)
    
    # Extract disagreement cases
    slack_only = []  # Slack found, Masked didn't
    masked_only = []  # Masked found, Slack didn't
    
    for entry in data:
        slack_rank = entry.get('slack_target_rank')
        masked_rank = entry.get('masked_target_rank')
        
        slack_found = slack_rank is not None and slack_rank <= 100
        masked_found = masked_rank is not None and masked_rank <= 100
        
        if slack_found and not masked_found:
            slack_only.append(entry)
        elif masked_found and not slack_found:
            masked_only.append(entry)
    
    print(f"Slack-only successes: {len(slack_only)}")
    print(f"Masked-only successes: {len(masked_only)}")
    print()
    
    # Analyze Slack-only successes (what Slack does better)
    print("🟢 SLACK-ONLY SUCCESSES (Slack found, Masked missed)")
    print("-" * 50)
    
    # Query length analysis
    slack_only_lengths = [len(entry['search_query'].split()) for entry in slack_only]
    masked_only_lengths = [len(entry['search_query'].split()) for entry in masked_only]
    
    print(f"Query length stats:")
    print(f"  Slack-only avg length: {sum(slack_only_lengths)/len(slack_only_lengths):.1f} words")
    print(f"  Masked-only avg length: {sum(masked_only_lengths)/len(masked_only_lengths):.1f} words")
    
    # Show examples of Slack-only successes
    print(f"\nExamples of queries Slack handles better:")
    slack_only_sorted = sorted(slack_only, key=lambda x: x['slack_target_rank'])
    for i, entry in enumerate(slack_only_sorted[:10]):
        query = entry['search_query']
        rank = entry['slack_target_rank']
        keywords = entry.get('keywords_matched', 0)
        print(f"  {i+1:2d}. \"{query}\" (Slack rank: {rank}, keywords tried: {keywords})")
    
    print(f"\n🔴 MASKED-ONLY SUCCESSES (Masked found, Slack missed)")
    print("-" * 50)
    
    # Show examples of Masked-only successes
    print(f"Examples of queries Masked handles better:")
    masked_only_sorted = sorted(masked_only, key=lambda x: x['masked_target_rank'])
    for i, entry in enumerate(masked_only_sorted[:10]):
        query = entry['search_query']
        rank = entry['masked_target_rank']
        keywords = entry.get('keywords_matched', 0)
        print(f"  {i+1:2d}. \"{query}\" (Masked rank: {rank}, keywords: {keywords})")
    
    # Analyze query patterns
    print(f"\n📊 PATTERN ANALYSIS")
    print("-" * 50)
    
    def analyze_query_patterns(queries, label):
        print(f"\n{label} query patterns:")
        
        # Word frequency
        all_words = []
        for entry in queries:
            words = entry['search_query'].lower().split()
            all_words.extend(words)
        
        common_words = Counter(all_words).most_common(10)
        print(f"  Most common words: {', '.join([f'{word}({count})' for word, count in common_words[:5]])}")
        
        # Query types
        query_types = defaultdict(list)
        for entry in queries:
            query = entry['search_query'].lower()
            if 'ticket' in query:
                query_types['ticket_queries'].append(query)
            elif 'error' in query or 'issue' in query or 'problem' in query:
                query_types['error_queries'].append(query)
            elif 'modal' in query:
                query_types['modal_queries'].append(query)
            elif any(word in query for word in ['app', 'id', 'permission']):
                query_types['app_queries'].append(query)
            else:
                query_types['other_queries'].append(query)
        
        for qtype, qlist in query_types.items():
            if qlist:
                print(f"  {qtype.replace('_', ' ').title()}: {len(qlist)} queries")
    
    analyze_query_patterns(slack_only, "SLACK-ONLY")
    analyze_query_patterns(masked_only, "MASKED-ONLY")
    
    # Keywords matched analysis for masked failures
    print(f"\n🔍 MASKED SEARCH BEHAVIOR ANALYSIS")
    print("-" * 50)
    
    keywords_when_slack_succeeds = [entry.get('keywords_matched', 0) for entry in slack_only]
    keyword_counts = Counter(keywords_when_slack_succeeds)
    
    print(f"When Slack succeeds but Masked fails, keywords attempted:")
    for k in sorted(keyword_counts.keys()):
        if k > 0:
            print(f"  {k} keywords: {keyword_counts[k]} queries")
        else:
            print(f"  No results: {keyword_counts[k]} queries")
    
    # Source analysis
    print(f"\n📍 SOURCE ANALYSIS")
    print("-" * 50)
    
    def analyze_sources(queries, label):
        sources = Counter([entry.get('source', 'unknown') for entry in queries])
        print(f"{label} by source:")
        for source, count in sources.items():
            print(f"  {source}: {count} queries")
    
    analyze_sources(slack_only, "Slack-only")
    analyze_sources(masked_only, "Masked-only")
    
    return slack_only, masked_only

if __name__ == "__main__":
    slack_only, masked_only = analyze_disagreements()