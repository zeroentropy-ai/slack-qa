#!/usr/bin/env python3
"""
Generate agreement matrix and agreement analysis for Slack vs Masked Solr
"""
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load step_3.json
    with open('step_3.json', 'r') as f:
        data = json.load(f)
    
    print(f'Analyzing {len(data)} total queries...\n')
    
    # Categorize results for Recall@100
    slack_hit = []
    masked_hit = []
    
    for entry in data:
        slack_rank = entry.get('slack_target_rank')
        masked_rank = entry.get('masked_target_rank')
        
        # Check if found within top 100 (since we fetch 100 results)
        slack_found = slack_rank is not None and slack_rank <= 100
        masked_found = masked_rank is not None and masked_rank <= 100
        
        slack_hit.append(slack_found)
        masked_hit.append(masked_found)
    
    # Create agreement matrix
    both_found = sum(1 for s, m in zip(slack_hit, masked_hit) if s and m)
    slack_only = sum(1 for s, m in zip(slack_hit, masked_hit) if s and not m)
    masked_only = sum(1 for s, m in zip(slack_hit, masked_hit) if not s and m)
    neither_found = sum(1 for s, m in zip(slack_hit, masked_hit) if not s and not m)
    
    total = len(data)
    
    print("AGREEMENT MATRIX - Recall@100")
    print("=" * 50)
    print(f"{'':>15} {'Masked Hit':>12} {'Masked Miss':>12} {'Total':>10}")
    print(f"{'Slack Hit':<15} {both_found:>12} {slack_only:>12} {sum(slack_hit):>10}")
    print(f"{'Slack Miss':<15} {masked_only:>12} {neither_found:>12} {total - sum(slack_hit):>10}")
    print(f"{'Total':<15} {sum(masked_hit):>12} {total - sum(masked_hit):>12} {total:>10}")
    
    print(f"\nDETAILED BREAKDOWN:")
    print(f"Both found target:     {both_found:4d} ({both_found/total*100:.1f}%)")
    print(f"Only Slack found:      {slack_only:4d} ({slack_only/total*100:.1f}%)")
    print(f"Only Masked found:     {masked_only:4d} ({masked_only/total*100:.1f}%)")
    print(f"Neither found:         {neither_found:4d} ({neither_found/total*100:.1f}%)")
    print(f"At least one found:    {both_found + slack_only + masked_only:4d} ({(both_found + slack_only + masked_only)/total*100:.1f}%)")
    
    # Agreement analysis
    agreements = both_found + neither_found
    disagreements = slack_only + masked_only
    
    print(f"\nAGREEMENT ANALYSIS:")
    print(f"Agreements:            {agreements:4d} ({agreements/total*100:.1f}%)")
    print(f"Disagreements:         {disagreements:4d} ({disagreements/total*100:.1f}%)")
    
    # Individual recall rates
    slack_recall = sum(slack_hit) / total * 100
    masked_recall = sum(masked_hit) / total * 100
    
    print(f"\nINDIVIDUAL RECALL@100:")
    print(f"Slack recall:          {sum(slack_hit):4d}/{total} ({slack_recall:.1f}%)")
    print(f"Masked recall:         {sum(masked_hit):4d}/{total} ({masked_recall:.1f}%)")
    
    # When both found, compare ranks
    if both_found > 0:
        rank_comparisons = []
        for entry in data:
            slack_rank = entry.get('slack_target_rank')
            masked_rank = entry.get('masked_target_rank')
            
            if (slack_rank is not None and slack_rank <= 100 and 
                masked_rank is not None and masked_rank <= 100):
                rank_comparisons.append((slack_rank, masked_rank))
        
        slack_better = sum(1 for s, m in rank_comparisons if s < m)
        masked_better = sum(1 for s, m in rank_comparisons if s > m)
        tied = sum(1 for s, m in rank_comparisons if s == m)
        
        print(f"\nWHEN BOTH FOUND TARGET ({both_found} queries):")
        print(f"Slack ranked higher:   {slack_better:4d} ({slack_better/both_found*100:.1f}%)")
        print(f"Masked ranked higher:  {masked_better:4d} ({masked_better/both_found*100:.1f}%)")
        print(f"Tied ranks:            {tied:4d} ({tied/both_found*100:.1f}%)")
    
    # Create visual agreement matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    
    agreement_data = np.array([[both_found, slack_only], 
                              [masked_only, neither_found]])
    
    im = ax.imshow(agreement_data, interpolation='nearest', cmap='Blues')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{agreement_data[i, j]}\n({agreement_data[i, j]/total*100:.1f}%)',
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Masked Hit', 'Masked Miss'])
    ax.set_yticklabels(['Slack Hit', 'Slack Miss'])
    ax.set_xlabel('Masked Solr Results')
    ax.set_ylabel('Slack Results')
    ax.set_title('Agreement Matrix: Slack vs Masked Solr\n(Recall@100)')
    
    plt.tight_layout()
    plt.savefig('agreement_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved agreement matrix visualization to agreement_matrix.png")

if __name__ == "__main__":
    main()