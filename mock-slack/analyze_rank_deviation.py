#!/usr/bin/env python3
"""
Analyze deviation between Slack and Masked Solr ranks
"""
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load step_3.json
    with open('step_3.json', 'r') as f:
        data = json.load(f)
    
    # Extract rank differences where both methods found the target
    deviations = []
    
    for entry in data:
        slack_rank = entry.get('slack_target_rank')
        masked_rank = entry.get('masked_target_rank')
        
        if slack_rank is not None and masked_rank is not None:
            deviation = slack_rank - masked_rank
            deviations.append(deviation)
    
    print(f'Found {len(deviations)} queries where both Slack and Masked Solr found the target')
    
    if not deviations:
        print('No data to plot!')
        return
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Plot histogram
    bins = range(min(deviations) - 1, max(deviations) + 2)
    plt.hist(deviations, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    
    plt.xlabel('Rank Deviation (Slack Rank - Masked Solr Rank)')
    plt.ylabel('Number of Queries')
    plt.title('Distribution of Rank Deviations: Slack vs Masked Solr\n(Negative = Masked Solr ranks higher)')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at 0
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Equal ranks')
    
    # Add statistics text
    mean_dev = np.mean(deviations)
    median_dev = np.median(deviations)
    std_dev = np.std(deviations)
    
    stats_text = f'Stats (n={len(deviations)}):\n'
    stats_text += f'Mean: {mean_dev:.2f}\n'
    stats_text += f'Median: {median_dev:.2f}\n'
    stats_text += f'Std Dev: {std_dev:.2f}\n\n'
    stats_text += f'Masked better: {sum(1 for d in deviations if d > 0)} ({sum(1 for d in deviations if d > 0)/len(deviations)*100:.1f}%)\n'
    stats_text += f'Slack better: {sum(1 for d in deviations if d < 0)} ({sum(1 for d in deviations if d < 0)/len(deviations)*100:.1f}%)\n'
    stats_text += f'Equal: {sum(1 for d in deviations if d == 0)} ({sum(1 for d in deviations if d == 0)/len(deviations)*100:.1f}%)'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('rank_deviation_histogram.png', dpi=300, bbox_inches='tight')
    print('Saved histogram to rank_deviation_histogram.png')
    
    # Show some examples
    print('\nExamples of large deviations:')
    
    # Sort by absolute deviation
    data_with_dev = []
    for i, entry in enumerate(data):
        slack_rank = entry.get('slack_target_rank')
        masked_rank = entry.get('masked_target_rank')
        
        if slack_rank is not None and masked_rank is not None:
            deviation = slack_rank - masked_rank
            data_with_dev.append((abs(deviation), deviation, entry))
    
    data_with_dev.sort(reverse=True)
    
    print('\nLargest absolute deviations:')
    for i, (abs_dev, dev, entry) in enumerate(data_with_dev[:10]):
        better = "Masked" if dev > 0 else "Slack"
        print(f'{i+1:2d}. "{entry["search_query"]}" (Slack: {entry["slack_target_rank"]}, Masked: {entry["masked_target_rank"]}, {better} better by {abs_dev})')

if __name__ == "__main__":
    main()