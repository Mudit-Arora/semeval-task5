"""
Data analysis script for AmbiStory dataset.
Provides statistics and insights about the data.
"""

import json
import numpy as np
from collections import Counter, defaultdict
import argparse

def load_data(file_path):
    """Load JSON data."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_dataset(data, name="Dataset"):
    """Perform comprehensive analysis of the dataset."""
    
    print(f"\n{'='*70}")
    print(f"{name} Analysis")
    print(f"{'='*70}\n")
    
    # Basic statistics
    num_samples = len(data)
    print(f"Total samples: {num_samples}")
    
    # Group by sample_id to find unique stories
    sample_ids = [item['sample_id'] for item in data.values()]
    unique_stories = len(set(sample_ids))
    print(f"Unique story setups: {unique_stories}")
    print(f"Average samples per story: {num_samples / unique_stories:.1f}")
    
    # Homonym analysis
    homonyms = [item['homonym'] for item in data.values()]
    homonym_counts = Counter(homonyms)
    print(f"\nUnique homonyms: {len(homonym_counts)}")
    print(f"Most common homonyms:")
    for word, count in homonym_counts.most_common(10):
        print(f"  {word}: {count} samples")
    
    # Rating statistics
    averages = [item['average'] for item in data.values()]
    stdevs = [item['stdev'] for item in data.values()]
    
    print(f"\nRating Statistics:")
    print(f"  Average rating:")
    print(f"    Mean: {np.mean(averages):.3f}")
    print(f"    Median: {np.median(averages):.3f}")
    print(f"    Std Dev: {np.std(averages):.3f}")
    print(f"    Min: {np.min(averages):.3f}")
    print(f"    Max: {np.max(averages):.3f}")
    
    print(f"\n  Standard deviation across annotators:")
    print(f"    Mean: {np.mean(stdevs):.3f}")
    print(f"    Median: {np.median(stdevs):.3f}")
    print(f"    Min: {np.min(stdevs):.3f}")
    print(f"    Max: {np.max(stdevs):.3f}")
    
    # Agreement analysis
    low_agreement = sum(1 for s in stdevs if s > 1.5)
    high_agreement = sum(1 for s in stdevs if s < 0.5)
    print(f"\n  Agreement levels:")
    print(f"    High agreement (stdev < 0.5): {high_agreement} ({100*high_agreement/num_samples:.1f}%)")
    print(f"    Low agreement (stdev > 1.5): {low_agreement} ({100*low_agreement/num_samples:.1f}%)")
    
    # Distribution of average ratings
    print(f"\n  Distribution of average ratings:")
    for rating in range(1, 6):
        count = sum(1 for a in averages if rating - 0.5 <= a < rating + 0.5)
        percentage = 100 * count / num_samples
        bar = '█' * int(percentage / 2)
        print(f"    {rating}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    # Ending analysis
    has_ending = sum(1 for item in data.values() if item.get('ending', ''))
    no_ending = num_samples - has_ending
    print(f"\n  Story structure:")
    print(f"    With ending: {has_ending} ({100*has_ending/num_samples:.1f}%)")
    print(f"    No ending: {no_ending} ({100*no_ending/num_samples:.1f}%)")
    
    # Word sense analysis
    meanings = defaultdict(list)
    for item in data.values():
        homonym = item['homonym']
        meaning = item['judged_meaning']
        meanings[homonym].append(meaning)
    
    print(f"\n  Word senses:")
    total_sense_pairs = 0
    for homonym, sense_list in meanings.items():
        unique_senses = len(set(sense_list))
        total_sense_pairs += unique_senses
    print(f"    Average senses per homonym: {total_sense_pairs / len(meanings):.1f}")
    
    # Nonsensical annotations
    nonsensical_counts = []
    for item in data.values():
        if 'nonsensical' in item:
            nonsensical_counts.append(sum(item['nonsensical']))
    
    if nonsensical_counts:
        total_nonsensical = sum(nonsensical_counts)
        print(f"\n  Nonsensical annotations:")
        print(f"    Total: {total_nonsensical}")
        print(f"    Samples with nonsensical ratings: {sum(1 for c in nonsensical_counts if c > 0)}")

def compare_datasets(train_data, dev_data):
    """Compare train and dev datasets."""
    
    print(f"\n{'='*70}")
    print("Dataset Comparison")
    print(f"{'='*70}\n")
    
    # Homonym overlap
    train_homonyms = set(item['homonym'] for item in train_data.values())
    dev_homonyms = set(item['homonym'] for item in dev_data.values())
    
    overlap = train_homonyms & dev_homonyms
    train_only = train_homonyms - dev_homonyms
    dev_only = dev_homonyms - train_homonyms
    
    print(f"Homonym overlap:")
    print(f"  Shared between train and dev: {len(overlap)}")
    print(f"  Only in train: {len(train_only)}")
    print(f"  Only in dev: {len(dev_only)}")
    
    if train_only:
        print(f"\n  Examples only in train: {list(train_only)[:5]}")
    if dev_only:
        print(f"  Examples only in dev: {list(dev_only)[:5]}")
    
    # Rating distribution comparison
    train_avgs = [item['average'] for item in train_data.values()]
    dev_avgs = [item['average'] for item in dev_data.values()]
    
    print(f"\nRating distributions:")
    print(f"  Train mean: {np.mean(train_avgs):.3f} ± {np.std(train_avgs):.3f}")
    print(f"  Dev mean:   {np.mean(dev_avgs):.3f} ± {np.std(dev_avgs):.3f}")
    
    print(f"\nAgreement (stdev) comparison:")
    train_stdevs = [item['stdev'] for item in train_data.values()]
    dev_stdevs = [item['stdev'] for item in dev_data.values()]
    print(f"  Train mean stdev: {np.mean(train_stdevs):.3f}")
    print(f"  Dev mean stdev:   {np.mean(dev_stdevs):.3f}")

def show_examples(data, n=3):
    """Show example samples from the dataset."""
    
    print(f"\n{'='*70}")
    print(f"Example Samples")
    print(f"{'='*70}\n")
    
    samples = list(data.items())[:n]
    
    for idx, (sample_id, sample) in enumerate(samples, 1):
        print(f"Example {idx} (ID: {sample_id}):")
        print(f"  Homonym: {sample['homonym']}")
        print(f"  Judged meaning: {sample['judged_meaning']}")
        print(f"\n  Precontext:")
        for line in sample['precontext'].split('. '):
            if line:
                print(f"    {line.strip()}.")
        print(f"\n  Ambiguous sentence: {sample['sentence']}")
        if sample.get('ending'):
            print(f"  Ending: {sample['ending']}")
        else:
            print(f"  Ending: [None]")
        print(f"\n  Example sentence: {sample['example_sentence']}")
        print(f"  Human ratings: {sample['choices']}")
        print(f"  Average: {sample['average']:.2f} ± {sample['stdev']:.2f}")
        print(f"  Sample ID: {sample['sample_id']}")
        print(f"\n{'-'*70}\n")

def find_challenging_samples(data, n=5):
    """Find samples that are challenging (high std dev or extreme ratings)."""
    
    print(f"\n{'='*70}")
    print("Challenging Samples")
    print(f"{'='*70}\n")
    
    # Sort by standard deviation (high disagreement)
    items = list(data.items())
    items_sorted = sorted(items, key=lambda x: x[1]['stdev'], reverse=True)
    
    print(f"Top {n} samples with highest disagreement (high stdev):")
    for idx, (sample_id, sample) in enumerate(items_sorted[:n], 1):
        print(f"\n{idx}. ID: {sample_id}")
        print(f"   Homonym: {sample['homonym']}")
        print(f"   Meaning: {sample['judged_meaning'][:60]}...")
        print(f"   Sentence: {sample['sentence']}")
        print(f"   Ratings: {sample['choices']} (avg: {sample['average']:.2f}, stdev: {sample['stdev']:.2f})")
    
    # Samples with extreme ratings
    extreme_low = [(sid, s) for sid, s in items if s['average'] < 1.5]
    extreme_high = [(sid, s) for sid, s in items if s['average'] > 4.5]
    
    print(f"\n\nSamples with extreme ratings:")
    print(f"  Very low plausibility (avg < 1.5): {len(extreme_low)}")
    print(f"  Very high plausibility (avg > 4.5): {len(extreme_high)}")

def main():
    parser = argparse.ArgumentParser(description='Analyze AmbiStory dataset')
    parser.add_argument('--train', type=str, default='train.json',
                        help='Path to training data')
    parser.add_argument('--dev', type=str, default='dev.json',
                        help='Path to development data')
    parser.add_argument('--examples', type=int, default=3,
                        help='Number of examples to show')
    parser.add_argument('--show-challenging', action='store_true',
                        help='Show challenging samples with high disagreement')
    parser.add_argument('--compare', action='store_true',
                        help='Compare train and dev datasets')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    train_data = load_data(args.train)
    dev_data = load_data(args.dev)
    
    # Analyze individual datasets
    analyze_dataset(train_data, name="Training Set")
    analyze_dataset(dev_data, name="Development Set")
    
    # Compare datasets
    if args.compare:
        compare_datasets(train_data, dev_data)
    
    # Show examples
    show_examples(dev_data, n=args.examples)
    
    # Show challenging samples
    if args.show_challenging:
        find_challenging_samples(dev_data, n=5)

if __name__ == "__main__":
    main()

