"""
Visualization and analysis of baseline model predictions.
"""

import json
import numpy as np
import argparse
from collections import defaultdict

def load_jsonl(file_path):
    """Load JSONL predictions."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data[item['id']] = item['prediction']
    return data

def load_gold_json(file_path):
    """Load gold standard JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_predictions(gold_data, predictions):
    """Analyze prediction patterns and errors."""
    
    print(f"\n{'='*70}")
    print("Prediction Analysis")
    print(f"{'='*70}\n")
    
    # Collect data
    errors = []
    correct_within_stdev = []
    by_rating = defaultdict(list)
    by_stdev = defaultdict(list)
    by_homonym = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': []})
    
    for sample_id, gold_sample in gold_data.items():
        if sample_id not in predictions:
            continue
        
        pred = predictions[sample_id]
        gold_avg = gold_sample['average']
        gold_std = max(gold_sample['stdev'], 1.0)
        
        error = abs(pred - gold_avg)
        errors.append(error)
        
        # Check if within stdev
        within_stdev = error <= gold_std
        correct_within_stdev.append(within_stdev)
        
        # Group by gold rating
        gold_rounded = round(gold_avg)
        by_rating[gold_rounded].append(error)
        
        # Group by stdev (agreement level)
        if gold_std < 0.5:
            stdev_category = 'high_agreement'
        elif gold_std < 1.0:
            stdev_category = 'medium_agreement'
        elif gold_std < 1.5:
            stdev_category = 'low_agreement'
        else:
            stdev_category = 'very_low_agreement'
        by_stdev[stdev_category].append(error)
        
        # Group by homonym
        homonym = gold_sample['homonym']
        by_homonym[homonym]['total'] += 1
        if within_stdev:
            by_homonym[homonym]['correct'] += 1
        else:
            by_homonym[homonym]['errors'].append(error)
    
    # Overall statistics
    print(f"Overall Performance:")
    print(f"  Total predictions: {len(errors)}")
    print(f"  Mean absolute error: {np.mean(errors):.3f}")
    print(f"  Median absolute error: {np.median(errors):.3f}")
    print(f"  Max error: {np.max(errors):.3f}")
    print(f"  Accuracy within stdev: {np.mean(correct_within_stdev):.3f}")
    
    # Distribution of errors
    print(f"\nError Distribution:")
    for threshold in [0.5, 1.0, 1.5, 2.0]:
        within_threshold = sum(1 for e in errors if e <= threshold)
        percentage = 100 * within_threshold / len(errors)
        print(f"  Within {threshold}: {within_threshold}/{len(errors)} ({percentage:.1f}%)")
    
    # Performance by gold rating
    print(f"\nPerformance by Gold Rating:")
    for rating in sorted(by_rating.keys()):
        rating_errors = by_rating[rating]
        mae = np.mean(rating_errors)
        count = len(rating_errors)
        print(f"  Rating {rating}: MAE = {mae:.3f} (n={count})")
    
    # Performance by agreement level
    print(f"\nPerformance by Agreement Level:")
    agreement_order = ['high_agreement', 'medium_agreement', 'low_agreement', 'very_low_agreement']
    for category in agreement_order:
        if category in by_stdev:
            cat_errors = by_stdev[category]
            mae = np.mean(cat_errors)
            count = len(cat_errors)
            accuracy = sum(1 for e in cat_errors if e <= 1.0) / count
            print(f"  {category.replace('_', ' ').title()}: MAE = {mae:.3f}, Acc = {accuracy:.3f} (n={count})")
    
    # Performance by homonym (top/bottom)
    print(f"\nBest Performing Homonyms (accuracy within stdev):")
    homonym_performance = [(h, d['correct'] / d['total']) for h, d in by_homonym.items() if d['total'] >= 3]
    homonym_performance.sort(key=lambda x: x[1], reverse=True)
    
    for homonym, accuracy in homonym_performance[:5]:
        count = by_homonym[homonym]['total']
        print(f"  {homonym}: {accuracy:.3f} ({by_homonym[homonym]['correct']}/{count})")
    
    print(f"\nWorst Performing Homonyms (accuracy within stdev):")
    for homonym, accuracy in homonym_performance[-5:]:
        count = by_homonym[homonym]['total']
        print(f"  {homonym}: {accuracy:.3f} ({by_homonym[homonym]['correct']}/{count})")

def show_error_examples(gold_data, predictions, n=5):
    """Show examples of large errors."""
    
    print(f"\n{'='*70}")
    print("Examples of Large Errors")
    print(f"{'='*70}\n")
    
    # Calculate errors and sort
    error_samples = []
    for sample_id, gold_sample in gold_data.items():
        if sample_id not in predictions:
            continue
        
        pred = predictions[sample_id]
        gold_avg = gold_sample['average']
        error = abs(pred - gold_avg)
        
        error_samples.append((error, sample_id, gold_sample, pred))
    
    error_samples.sort(reverse=True)
    
    print(f"Top {n} largest errors:\n")
    
    for idx, (error, sample_id, sample, pred) in enumerate(error_samples[:n], 1):
        print(f"{idx}. Error: {error:.2f} (Predicted: {pred}, Gold avg: {sample['average']:.2f})")
        print(f"   ID: {sample_id}")
        print(f"   Homonym: {sample['homonym']}")
        print(f"   Meaning: {sample['judged_meaning'][:70]}...")
        print(f"   Sentence: {sample['sentence']}")
        if sample.get('ending'):
            print(f"   Ending: {sample['ending'][:70]}...")
        print(f"   Human ratings: {sample['choices']} (stdev: {sample['stdev']:.2f})")
        print()

def create_confusion_matrix(gold_data, predictions):
    """Create a confusion matrix of predicted vs gold ratings."""
    
    print(f"\n{'='*70}")
    print("Confusion Matrix (Predicted vs Gold Rounded)")
    print(f"{'='*70}\n")
    
    # Create matrix
    matrix = np.zeros((5, 5), dtype=int)
    
    for sample_id, gold_sample in gold_data.items():
        if sample_id not in predictions:
            continue
        
        pred = predictions[sample_id]
        gold_rounded = round(gold_sample['average'])
        
        # Ensure values are in valid range
        if 1 <= pred <= 5 and 1 <= gold_rounded <= 5:
            matrix[gold_rounded - 1][pred - 1] += 1
    
    # Print matrix
    print("       Predicted")
    print("       1    2    3    4    5")
    print("    " + "-" * 30)
    for i, row in enumerate(matrix):
        print(f"  {i+1} | {' '.join(f'{val:4d}' for val in row)}")
    print()
    
    # Calculate per-class accuracy
    print("Per-class statistics:")
    for i in range(5):
        total = np.sum(matrix[i])
        correct = matrix[i][i]
        if total > 0:
            accuracy = correct / total
            print(f"  Gold {i+1}: {correct}/{total} correct ({accuracy:.3f})")

def main():
    parser = argparse.ArgumentParser(description='Visualize and analyze predictions')
    parser.add_argument('gold_file', type=str,
                        help='Path to gold standard file')
    parser.add_argument('prediction_file', type=str,
                        help='Path to predictions file')
    parser.add_argument('--show-errors', type=int, default=5,
                        help='Number of error examples to show')
    parser.add_argument('--confusion-matrix', action='store_true',
                        help='Show confusion matrix')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    gold_data = load_gold_json(args.gold_file)
    predictions = load_jsonl(args.prediction_file)
    
    print(f"Loaded {len(gold_data)} gold samples")
    print(f"Loaded {len(predictions)} predictions")
    
    # Analyze predictions
    analyze_predictions(gold_data, predictions)
    
    # Show error examples
    show_error_examples(gold_data, predictions, n=args.show_errors)
    
    # Show confusion matrix
    if args.confusion_matrix:
        create_confusion_matrix(gold_data, predictions)

if __name__ == "__main__":
    main()

