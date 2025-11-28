"""
Example script showing how to use the baseline system programmatically.
This demonstrates the full workflow from data loading to evaluation.
"""

import json
import os
from baseline import create_prompt, get_plausibility_rating, process_dataset
from scoring import calculate_metrics
import numpy as np

def example_single_prediction():
    """Example: Get prediction for a single sample."""
    
    print("="*70)
    print("Example 1: Single Sample Prediction")
    print("="*70)
    
    # Load one sample from dev.json
    with open('dev.json', 'r') as f:
        data = json.load(f)
    
    # Get the first sample
    sample_id, sample = list(data.items())[0]
    
    print(f"\nSample ID: {sample_id}")
    print(f"Homonym: {sample['homonym']}")
    print(f"Meaning: {sample['judged_meaning']}")
    print(f"\nStory:")
    print(f"  {sample['precontext']}")
    print(f"  {sample['sentence']}")
    if sample.get('ending'):
        print(f"  {sample['ending']}")
    
    print(f"\nHuman ratings: {sample['choices']}")
    print(f"Human average: {sample['average']:.2f} ± {sample['stdev']:.2f}")
    
    # Get model prediction
    if os.environ.get("OPENAI_API_KEY"):
        print("\nGetting model prediction...")
        prediction = get_plausibility_rating(sample, model="gpt-4o", temperature=0.3)
        print(f"Model prediction: {prediction}")
        
        # Compare to human average
        error = abs(prediction - sample['average'])
        within_stdev = error <= max(sample['stdev'], 1.0)
        print(f"Error: {error:.2f}")
        print(f"Within stdev: {'✅ Yes' if within_stdev else '❌ No'}")
    else:
        print("\n⚠️  OPENAI_API_KEY not set, skipping prediction")

def example_batch_prediction():
    """Example: Process multiple samples."""
    
    print("\n" + "="*70)
    print("Example 2: Batch Prediction (10 samples)")
    print("="*70)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set, skipping this example")
        return
    
    # Process 10 samples
    predictions = process_dataset(
        data_path='dev.json',
        output_path='input/res/example_predictions.jsonl',
        model='gpt-4o',
        temperature=0.3,
        max_samples=10
    )
    
    print(f"\n✅ Generated {len(predictions)} predictions")
    print(f"Saved to: input/res/example_predictions.jsonl")

def example_evaluation():
    """Example: Evaluate predictions."""
    
    print("\n" + "="*70)
    print("Example 3: Evaluation")
    print("="*70)
    
    # Check if predictions exist
    if not os.path.exists('input/res/example_predictions.jsonl'):
        print("\n⚠️  No predictions found. Run example_batch_prediction first.")
        return
    
    # Load gold data
    with open('dev.json', 'r') as f:
        gold_data = json.load(f)
    
    # Load predictions
    predictions = {}
    with open('input/res/example_predictions.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            predictions[item['id']] = item['prediction']
    
    # Calculate metrics
    results = calculate_metrics(gold_data, predictions)
    
    if results:
        print(f"\nEvaluation Results:")
        print(f"  Spearman Correlation: {results['spearman_correlation']:.4f}")
        print(f"  Accuracy (within stdev): {results['accuracy_within_stdev']:.4f}")
        print(f"  Mean Absolute Error: {results['mean_absolute_error']:.4f}")
        print(f"  Samples evaluated: {results['num_samples']}")

def example_data_analysis():
    """Example: Analyze dataset characteristics."""
    
    print("\n" + "="*70)
    print("Example 4: Data Analysis")
    print("="*70)
    
    # Load data
    with open('dev.json', 'r') as f:
        data = json.load(f)
    
    # Basic statistics
    num_samples = len(data)
    homonyms = [item['homonym'] for item in data.values()]
    unique_homonyms = len(set(homonyms))
    
    averages = [item['average'] for item in data.values()]
    stdevs = [item['stdev'] for item in data.values()]
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {num_samples}")
    print(f"  Unique homonyms: {unique_homonyms}")
    print(f"  Average rating: {np.mean(averages):.2f} ± {np.std(averages):.2f}")
    print(f"  Average stdev: {np.mean(stdevs):.2f}")
    
    # Agreement analysis
    high_agreement = sum(1 for s in stdevs if s < 0.5)
    low_agreement = sum(1 for s in stdevs if s > 1.5)
    
    print(f"\nAgreement Levels:")
    print(f"  High agreement (stdev < 0.5): {high_agreement} ({100*high_agreement/num_samples:.1f}%)")
    print(f"  Low agreement (stdev > 1.5): {low_agreement} ({100*low_agreement/num_samples:.1f}%)")
    
    # Show an interesting example
    print(f"\nExample of high disagreement:")
    high_disagreement = [(sid, s) for sid, s in data.items() if s['stdev'] > 1.5]
    if high_disagreement:
        sample_id, sample = high_disagreement[0]
        print(f"  ID: {sample_id}")
        print(f"  Homonym: {sample['homonym']}")
        print(f"  Sentence: {sample['sentence']}")
        print(f"  Ratings: {sample['choices']} (stdev: {sample['stdev']:.2f})")

def example_prompt_inspection():
    """Example: Inspect the prompt that will be sent to the model."""
    
    print("\n" + "="*70)
    print("Example 5: Prompt Inspection")
    print("="*70)
    
    # Load a sample
    with open('dev.json', 'r') as f:
        data = json.load(f)
    
    sample_id, sample = list(data.items())[0]
    
    # Create prompt
    prompt = create_prompt(sample)
    
    print(f"\nPrompt for sample {sample_id}:")
    print("-" * 70)
    print(prompt)
    print("-" * 70)
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Estimated tokens: ~{len(prompt.split())} words")

def main():
    """Run all examples."""
    
    print("\n" + "="*70)
    print("AmbiStory WSD Baseline - Example Usage")
    print("="*70)
    print("\nThis script demonstrates how to use the baseline system.")
    print("You can run each example individually or all together.")
    print()
    
    # Check for API key
    if os.environ.get("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY is set")
    else:
        print("⚠️  OPENAI_API_KEY is not set")
        print("   Some examples will be skipped.")
        print("   Set it with: export OPENAI_API_KEY='your-key'")
    print()
    
    # Run examples
    example_data_analysis()
    example_prompt_inspection()
    example_single_prediction()
    example_batch_prediction()
    example_evaluation()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the generated predictions: input/res/example_predictions.jsonl")
    print("  2. Try modifying the prompts in baseline.py")
    print("  3. Run the full baseline: python baseline.py --data dev.json")
    print("  4. Evaluate your predictions: python scoring.py dev.json predictions.jsonl scores.json")
    print()

if __name__ == "__main__":
    main()

