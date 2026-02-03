"""
Comparison Script: GEPA vs 5-Run Averaging vs Hybrid Approach
Tests all three approaches on a small sample and compares results.
"""

import json
import os
import random
import numpy as np
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from scipy.stats import spearmanr
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ============== CONFIGURATION ==============
MODEL = "gpt-4o-mini"
SAMPLE_SIZE = 50  # Number of test samples
GEPA_BUDGET = 80  # Budget for GEPA optimization
NUM_RUNS_AVERAGING = 5  # Number of runs for averaging approach
NUM_RUNS_HYBRID = 3  # Number of runs for hybrid approach

# ============== BASE PROMPT ==============
BASE_PROMPT = """You are an expert linguist evaluating word sense disambiguation in narrative contexts.

Rate the plausibility of a specific word sense in a short story on a scale from 1 to 5.

**RATING SCALE**:
- 5 (Highly plausible): The word sense fits perfectly with the narrative context.
- 4 (Plausible): The word sense fits well. It's a reasonable interpretation.
- 3 (Somewhat plausible): The word sense could fit, but it's not entirely clear.
- 2 (Less plausible): The word sense doesn't fit well with the context.
- 1 (Implausible): The word sense doesn't fit at all.

Respond with ONLY a single integer from 1 to 5."""

# ============== GEPA PROMPTS ==============
META_PROMPT = """You are an expert at improving LLM prompts for word sense disambiguation tasks.

## Current Prompt:
{candidate}

## Feedback from Evaluation:
{feedback}

## Goal:
Propose an improved prompt that:
1. Keeps what works well
2. Fixes identified weaknesses  
3. Improves accuracy on plausibility ratings
4. Remains concise and clear

Output ONLY the new improved prompt text. Do not include examples or explanations."""


def extract_rating(text: str) -> int:
    """Extract rating (1-5) from model response."""
    text = text.strip()
    if "RATING:" in text.upper():
        parts = text.upper().split("RATING:")
        if len(parts) > 1:
            for char in parts[-1].strip():
                if char.isdigit():
                    rating = int(char)
                    if 1 <= rating <= 5:
                        return rating
    for char in text:
        if char.isdigit():
            rating = int(char)
            if 1 <= rating <= 5:
                return rating
    return 3


def format_sample(sample: Dict[str, Any]) -> str:
    """Format a sample for the model."""
    ending = sample.get('ending', '')
    return f"""**Story Context:**
Precontext: {sample['precontext']}
Sentence: {sample['sentence']}
Ending: {ending if ending else 'N/A'}

**Word Sense to Evaluate:**
Word: "{sample['homonym']}"
Meaning: {sample['judged_meaning']}
Example: {sample['example_sentence']}"""


def get_single_prediction(prompt: str, sample: Dict[str, Any], temperature: float = 0.3) -> int:
    """Get a single prediction from the model."""
    sample_text = format_sample(sample)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": sample_text}
            ],
            temperature=temperature,
            max_tokens=50
        )
        return extract_rating(response.choices[0].message.content)
    except Exception as e:
        print(f"API Error: {e}")
        return 3


def get_averaged_prediction(prompt: str, sample: Dict[str, Any], num_runs: int = 5) -> int:
    """Get prediction by averaging multiple runs."""
    ratings = []
    for _ in range(num_runs):
        rating = get_single_prediction(prompt, sample, temperature=0.7)
        ratings.append(rating)
    avg = round(sum(ratings) / len(ratings))
    return max(1, min(5, avg))


# ============== SCORING FUNCTIONS ==============
def is_within_std(prediction: int, mean: float, std: float) -> bool:
    """Check if prediction is within acceptable range."""
    if (mean - std) < prediction < (mean + std):
        return True
    if abs(mean - prediction) < 1:
        return True
    return False


def calculate_metrics(predictions: List[int], samples: List[Dict]) -> Dict[str, float]:
    """Calculate Spearman correlation and accuracy within std."""
    pred_list = predictions
    gold_list = [s['average'] for s in samples]
    
    # Spearman correlation
    corr, p_value = spearmanr(pred_list, gold_list)
    
    # Accuracy within std
    correct = 0
    for pred, sample in zip(predictions, samples):
        if is_within_std(pred, sample['average'], sample['stdev']):
            correct += 1
    accuracy = correct / len(predictions)
    
    # Mean absolute error
    mae = np.mean([abs(p - s['average']) for p, s in zip(predictions, samples)])
    
    return {
        "spearman": corr,
        "p_value": p_value,
        "accuracy": accuracy,
        "mae": mae,
        "correct": correct,
        "total": len(predictions)
    }


# ============== SIMPLIFIED GEPA ==============
class SimpleGEPA:
    """Simplified GEPA for comparison testing."""
    
    def __init__(self, train_samples: List[Dict], budget: int = 50):
        self.train_samples = train_samples
        self.budget = budget
        self.budget_used = 0
        self.best_prompt = BASE_PROMPT
        self.best_score = 0.0
    
    def evaluate_prompt(self, prompt: str, samples: List[Dict]) -> float:
        """Evaluate a prompt on samples, return mean reward."""
        rewards = []
        for sample in samples:
            pred = get_single_prediction(prompt, sample)
            self.budget_used += 1
            
            distance = abs(pred - sample['average'])
            reward = max(0, 1 - distance / 4)
            if distance <= sample['stdev']:
                reward += 0.25
            rewards.append(min(reward, 1.0))
        
        return np.mean(rewards)
    
    def generate_feedback(self, prompt: str, samples: List[Dict]) -> str:
        """Generate feedback for reflection."""
        feedback_items = []
        for sample in samples:
            pred = get_single_prediction(prompt, sample)
            self.budget_used += 1
            
            distance = abs(pred - sample['average'])
            quality = "Good" if distance <= 1 else "Poor"
            
            feedback_items.append(
                f"Word: {sample['homonym']}, Meaning: {sample['judged_meaning'][:40]}...\n"
                f"Prediction: {pred}, Human Mean: {sample['average']:.1f}, Distance: {distance:.1f}, Quality: {quality}"
            )
        
        return "\n\n".join(feedback_items)
    
    def reflect(self, prompt: str, feedback: str) -> str:
        """Get improved prompt through reflection."""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": META_PROMPT.format(
                        candidate=prompt,
                        feedback=feedback
                    )}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            self.budget_used += 1
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Reflection error: {e}")
            return prompt
    
    def optimize(self) -> str:
        """Run simplified GEPA optimization."""
        print(f"  Starting GEPA optimization (budget: {self.budget})...")
        
        current_prompt = BASE_PROMPT
        
        # Initial evaluation
        eval_samples = random.sample(self.train_samples, min(10, len(self.train_samples)))
        self.best_score = self.evaluate_prompt(current_prompt, eval_samples)
        print(f"  Initial score: {self.best_score:.3f}")
        
        iteration = 0
        while self.budget_used < self.budget:
            iteration += 1
            
            # Sample for feedback
            feedback_samples = random.sample(self.train_samples, min(5, len(self.train_samples)))
            
            # Generate feedback
            feedback = self.generate_feedback(current_prompt, feedback_samples)
            
            # Reflect and get new prompt
            new_prompt = self.reflect(current_prompt, feedback)
            
            # Evaluate new prompt
            new_score = self.evaluate_prompt(new_prompt, eval_samples)
            
            if new_score > self.best_score:
                self.best_score = new_score
                self.best_prompt = new_prompt
                current_prompt = new_prompt
                print(f"  Iteration {iteration}: Improved! Score: {new_score:.3f}")
            else:
                print(f"  Iteration {iteration}: No improvement (score: {new_score:.3f})")
            
            if self.budget_used >= self.budget:
                break
        
        print(f"  Optimization complete. Best score: {self.best_score:.3f}")
        print(f"  Budget used: {self.budget_used}")
        
        return self.best_prompt


def run_comparison(
    data_path: str = "dev.json",
    train_path: str = "train.json",
    sample_size: int = SAMPLE_SIZE,
    output_dir: str = "comparison_results"
):
    """Run comparison of all three approaches."""
    
    print("=" * 60)
    print("APPROACH COMPARISON: GEPA vs 5-Run Avg vs Hybrid")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    with open(data_path, 'r') as f:
        test_data = json.load(f)
    
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    # Sample test data
    test_items = list(test_data.items())
    random.seed(42)  # For reproducibility
    sampled_items = random.sample(test_items, min(sample_size, len(test_items)))
    test_samples = [{"id": k, **v} for k, v in sampled_items]
    
    # Sample train data for GEPA
    train_items = list(train_data.items())
    train_samples = [{"id": k, **v} for k, v in random.sample(train_items, min(100, len(train_items)))]
    
    print(f"  Test samples: {len(test_samples)}")
    print(f"  Train samples for GEPA: {len(train_samples)}")
    
    results = {}
    
    # ============== APPROACH 1: Baseline (Single Run) ==============
    print("\n" + "=" * 60)
    print("[2] BASELINE: Single Run with Base Prompt")
    print("=" * 60)
    
    baseline_preds = []
    for sample in tqdm(test_samples, desc="Baseline"):
        pred = get_single_prediction(BASE_PROMPT, sample)
        baseline_preds.append(pred)
    
    results["baseline"] = calculate_metrics(baseline_preds, test_samples)
    print(f"  Spearman: {results['baseline']['spearman']:.4f}")
    print(f"  Accuracy: {results['baseline']['accuracy']:.4f} ({results['baseline']['correct']}/{results['baseline']['total']})")
    print(f"  MAE: {results['baseline']['mae']:.4f}")
    
    # ============== APPROACH 2: 5-Run Averaging ==============
    print("\n" + "=" * 60)
    print(f"[3] 5-RUN AVERAGING: {NUM_RUNS_AVERAGING} runs per sample")
    print("=" * 60)
    
    avg_preds = []
    for sample in tqdm(test_samples, desc="5-Run Avg"):
        pred = get_averaged_prediction(BASE_PROMPT, sample, num_runs=NUM_RUNS_AVERAGING)
        avg_preds.append(pred)
    
    results["5_run_avg"] = calculate_metrics(avg_preds, test_samples)
    print(f"  Spearman: {results['5_run_avg']['spearman']:.4f}")
    print(f"  Accuracy: {results['5_run_avg']['accuracy']:.4f} ({results['5_run_avg']['correct']}/{results['5_run_avg']['total']})")
    print(f"  MAE: {results['5_run_avg']['mae']:.4f}")
    
    # ============== APPROACH 3: GEPA ==============
    print("\n" + "=" * 60)
    print("[4] GEPA: Reflective Prompt Evolution")
    print("=" * 60)
    
    gepa = SimpleGEPA(train_samples, budget=GEPA_BUDGET)
    optimized_prompt = gepa.optimize()
    
    print("\n  Optimized Prompt:")
    print("  " + "-" * 40)
    print(f"  {optimized_prompt[:200]}..." if len(optimized_prompt) > 200 else f"  {optimized_prompt}")
    print("  " + "-" * 40)
    
    gepa_preds = []
    for sample in tqdm(test_samples, desc="GEPA"):
        pred = get_single_prediction(optimized_prompt, sample)
        gepa_preds.append(pred)
    
    results["gepa"] = calculate_metrics(gepa_preds, test_samples)
    print(f"  Spearman: {results['gepa']['spearman']:.4f}")
    print(f"  Accuracy: {results['gepa']['accuracy']:.4f} ({results['gepa']['correct']}/{results['gepa']['total']})")
    print(f"  MAE: {results['gepa']['mae']:.4f}")
    
    # ============== APPROACH 4: Hybrid (GEPA + 3-Run Avg) ==============
    print("\n" + "=" * 60)
    print(f"[5] HYBRID: GEPA Prompt + {NUM_RUNS_HYBRID}-Run Averaging")
    print("=" * 60)
    
    hybrid_preds = []
    for sample in tqdm(test_samples, desc="Hybrid"):
        pred = get_averaged_prediction(optimized_prompt, sample, num_runs=NUM_RUNS_HYBRID)
        hybrid_preds.append(pred)
    
    results["hybrid"] = calculate_metrics(hybrid_preds, test_samples)
    print(f"  Spearman: {results['hybrid']['spearman']:.4f}")
    print(f"  Accuracy: {results['hybrid']['accuracy']:.4f} ({results['hybrid']['correct']}/{results['hybrid']['total']})")
    print(f"  MAE: {results['hybrid']['mae']:.4f}")
    
    # ============== SUMMARY ==============
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Approach':<20} {'Spearman':>10} {'Accuracy':>10} {'MAE':>10}")
    print("-" * 52)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['spearman']:>10.4f} {metrics['accuracy']:>10.4f} {metrics['mae']:>10.4f}")
    
    # Find best approach
    print("\n" + "-" * 52)
    best_spearman = max(results.items(), key=lambda x: x[1]['spearman'])
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_mae = min(results.items(), key=lambda x: x[1]['mae'])
    
    print(f"Best Spearman:  {best_spearman[0]} ({best_spearman[1]['spearman']:.4f})")
    print(f"Best Accuracy:  {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"Best MAE:       {best_mae[0]} ({best_mae[1]['mae']:.4f})")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, "comparison_metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save optimized prompt
    with open(os.path.join(output_dir, "gepa_optimized_prompt.txt"), 'w') as f:
        f.write(optimized_prompt)
    
    # Save predictions for each approach
    for name, preds in [("baseline", baseline_preds), ("5_run_avg", avg_preds), 
                        ("gepa", gepa_preds), ("hybrid", hybrid_preds)]:
        pred_file = os.path.join(output_dir, f"predictions_{name}.jsonl")
        with open(pred_file, 'w') as f:
            for sample, pred in zip(test_samples, preds):
                f.write(json.dumps({"id": sample["id"], "prediction": pred}) + '\n')
    
    print(f"\nResults saved to {output_dir}/")
    
    return results, optimized_prompt


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare GEPA, 5-Run Avg, and Hybrid approaches')
    parser.add_argument('--data', type=str, default='dev.json', help='Test data path')
    parser.add_argument('--train', type=str, default='train.json', help='Training data path')
    parser.add_argument('--samples', type=int, default=50, help='Number of test samples')
    parser.add_argument('--gepa-budget', type=int, default=80, help='GEPA optimization budget')
    parser.add_argument('--output', type=str, default='comparison_results', help='Output directory')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model')
    
    args = parser.parse_args()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    global MODEL, GEPA_BUDGET, SAMPLE_SIZE
    MODEL = args.model
    GEPA_BUDGET = args.gepa_budget
    SAMPLE_SIZE = args.samples
    
    run_comparison(
        data_path=args.data,
        train_path=args.train,
        sample_size=args.samples,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
