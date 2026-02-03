"""
GEPA: Reflective Prompt Evolution for Word Sense Plausibility Rating
Implements the GEPA approach using OpenAI for the SemEval AmbiStory WSD task.
"""

import json
import os
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from collections import Counter
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ============== CONFIGURATION ==============
MODEL = "gpt-4o-mini"  # Target model for predictions
REFLECTION_MODEL = "gpt-4o-mini"  # Model for reflection/prompt evolution
BUDGET = 100  # Total API calls budget
MINI_BATCH_SIZE = 5  # Samples per feedback batch
NUM_INITIAL_CANDIDATES = 3  # Number of initial prompt candidates
EXPLOIT_PROB = 0.85  # Probability to exploit vs explore
MERGE_PROB = 0.80  # Probability to merge vs select single

# ============== PROMPTS ==============
SEED_PROMPT = """You are an expert linguist evaluating word sense disambiguation in narrative contexts.

Rate the plausibility of a specific word sense in a short story on a scale from 1 to 5.

**RATING SCALE**:
- 5 (Highly plausible): The word sense fits perfectly with the narrative context.
- 4 (Plausible): The word sense fits well. It's a reasonable interpretation.
- 3 (Somewhat plausible): The word sense could fit, but it's not entirely clear.
- 2 (Less plausible): The word sense doesn't fit well with the context.
- 1 (Implausible): The word sense doesn't fit at all.

Respond with ONLY a single integer from 1 to 5."""

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

MERGE_PROMPT = """You are an expert at synthesizing prompts.

## Candidate Prompts to Merge:
{candidates}

## Goal:
Create a single, coherent prompt that combines the strengths of all candidates.
The merged prompt should address weaknesses of individual candidates.

Output ONLY the merged prompt text. Do not include examples."""

INITIAL_CANDIDATES_PROMPT = """You are an expert at generating LLM prompts for word sense disambiguation.

## Seed Prompt:
{seed_prompt}

## Task Examples:
{examples}

## Goal:
Generate {num_prompts} diverse, high-quality prompt variations for rating word sense plausibility (1-5 scale).
Each prompt should approach the task differently while maintaining accuracy.

Output each prompt on a new line, separated by "---"."""


def extract_rating(text: str) -> int:
    """Extract rating (1-5) from model response."""
    text = text.strip()
    
    # Look for "RATING: X" pattern
    if "RATING:" in text.upper():
        parts = text.upper().split("RATING:")
        if len(parts) > 1:
            for char in parts[-1].strip():
                if char.isdigit():
                    rating = int(char)
                    if 1 <= rating <= 5:
                        return rating
    
    # Look for any digit 1-5
    for char in text:
        if char.isdigit():
            rating = int(char)
            if 1 <= rating <= 5:
                return rating
    
    return 3  # Fallback


def format_sample_for_prompt(sample: Dict[str, Any]) -> str:
    """Format a sample into the input format for the model."""
    ending = sample.get('ending', '')
    return f"""**Story Context:**
Precontext: {sample['precontext']}
Sentence: {sample['sentence']}
Ending: {ending if ending else 'N/A'}

**Word Sense to Evaluate:**
Word: "{sample['homonym']}"
Meaning: {sample['judged_meaning']}
Example: {sample['example_sentence']}"""


def get_prediction(prompt: str, sample: Dict[str, Any], model: str = MODEL) -> int:
    """Get a plausibility rating from the model."""
    sample_text = format_sample_for_prompt(sample)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": sample_text}
            ],
            temperature=0.3,
            max_tokens=50
        )
        return extract_rating(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
        return 3


def evaluate_prediction(pred: int, mean: float, std: float) -> float:
    """
    Evaluate prediction quality based on distance from human mean.
    Returns a score between 0 and 1.
    """
    distance = abs(pred - mean)
    # Base reward: closer to mean = higher reward
    reward = max(0, 1 - distance / 4)
    # Bonus if within 1 standard deviation
    if distance <= std:
        reward += 0.25
    return min(reward, 1.0)


def generate_feedback(pred: int, mean: float, std: float, sample: Dict[str, Any]) -> str:
    """Generate natural language feedback for a prediction."""
    distance = abs(pred - mean)
    
    if distance <= 0.5:
        quality = "Excellent"
    elif distance <= 1.0:
        quality = "Good"
    elif distance <= 1.5:
        quality = "Fair"
    else:
        quality = "Poor"
    
    feedback = f"""Prediction: {pred}
Human Mean: {mean:.2f} (std: {std:.2f})
Quality: {quality}
Distance from mean: {distance:.2f}
"""
    
    if distance > 1.0:
        if pred > mean:
            feedback += "Issue: Overestimated plausibility. Consider contextual mismatches more carefully."
        else:
            feedback += "Issue: Underestimated plausibility. The word sense may fit better than assessed."
    
    return feedback


def get_reflection(current_prompt: str, feedback_items: List[str]) -> str:
    """Use reflection model to generate an improved prompt."""
    feedback_text = "\n\n".join(feedback_items)
    
    try:
        response = client.chat.completions.create(
            model=REFLECTION_MODEL,
            messages=[
                {"role": "user", "content": META_PROMPT.format(
                    candidate=current_prompt,
                    feedback=feedback_text
                )}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Reflection error: {e}")
        return current_prompt


def merge_prompts(prompts: List[str]) -> str:
    """Merge multiple prompts into one."""
    candidates_text = "\n\n---\n\n".join([f"Prompt {i+1}:\n{p}" for i, p in enumerate(prompts)])
    
    try:
        response = client.chat.completions.create(
            model=REFLECTION_MODEL,
            messages=[
                {"role": "user", "content": MERGE_PROMPT.format(candidates=candidates_text)}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Merge error: {e}")
        return prompts[0]


def generate_initial_candidates(seed_prompt: str, samples: List[Dict], num_prompts: int = 3) -> List[str]:
    """Generate initial candidate prompts from seed."""
    # Format a few examples
    examples = []
    for s in samples[:3]:
        examples.append(f"Word: {s['homonym']}, Meaning: {s['judged_meaning']}, Human Rating: {s['average']:.1f}")
    examples_text = "\n".join(examples)
    
    try:
        response = client.chat.completions.create(
            model=REFLECTION_MODEL,
            messages=[
                {"role": "user", "content": INITIAL_CANDIDATES_PROMPT.format(
                    seed_prompt=seed_prompt,
                    examples=examples_text,
                    num_prompts=num_prompts
                )}
            ],
            temperature=0.9,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        # Split by separator
        candidates = [p.strip() for p in content.split("---") if p.strip()]
        
        # Always include seed prompt
        if seed_prompt not in candidates:
            candidates.insert(0, seed_prompt)
        
        return candidates[:num_prompts + 1]
    except Exception as e:
        print(f"Initial candidates error: {e}")
        return [seed_prompt]


def select_candidates_pareto(candidates: List[Dict], scores: np.ndarray) -> Tuple[List[int], List[float]]:
    """
    Select candidates using Pareto-based selection.
    Returns candidate indices and selection probabilities.
    """
    num_candidates = len(candidates)
    num_tasks = scores.shape[1]
    
    if num_candidates == 1:
        return [0], [1.0]
    
    # Find best score for each task
    s_star = np.max(scores, axis=0)
    
    # Find candidates that achieve best score for each task
    P_star = [set() for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_candidates):
            if scores[j, i] == s_star[i]:
                P_star[i].add(j)
    
    # Get unique candidates in Pareto frontier
    C = set()
    for p_set in P_star:
        C.update(p_set)
    
    # Remove dominated candidates
    D = set()
    C_list = list(C)
    
    for idx1 in C_list:
        for idx2 in C_list:
            if idx1 == idx2:
                continue
            # Check if idx1 is dominated by idx2
            all_leq = all(scores[idx1, i] <= scores[idx2, i] for i in range(num_tasks))
            any_lt = any(scores[idx1, i] < scores[idx2, i] for i in range(num_tasks))
            if all_leq and any_lt:
                D.add(idx1)
                break
    
    # Get non-dominated candidates
    hat_C = [c for c in C_list if c not in D]
    
    if not hat_C:
        # Fallback to best mean score
        mean_scores = np.mean(scores, axis=1)
        return [int(np.argmax(mean_scores))], [1.0]
    
    # Calculate selection probabilities based on frequency in P_star
    f = Counter()
    for p_set in P_star:
        for k in (p_set - D):
            f[k] += 1
    
    probs = [f[k] for k in hat_C]
    total = sum(probs)
    probs = [p / total for p in probs]
    
    return hat_C, probs


class GEPA:
    """GEPA optimizer for word sense plausibility rating."""
    
    def __init__(
        self,
        train_data: Dict[str, Dict],
        model: str = MODEL,
        budget: int = BUDGET,
        mini_batch_size: int = MINI_BATCH_SIZE
    ):
        self.train_data = train_data
        self.model = model
        self.budget = budget
        self.mini_batch_size = mini_batch_size
        self.budget_used = 0
        
        # Split data into Pareto (for scoring) and Feedback (for learning)
        items = list(train_data.items())
        random.shuffle(items)
        
        split_idx = min(20, len(items) // 2)
        self.pareto_samples = dict(items[:split_idx])
        self.feedback_samples = dict(items[split_idx:])
        
        # Candidate prompts
        self.candidates = []  # List of {"id", "prompt", "scores", "mean_score"}
        self.scores_matrix = None  # Shape: (num_candidates, num_pareto_samples)
        
        self.best_prompt = SEED_PROMPT
        self.best_score = 0.0
    
    def run(self) -> str:
        """Run GEPA optimization and return the best prompt."""
        print(f"Starting GEPA optimization...")
        print(f"Budget: {self.budget}, Pareto samples: {len(self.pareto_samples)}, Feedback samples: {len(self.feedback_samples)}")
        
        # Generate initial candidates
        print("\n[1] Generating initial candidate prompts...")
        pareto_list = [v for v in self.pareto_samples.values()]
        initial_prompts = generate_initial_candidates(SEED_PROMPT, pareto_list, NUM_INITIAL_CANDIDATES)
        self.budget_used += 1
        
        print(f"Generated {len(initial_prompts)} initial candidates")
        
        # Initialize candidates
        for i, prompt in enumerate(initial_prompts):
            self.candidates.append({
                "id": i,
                "prompt": prompt,
                "scores": [],
                "mean_score": 0.0
            })
        
        # Evaluate initial candidates on Pareto samples
        print("\n[2] Evaluating initial candidates on Pareto samples...")
        pareto_ids = list(self.pareto_samples.keys())
        self.scores_matrix = np.zeros((len(self.candidates), len(pareto_ids)))
        
        for c_idx, candidate in enumerate(tqdm(self.candidates, desc="Evaluating candidates")):
            scores = []
            for p_idx, sample_id in enumerate(pareto_ids):
                sample = self.pareto_samples[sample_id]
                pred = get_prediction(candidate["prompt"], sample, self.model)
                score = evaluate_prediction(pred, sample["average"], sample["stdev"])
                scores.append(score)
                self.scores_matrix[c_idx, p_idx] = score
                self.budget_used += 1
            
            candidate["scores"] = scores
            candidate["mean_score"] = np.mean(scores)
            print(f"  Candidate {c_idx}: mean_score = {candidate['mean_score']:.3f}")
        
        # Main optimization loop
        print("\n[3] Starting optimization loop...")
        iteration = 0
        
        while self.budget_used < self.budget:
            iteration += 1
            print(f"\n--- Iteration {iteration} (budget used: {self.budget_used}/{self.budget}) ---")
            
            # Sample mini-batch from feedback samples
            feedback_ids = random.sample(
                list(self.feedback_samples.keys()),
                min(self.mini_batch_size, len(self.feedback_samples))
            )
            
            # Select candidate (exploit vs explore)
            if random.random() < EXPLOIT_PROB:
                # Exploit: use Pareto selection
                selected_ids, probs = select_candidates_pareto(self.candidates, self.scores_matrix)
                
                if len(selected_ids) > 1 and random.random() > MERGE_PROB:
                    # Merge candidates
                    print(f"  [Exploit-Merge] Merging {len(selected_ids)} candidates...")
                    prompts_to_merge = [self.candidates[i]["prompt"] for i in selected_ids]
                    merged_prompt = merge_prompts(prompts_to_merge)
                    self.budget_used += 1
                    
                    selected_candidate = {
                        "id": len(self.candidates),
                        "prompt": merged_prompt,
                        "scores": [],
                        "mean_score": 0.0
                    }
                else:
                    # Select one candidate
                    selected_idx = random.choices(selected_ids, weights=probs, k=1)[0]
                    selected_candidate = self.candidates[selected_idx]
                    print(f"  [Exploit] Selected candidate {selected_candidate['id']}")
            else:
                # Explore: random selection
                selected_candidate = random.choice(self.candidates)
                print(f"  [Explore] Selected candidate {selected_candidate['id']}")
            
            # Evaluate on mini-batch and collect feedback
            feedback_items = []
            minibatch_scores = []
            
            for sample_id in feedback_ids:
                sample = self.feedback_samples[sample_id]
                pred = get_prediction(selected_candidate["prompt"], sample, self.model)
                score = evaluate_prediction(pred, sample["average"], sample["stdev"])
                minibatch_scores.append(score)
                self.budget_used += 1
                
                feedback = generate_feedback(pred, sample["average"], sample["stdev"], sample)
                feedback_items.append(f"Sample: {sample['homonym']} - {sample['judged_meaning'][:50]}...\n{feedback}")
            
            mean_minibatch_score = np.mean(minibatch_scores)
            print(f"  Mini-batch score: {mean_minibatch_score:.3f}")
            
            # Reflect and generate new prompt
            print(f"  Reflecting and generating new prompt...")
            new_prompt = get_reflection(selected_candidate["prompt"], feedback_items)
            self.budget_used += 1
            
            # Evaluate new prompt on mini-batch
            new_minibatch_scores = []
            for sample_id in feedback_ids:
                sample = self.feedback_samples[sample_id]
                pred = get_prediction(new_prompt, sample, self.model)
                score = evaluate_prediction(pred, sample["average"], sample["stdev"])
                new_minibatch_scores.append(score)
                self.budget_used += 1
            
            mean_new_score = np.mean(new_minibatch_scores)
            print(f"  New prompt mini-batch score: {mean_new_score:.3f}")
            
            # If improved, evaluate on Pareto and add to candidates
            if mean_new_score >= mean_minibatch_score:
                print(f"  Improvement detected! Evaluating on Pareto samples...")
                
                new_pareto_scores = []
                for sample_id in pareto_ids:
                    sample = self.pareto_samples[sample_id]
                    pred = get_prediction(new_prompt, sample, self.model)
                    score = evaluate_prediction(pred, sample["average"], sample["stdev"])
                    new_pareto_scores.append(score)
                    self.budget_used += 1
                
                new_mean_score = np.mean(new_pareto_scores)
                print(f"  New prompt Pareto score: {new_mean_score:.3f}")
                
                # Add new candidate
                new_candidate = {
                    "id": len(self.candidates),
                    "prompt": new_prompt,
                    "scores": new_pareto_scores,
                    "mean_score": new_mean_score
                }
                self.candidates.append(new_candidate)
                
                # Update scores matrix
                self.scores_matrix = np.vstack([self.scores_matrix, new_pareto_scores])
                
                # Update best if improved
                if new_mean_score > self.best_score:
                    self.best_score = new_mean_score
                    self.best_prompt = new_prompt
                    print(f"  New best prompt! Score: {self.best_score:.3f}")
            else:
                print(f"  No improvement, skipping...")
        
        # Final best selection
        best_idx = np.argmax([c["mean_score"] for c in self.candidates])
        self.best_prompt = self.candidates[best_idx]["prompt"]
        self.best_score = self.candidates[best_idx]["mean_score"]
        
        print(f"\n[4] Optimization complete!")
        print(f"Best candidate: {best_idx}, Score: {self.best_score:.3f}")
        print(f"Total budget used: {self.budget_used}")
        
        return self.best_prompt


def process_dataset_with_gepa(
    train_path: str,
    test_path: str,
    output_path: str,
    budget: int = BUDGET,
    max_test_samples: Optional[int] = None
):
    """
    Run GEPA on training data to optimize prompt, then evaluate on test data.
    """
    # Load data
    print("Loading data...")
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    # Run GEPA optimization
    print(f"\n{'='*50}")
    print("GEPA OPTIMIZATION PHASE")
    print(f"{'='*50}")
    
    gepa = GEPA(train_data, budget=budget)
    best_prompt = gepa.run()
    
    print(f"\n{'='*50}")
    print("BEST PROMPT:")
    print(f"{'='*50}")
    print(best_prompt)
    
    # Apply to test data
    print(f"\n{'='*50}")
    print("INFERENCE PHASE")
    print(f"{'='*50}")
    
    predictions = []
    test_items = list(test_data.items())
    if max_test_samples:
        test_items = test_items[:max_test_samples]
    
    print(f"Processing {len(test_items)} test samples...")
    
    for sample_id, sample in tqdm(test_items, desc="Generating predictions"):
        pred = get_prediction(best_prompt, sample)
        predictions.append({
            "id": sample_id,
            "prediction": pred
        })
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    print(f"\nPredictions saved to {output_path}")
    
    # Save the best prompt
    prompt_path = output_path.replace('.jsonl', '_best_prompt.txt')
    with open(prompt_path, 'w') as f:
        f.write(best_prompt)
    print(f"Best prompt saved to {prompt_path}")
    
    return predictions, best_prompt


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GEPA for Word Sense Plausibility Rating')
    parser.add_argument('--train', type=str, default='train.json', help='Training data path')
    parser.add_argument('--test', type=str, default='dev.json', help='Test data path')
    parser.add_argument('--output', type=str, default='input/res/predictions_gepa.jsonl', help='Output path')
    parser.add_argument('--budget', type=int, default=100, help='API call budget for optimization')
    parser.add_argument('--max-test', type=int, default=None, help='Max test samples')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use')
    
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Update global model
    global MODEL, REFLECTION_MODEL
    MODEL = args.model
    REFLECTION_MODEL = args.model
    
    process_dataset_with_gepa(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        budget=args.budget,
        max_test_samples=args.max_test
    )


if __name__ == "__main__":
    main()
