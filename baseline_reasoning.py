"""
Enhanced baseline model with support for multiple reasoning models.
Supports OpenAI, Anthropic Claude, and other LLM providers.
"""

import json
import os
from typing import Optional, Dict, Any
from tqdm import tqdm
import time

# Try importing different API clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not available")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

def create_prompt(sample: Dict[str, Any], verbose: bool = False) -> str:
    """
    Create a detailed reasoning prompt for the model to rate plausibility of a word sense.
    """
    prompt = f"""You are an expert linguist evaluating word sense disambiguation in narrative contexts.

**TASK**: Rate the plausibility of a specific word sense in a short story on a scale from 1 to 5.

**RATING SCALE**:
- **5 (Highly plausible)**: The word sense fits perfectly with the narrative context. It's the most natural interpretation.
- **4 (Plausible)**: The word sense fits well. It's a reasonable interpretation given the context.
- **3 (Somewhat plausible)**: The word sense could fit, but it's not entirely clear or natural. Multiple interpretations are equally valid.
- **2 (Less plausible)**: The word sense doesn't fit well with the context. Other interpretations seem more natural.
- **1 (Implausible)**: The word sense doesn't fit at all. It would be confusing or nonsensical in this context.

**STORY CONTEXT**:

**Precontext** (3 sentences setting up the narrative):
{sample['precontext']}

**Ambiguous Sentence** (containing the target word "{sample['homonym']}"):
{sample['sentence']}

**Ending** (if provided):
{sample.get('ending', 'No ending provided - evaluate based on precontext and ambiguous sentence only.')}

---

**WORD SENSE TO EVALUATE**:
- **Target word**: {sample['homonym']}
- **Meaning to judge**: {sample['judged_meaning']}
- **Example sentence**: {sample['example_sentence']}

**REASONING INSTRUCTIONS**:
1. Read the complete narrative carefully (precontext + ambiguous sentence + ending)
2. Identify the narrative theme and context clues
3. Consider how the specified meaning of "{sample['homonym']}" would fit in the ambiguous sentence
4. Evaluate coherence with the overall story
5. Consider if other word senses would fit better or equally well
6. Assign a rating from 1 to 5 based on the scale above

**RESPONSE FORMAT**:
Think through your reasoning briefly, then provide your rating.

Your response should end with: **RATING: [number]**

Where [number] is a single integer from 1 to 5."""
    
    return prompt

def create_simple_prompt(sample: Dict[str, Any]) -> str:
    """
    Create a simpler, more concise prompt for faster models.
    """
    ending_text = sample.get('ending', '')
    full_story = f"{sample['precontext']}\n{sample['sentence']}"
    if ending_text:
        full_story += f"\n{ending_text}"
    
    prompt = f"""Rate the plausibility (1-5) of this word sense in the story context:

Story:
{full_story}

Word: "{sample['homonym']}"
Meaning: {sample['judged_meaning']}
Example: {sample['example_sentence']}

Scale:
5=Highly plausible, 4=Plausible, 3=Somewhat plausible, 2=Less plausible, 1=Implausible

Respond with ONLY a number from 1 to 5."""
    
    return prompt

def extract_rating(text: str) -> int:
    """
    Extract rating from model response.
    Handles various formats including reasoning followed by rating.
    """
    # Look for "RATING: X" pattern
    if "RATING:" in text.upper():
        parts = text.upper().split("RATING:")
        if len(parts) > 1:
            rating_part = parts[-1].strip()
            for char in rating_part:
                if char.isdigit():
                    rating = int(char)
                    if 1 <= rating <= 5:
                        return rating
    
    # Look for any digit between 1 and 5
    for char in text:
        if char.isdigit():
            rating = int(char)
            if 1 <= rating <= 5:
                return rating
    
    # Fallback to middle rating
    print(f"Warning: Could not parse rating from '{text[:100]}...', using 3 as fallback")
    return 3

class OpenAIRater:
    """Rater using OpenAI models."""
    
    def __init__(self, model: str = "gpt-5.1", temperature: float = 0.3, num_runs: int = 5):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.num_runs = num_runs
        
        # Use simple prompt for faster models
        self.use_simple_prompt = "gpt-3.5" in model.lower()
    
    def get_rating(self, sample: Dict[str, Any]) -> int:
        """Get plausibility rating from OpenAI model by averaging multiple runs."""
        prompt = create_simple_prompt(sample) if self.use_simple_prompt else create_prompt(sample)
        
        ratings = []
        for run in range(self.num_runs):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at understanding nuanced word meanings and evaluating their plausibility in narrative contexts."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    #max_tokens=50 if self.use_simple_prompt else 500
                )
                
                rating_text = response.choices[0].message.content.strip()
                rating = extract_rating(rating_text)
                ratings.append(rating)
                
            except Exception as e:
                print(f"Error calling OpenAI API (run {run + 1}): {e}")
                ratings.append(3)  # Fallback rating
        
        # Calculate average and round to nearest integer
        avg_rating = round(sum(ratings) / len(ratings))
        # Ensure rating is within valid range [1, 5]
        avg_rating = max(1, min(5, avg_rating))
        return avg_rating

class AnthropicRater:
    """Rater using Anthropic Claude models."""
    
    def __init__(self, model: str = "claude-sonnet-4-5-20250929", temperature: float = 0.3):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not available. Install with: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature
    
    def get_rating(self, sample: Dict[str, Any]) -> int:
        """Get plausibility rating from Anthropic model."""
        prompt = create_prompt(sample)
        
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            rating_text = message.content[0].text.strip()
            return extract_rating(rating_text)
            
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return 3

class GeminiRater:
    """Rater using Google Gemini models."""
    
    def __init__(self, model: str = "gemini-2.0-flash-exp", temperature: float = 0.3):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenerativeAI library not available. Install with: pip install google-generativeai")
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 500,
            }
        )
    
    def get_rating(self, sample: Dict[str, Any]) -> int:
        """Get plausibility rating from Gemini model."""
        prompt = create_prompt(sample)
        
        try:
            response = self.model.generate_content(prompt)
            rating_text = response.text.strip()
            return extract_rating(rating_text)
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return 3

def get_rater(provider: str, model: str, temperature: float, num_runs: int = 5):
    """Factory function to get appropriate rater."""
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIRater(model=model, temperature=temperature, num_runs=num_runs)
    elif provider == "anthropic":
        return AnthropicRater(model=model, temperature=temperature)
    elif provider == "gemini":
        return GeminiRater(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'openai', 'anthropic', or 'gemini'")

def process_dataset(
    data_path: str,
    output_path: str,
    provider: str = "openai",
    model: str = "gpt-5.1",
    temperature: float = 0.3,
    max_samples: Optional[int] = None,
    delay: float = 0.1,
    num_runs: int = 5
):
    """
    Process entire dataset and generate predictions.
    For OpenAI, runs each sample num_runs times and averages the scores.
    """
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Initialize rater
    rater = get_rater(provider, model, temperature, num_runs=num_runs)
    
    predictions = []
    
    # Process each sample
    samples = list(data.items())
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Processing {len(samples)} samples with {provider}/{model} (temperature={temperature}, num_runs={num_runs})...")
    
    for idx, (sample_id, sample) in enumerate(tqdm(samples)):
        rating = rater.get_rating(sample)
        predictions.append({
            "id": sample_id,
            "prediction": rating
        })
        
        # Add delay to avoid rate limits
        if delay > 0 and (idx + 1) % 10 == 0:
            time.sleep(delay)
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    print(f"Predictions saved to {output_path}")
    return predictions

def main():
    """Main function to run enhanced baseline model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced baseline model for AmbiStory WSD task')
    parser.add_argument('--data', type=str, default='dev.json',
                        help='Path to input data file')
    parser.add_argument('--output', type=str, default='input/res/predictions.jsonl',
                        help='Path to output predictions file')
    parser.add_argument('--provider', type=str, default='openai',
                        choices=['openai', 'anthropic', 'gemini'],
                        help='LLM provider to use')
    parser.add_argument('--model', type=str, default='gpt-5.1',
                        help='Model to use (e.g., gpt-4o, claude-3-5-sonnet-20241022, gemini-2.0-flash-exp)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Temperature for model sampling')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between API calls (seconds)')
    parser.add_argument('--num-runs', type=int, default=5,
                        help='Number of times to run each sample and average (default: 5)')
    
    args = parser.parse_args()
    
    # Check for API key
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")
        return
    
    if args.provider == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        return
    
    # Process dataset
    process_dataset(
        data_path=args.data,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_samples=args.max_samples,
        delay=args.delay,
        num_runs=args.num_runs
    )

if __name__ == "__main__":
    main()

