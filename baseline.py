import json
import os
from openai import OpenAI
from tqdm import tqdm
import time

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def create_prompt(sample):
    """
    Create a prompt for the model to rate plausibility of a word sense.
    """
    prompt = f"""You are evaluating the plausibility of a specific word sense in a short story context.

**Task**: Rate how plausible the given word sense is in the context of the story on a scale from 1 to 5.

**Rating Scale**:
- 5: Highly plausible - the word sense fits perfectly with the context
- 4: Plausible - the word sense fits well with the context
- 3: Somewhat plausible - the word sense could fit but is not entirely clear
- 2: Less plausible - the word sense doesn't fit well with the context
- 1: Implausible - the word sense doesn't fit at all with the context

**Story Components**:

**Precontext** (setting up the story):
{sample['precontext']}

**Ambiguous Sentence** (containing the target word '{sample['homonym']}'):
{sample['sentence']}

**Ending** (if provided):
{sample.get('ending', 'No ending provided.')}

**Target Word**: {sample['homonym']}

**Word Sense Being Judged**: {sample['judged_meaning']}

**Example of this word sense**: {sample['example_sentence']}

**Instructions**:
1. Read the entire story context carefully
2. Consider how well the specified word sense of '{sample['homonym']}' fits in the ambiguous sentence given the precontext and ending
3. Think about whether the meaning makes sense narratively and contextually
4. Provide ONLY a single integer rating from 1 to 5

Your response must be ONLY the integer rating (1, 2, 3, 4, or 5) with no additional text."""
    
    return prompt

def get_plausibility_rating(sample, model="gpt-4o", temperature=0.3):
    """
    Get plausibility rating from OpenAI model.
    """
    prompt = create_prompt(sample)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at understanding nuanced word meanings and evaluating their plausibility in context. You provide ratings as single integers."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=10
        )
        
        rating_text = response.choices[0].message.content.strip()
        
        # Extract the rating (handle cases where model might add extra text)
        for char in rating_text:
            if char.isdigit():
                rating = int(char)
                if 1 <= rating <= 5:
                    return rating
        
        # Fallback to middle rating if parsing fails
        print(f"Warning: Could not parse rating '{rating_text}', using 3 as fallback")
        return 3
        
    except Exception as e:
        print(f"Error calling API: {e}")
        return 3  # Return middle rating on error

def process_dataset(data_path, output_path, model="gpt-4o", temperature=0.3, max_samples=None):
    """
    Process entire dataset and generate predictions.
    """
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    predictions = []
    
    # Process each sample
    samples = list(data.items())
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Processing {len(samples)} samples with model {model}...")
    
    for idx, (sample_id, sample) in enumerate(tqdm(samples)):
        rating = get_plausibility_rating(sample, model=model, temperature=temperature)
        predictions.append({
            "id": sample_id,
            "prediction": rating
        })
        
        # Add small delay to avoid rate limits
        if (idx + 1) % 10 == 0:
            time.sleep(1)
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    print(f"Predictions saved to {output_path}")
    return predictions

def main():
    """
    Main function to run baseline model.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Baseline model for AmbiStory WSD task')
    parser.add_argument('--data', type=str, default='dev.json', 
                        help='Path to input data file')
    parser.add_argument('--output', type=str, default='input/res/predictions.jsonl',
                        help='Path to output predictions file')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='OpenAI model to use (e.g., gpt-4o, gpt-4-turbo, gpt-3.5-turbo)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Temperature for model sampling')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    # Process dataset
    process_dataset(
        data_path=args.data,
        output_path=args.output,
        model=args.model,
        temperature=args.temperature,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()

