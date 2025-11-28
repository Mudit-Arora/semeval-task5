# SemEval 2026 Task 5: AmbiStory Word Sense Disambiguation

Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding

## Overview

This repository contains baseline models for the AmbiStory dataset, which focuses on Word Sense Disambiguation (WSD) in narrative contexts. Unlike traditional WSD tasks that assume one "correct" sense, this task acknowledges that ambiguities, underspecification, and personal opinions can influence which word senses are plausible in a given context.

### Task Description

The task is to predict the **human-perceived plausibility** of a word sense in the context of a 5-sentence short story. Each story consists of:

1. **Precontext** (3 sentences): Sets up the narrative
2. **Ambiguous Sentence** (1 sentence): Contains a homonym with multiple plausible interpretations
3. **Ending** (optional, 1 sentence): May imply a specific word sense

You must rate the plausibility of a given word sense on a **scale from 1 to 5**, where:
- 5 = Highly plausible
- 4 = Plausible
- 3 = Somewhat plausible
- 2 = Less plausible
- 1 = Implausible

### Evaluation Metrics

Models are evaluated using:

1. **Spearman Correlation**: How well predicted scores correlate with human averages
2. **Accuracy Within Standard Deviation**: Proportion of predictions within ±1 std dev of human average

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd semeval-task5

# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

### 2. Run Quick Test

Test the baseline on 20 samples (~2 minutes):

```bash
./run_baseline_test.sh
```

### 3. Run Full Baseline

Process the entire development set:

```bash
python baseline.py --data dev.json --output input/res/predictions.jsonl
```

### 4. Evaluate Predictions

```bash
python scoring.py dev.json input/res/predictions.jsonl output/scores.json
```

## Files and Structure

```
semeval-task5/
├── baseline.py                  # Simple OpenAI-based baseline
├── baseline_reasoning.py        # Enhanced baseline with multiple providers
├── scoring.py                   # Evaluation script
├── run_baseline_test.sh        # Quick test script
├── train.json                   # Training data (~9,512 samples)
├── dev.json                     # Development data (~2,457 samples)
├── requirements.txt             # Python dependencies
├── USAGE.md                     # Detailed usage guide
└── README.md                    # This file
```

## Baseline Models

### Basic Baseline (`baseline.py`)

Simple baseline using OpenAI models (GPT-4o, GPT-4-turbo, GPT-3.5-turbo).

**Usage:**
```bash
python baseline.py --data dev.json --model gpt-4o --temperature 0.3
```

**Features:**
- Clean, straightforward implementation
- Good for getting started quickly
- Uses structured prompts for rating
- Includes rate limiting and error handling

### Enhanced Baseline (`baseline_reasoning.py`)

More sophisticated baseline supporting multiple LLM providers.

**Usage:**
```bash
# Using OpenAI (GPT-4o)
python baseline_reasoning.py --provider openai --model gpt-4o --data dev.json

# Using Anthropic (Claude)
export ANTHROPIC_API_KEY='your-key'
python baseline_reasoning.py --provider anthropic --model claude-3-5-sonnet-20241022 --data dev.json
```

**Features:**
- Support for multiple providers (OpenAI, Anthropic)
- Enhanced reasoning prompts
- Flexible rating extraction
- Better error handling

## Example

**Input Sample:**

```json
{
  "homonym": "track",
  "judged_meaning": "a pair of parallel rails providing a runway for wheels",
  "precontext": "The detectives arrived at the abandoned train station. They were looking for signs of the missing artifact. A faint trail caught their attention.",
  "sentence": "They followed the track.",
  "ending": "They began to run along the abandoned railway line, hopping from wooden sleeper to sleeper to avoid twisting an ankle.",
  "example_sentence": "The train glided smoothly along the track."
}
```

**Expected Output:**

```json
{"id": "0", "prediction": 4}
```

*Reasoning*: The ending strongly suggests the "railway rails" meaning, making it highly plausible (4-5 range).

## Advanced Usage

### Testing with Subset

```bash
python baseline.py --data dev.json --max-samples 100 --output input/res/test.jsonl
```

### Adjusting Temperature

```bash
# More deterministic (recommended)
python baseline.py --temperature 0.1

# More diverse
python baseline.py --temperature 0.5
```

### Using Different Models

```bash
# GPT-4o (recommended - best balance)
python baseline.py --model gpt-4o

# GPT-4 Turbo (more expensive, potentially better)
python baseline.py --model gpt-4-turbo

# GPT-3.5 Turbo (faster, cheaper, lower accuracy)
python baseline.py --model gpt-3.5-turbo

# Claude 3.5 Sonnet (requires Anthropic API)
python baseline_reasoning.py --provider anthropic --model claude-3-5-sonnet-20241022
```

## Dataset Statistics

### Development Set (`dev.json`)
- Total samples: ~2,457
- Unique story setups: ~410
- Samples per setup: 6 (2 word senses × 3 ending types)
- Average human rating: ~3.5
- Average standard deviation: ~1.0

### Training Set (`train.json`)
- Total samples: ~9,512
- Can be used for few-shot prompting or fine-tuning

## Tips for Better Performance

1. **Use GPT-4o or Claude 3.5 Sonnet**: Best understanding of nuanced word meanings
2. **Lower Temperature (0.1-0.3)**: More consistent predictions
3. **Test Small First**: Use `--max-samples` to test before full run
4. **Consider Context Carefully**: The ending often provides crucial disambiguation cues
5. **Monitor Costs**: ~2,500 samples × $0.01/sample = ~$25 for full dev set with GPT-4o

## Expected Performance

Baseline models typically achieve:
- **Spearman Correlation**: 0.40-0.65
- **Accuracy Within Std Dev**: 0.60-0.80
- **Mean Absolute Error**: 0.7-1.2

State-of-the-art systems should aim for:
- **Spearman Correlation**: >0.70
- **Accuracy Within Std Dev**: >0.85

## Troubleshooting

### API Key Not Set
```bash
export OPENAI_API_KEY='your-key-here'
# Or for Anthropic:
export ANTHROPIC_API_KEY='your-key-here'
```

### Rate Limit Errors
- Increase delays in the code
- Use a lower tier model
- Process in smaller batches with `--max-samples`

### Module Import Errors
```bash
pip install -r requirements.txt --upgrade
```

## Citation

If you use this dataset or baseline, please cite:

```bibtex
@inproceedings{semeval2026task5,
  title={SemEval 2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding},
  author={[Authors]},
  booktitle={Proceedings of SemEval 2026},
  year={2026}
}
```

## License

[Specify license here]

## Contact

For questions or issues, please [specify contact method].
