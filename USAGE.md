# AmbiStory WSD Baseline Model - Usage Guide

This guide explains how to use the baseline model for the AmbiStory Word Sense Disambiguation task.

## Overview

The baseline model uses OpenAI's language models to predict the plausibility of word senses in ambiguous contexts. The model receives a story with context and rates how plausible a specific word sense is on a scale from 1 to 5.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

You need an OpenAI API key to run the baseline model:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or add it to your `.bashrc` or `.zshrc` for persistence.

### 3. Verify Folder Structure

The script will create these folders automatically, but you can verify:

```
semeval-task5/
├── input/
│   ├── ref/          # Reference/gold data
│   └── res/          # Results/predictions
├── output/           # Evaluation scores
├── baseline.py       # Baseline model script
├── scoring.py        # Evaluation script
├── dev.json          # Development set
└── train.json        # Training set
```

## Running the Baseline Model

### Basic Usage

Run the baseline model on the development set:

```bash
python baseline.py --data dev.json --output input/res/predictions.jsonl
```

### Test with Sample Data

To test the model on a small subset first (recommended):

```bash
python baseline.py --data dev.json --output input/res/predictions.jsonl --max-samples 10
```

### Using Different Models

You can experiment with different OpenAI models:

```bash
# Using GPT-4o (default, recommended for best performance)
python baseline.py --data dev.json --model gpt-4o

# Using GPT-4 Turbo
python baseline.py --data dev.json --model gpt-4-turbo

# Using GPT-3.5 Turbo (faster, cheaper, but lower accuracy)
python baseline.py --data dev.json --model gpt-3.5-turbo
```

### Adjusting Temperature

Temperature controls randomness (0.0 = deterministic, 1.0 = more random):

```bash
python baseline.py --data dev.json --temperature 0.1  # More consistent
python baseline.py --data dev.json --temperature 0.5  # More diverse
```

### Full Command Options

```bash
python baseline.py \
    --data dev.json \
    --output input/res/predictions.jsonl \
    --model gpt-4o \
    --temperature 0.3 \
    --max-samples 100
```

## Evaluating Predictions

### Prepare Gold Standard

Copy your gold standard data to the reference folder:

```bash
cp dev.json input/ref/solution.jsonl
# Or if you already have it in the right format:
# cp your_gold_file.jsonl input/ref/solution.jsonl
```

### Run Evaluation

```bash
python scoring.py input/ref/dev.json input/res/predictions.jsonl output/scores.json
```

Or if using JSONL format:

```bash
python scoring.py input/ref/solution.jsonl input/res/predictions.jsonl output/scores.json
```

### Understanding the Metrics

The evaluation script calculates:

1. **Spearman Correlation**: Measures how well predicted scores correlate with human averages
   - Range: -1 to 1 (1 = perfect correlation)
   - Primary metric for ranking systems

2. **Accuracy Within Standard Deviation**: Percentage of predictions within ±1 std dev of human average
   - Accounts for human disagreement
   - More lenient metric for samples with high variance

3. **Mean Absolute Error (MAE)**: Average absolute difference from human average
   - Lower is better
   - Interpretable in terms of rating scale (1-5)

4. **Root Mean Squared Error (RMSE)**: Square root of average squared differences
   - Penalizes larger errors more heavily

5. **Exact Accuracy**: Percentage of exact matches with rounded human average
   - Strict metric, typically lower

## Output Format

### Predictions File (`input/res/predictions.jsonl`)

Each line contains one prediction:

```json
{"id": "0", "prediction": 3}
{"id": "1", "prediction": 4}
{"id": "2", "prediction": 2}
```

- `id`: Must match the sample ID from the gold data
- `prediction`: Integer from 1 to 5

### Scores File (`output/scores.json`)

Contains evaluation metrics:

```json
{
  "spearman_correlation": 0.6234,
  "spearman_p_value": 0.000001,
  "accuracy_within_stdev": 0.7845,
  "mean_absolute_error": 0.8123,
  "root_mean_squared_error": 1.0234,
  "exact_accuracy": 0.4521,
  "num_samples": 1000,
  "num_correct_within_stdev": 784
}
```

## Tips for Better Performance

1. **Use GPT-4o or GPT-4**: Better at nuanced understanding of word senses
2. **Lower Temperature**: Use 0.1-0.3 for more consistent predictions
3. **Rate Limits**: The script includes delays to avoid API rate limits
4. **Cost Considerations**: 
   - Test with `--max-samples` first
   - Each sample costs 1-2 API calls
   - dev.json has ~2457 samples (6 per story setup)
5. **Monitor Progress**: The script uses tqdm to show progress bars

## Workflow Example

Complete workflow from start to evaluation:

```bash
# 1. Test on small sample
python baseline.py --data dev.json --output input/res/test_predictions.jsonl --max-samples 20

# 2. Evaluate test predictions
python scoring.py dev.json input/res/test_predictions.jsonl output/test_scores.json

# 3. If satisfied, run on full dataset
python baseline.py --data dev.json --output input/res/predictions.jsonl --model gpt-4o

# 4. Final evaluation
python scoring.py dev.json input/res/predictions.jsonl output/scores.json

# 5. View results
cat output/scores.json
```

## Troubleshooting

### API Key Issues

```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set it if missing
export OPENAI_API_KEY='your-key'
```

### Rate Limit Errors

If you hit rate limits, the script will use fallback ratings. To avoid:
- Use a lower tier model (gpt-3.5-turbo)
- Add longer delays in the code
- Process in smaller batches

### Parsing Errors

If the model returns unexpected format, the script defaults to rating 3. Check:
- Model choice (gpt-4o recommended)
- Temperature setting (lower is more consistent)

## Advanced: Modifying the Prompt

The prompt engineering is in `baseline.py` in the `create_prompt()` function. You can experiment with:
- Different instruction formats
- More or fewer examples
- Emphasis on different aspects (narrative vs. linguistic)

## Citation

When using this baseline, please cite:

```
SemEval 2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding
```

## Support

For issues or questions, please refer to the task description or contact the organizers.

