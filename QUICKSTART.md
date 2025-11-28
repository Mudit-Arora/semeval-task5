# AmbiStory WSD - Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- OpenAI API key

## Setup (1 minute)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
export OPENAI_API_KEY='your-api-key-here'
```

## Quick Test (2 minutes)

Run the automated test on 20 samples:

```bash
./run_baseline_test.sh
```

This will:
- ✅ Check your setup
- ✅ Run the baseline model
- ✅ Evaluate predictions
- ✅ Show results

## Full Run (30-60 minutes)

Process the entire development set:

```bash
# Run baseline
python baseline.py --data dev.json --output input/res/predictions.jsonl

# Evaluate
python scoring.py dev.json input/res/predictions.jsonl output/scores.json

# View results
cat output/scores.json
```

## Analyze Results

```bash
# Detailed prediction analysis
python visualize_results.py dev.json input/res/predictions.jsonl --confusion-matrix

# Dataset statistics
python analyze_data.py --compare
```

## Common Commands

### Test on Small Sample
```bash
python baseline.py --data dev.json --max-samples 50 --output input/res/test.jsonl
```

### Use Different Model
```bash
# GPT-4o (recommended)
python baseline.py --model gpt-4o

# GPT-3.5-turbo (faster, cheaper)
python baseline.py --model gpt-3.5-turbo

# Claude (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY='your-key'
python baseline_reasoning.py --provider anthropic --model claude-3-5-sonnet-20241022
```

### Lower Temperature (More Consistent)
```bash
python baseline.py --temperature 0.1
```

## Understanding Output

### Predictions File Format
```jsonl
{"id": "0", "prediction": 4}
{"id": "1", "prediction": 3}
```

### Evaluation Metrics
- **Spearman Correlation** (0.0-1.0): How well rankings match humans
  - 0.60+ is good for baseline
  - 0.75+ is state-of-the-art goal
  
- **Accuracy Within Std Dev** (0.0-1.0): Predictions within human agreement range
  - 0.70+ is good for baseline
  - 0.85+ is state-of-the-art goal

- **Mean Absolute Error** (0.0-4.0): Average prediction error
  - <1.0 is good
  - <0.7 is excellent

## File Structure

```
semeval-task5/
├── baseline.py              # Simple OpenAI baseline ⭐
├── baseline_reasoning.py    # Multi-provider baseline
├── scoring.py               # Evaluation script ⭐
├── analyze_data.py          # Data analysis
├── visualize_results.py     # Results visualization
├── run_baseline_test.sh     # Quick test script ⭐
├── dev.json                 # Development data (2,457 samples)
├── train.json               # Training data (9,512 samples)
└── input/res/               # Your predictions go here
```

⭐ = Most commonly used

## Troubleshooting

### "Error: OPENAI_API_KEY not set"
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### Rate Limit Errors
- Use `--max-samples` to test on smaller subset first
- Add delays between requests (edit the delay parameter in code)
- Use cheaper model: `--model gpt-3.5-turbo`

### Permission Denied on Script
```bash
chmod +x run_baseline_test.sh
```

## Cost Estimates

- **Quick test (20 samples)**: ~$0.50
- **Small test (100 samples)**: ~$2-3
- **Full dev set (2,457 samples)**: ~$25-50 (GPT-4o)
- **Full dev set (GPT-3.5-turbo)**: ~$2-5

## Next Steps

1. ✅ Run quick test
2. ✅ Analyze a small subset
3. ✅ Review error examples
4. ✅ Tune parameters (temperature, prompt)
5. ✅ Run full evaluation
6. ✅ Compare different models

## Getting Help

- **Detailed usage**: See `USAGE.md`
- **Implementation details**: See `IMPLEMENTATION_SUMMARY.md`
- **Task description**: See `README.md`

## Example Workflow

```bash
# 1. Quick test to verify setup
./run_baseline_test.sh

# 2. Analyze the data
python analyze_data.py --examples 5

# 3. Test on 100 samples
python baseline.py --max-samples 100 --output input/res/test100.jsonl
python scoring.py dev.json input/res/test100.jsonl output/test100_scores.json

# 4. Review results
python visualize_results.py dev.json input/res/test100.jsonl

# 5. If satisfied, run full evaluation
python baseline.py --data dev.json --output input/res/predictions.jsonl
python scoring.py dev.json input/res/predictions.jsonl output/scores.json
```

## Tips for Better Results

1. **Use GPT-4o**: Best balance of quality and cost
2. **Lower temperature (0.1-0.3)**: More consistent predictions
3. **Test incrementally**: Start small, then scale up
4. **Analyze errors**: Use `visualize_results.py` to understand mistakes
5. **Compare models**: Try different models to find the best one

## Done!

You're ready to start developing your WSD system. Good luck! 🚀

