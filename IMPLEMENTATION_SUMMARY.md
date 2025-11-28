# Implementation Summary: AmbiStory WSD Baseline System

This document summarizes the complete baseline system implementation for the SemEval 2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding.

## What Was Built

### Core Components

#### 1. **baseline.py** - Simple OpenAI Baseline
A clean, straightforward baseline using OpenAI's GPT models.

**Features:**
- Structured prompts that present the story context and word sense
- 5-point plausibility rating scale
- Rate limiting to avoid API throttling
- Error handling with fallback to middle rating
- Progress tracking with tqdm
- Command-line interface for easy configuration

**Key Parameters:**
- `--model`: Choice of GPT-4o, GPT-4-turbo, or GPT-3.5-turbo
- `--temperature`: Control randomness (0.0-1.0)
- `--max-samples`: Test on subset before full run
- `--data`: Input data file (train.json or dev.json)
- `--output`: Output predictions file

**Prompt Design:**
The prompt includes:
- Clear task description and rating scale
- Precontext (3 sentences)
- Ambiguous sentence with target homonym
- Optional ending
- Word sense definition and example
- Structured instructions for reasoning

#### 2. **baseline_reasoning.py** - Enhanced Multi-Provider Baseline
More sophisticated baseline with support for multiple LLM providers.

**Features:**
- Support for OpenAI (GPT-4o, GPT-4-turbo, GPT-3.5-turbo)
- Support for Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
- Enhanced prompts that encourage reasoning
- Flexible rating extraction from model responses
- Provider abstraction for easy extension
- Both verbose and concise prompt modes

**Advantages:**
- Can compare performance across different models
- More robust rating extraction
- Better prompts for reasoning models
- Extensible to new providers

#### 3. **scoring.py** - Comprehensive Evaluation Script
Evaluates predictions against gold standard data.

**Metrics Calculated:**
1. **Spearman Correlation** - Primary ranking metric
2. **Accuracy Within Standard Deviation** - Accounts for human disagreement
3. **Mean Absolute Error (MAE)** - Average prediction error
4. **Root Mean Squared Error (RMSE)** - Penalizes large errors
5. **Exact Accuracy** - Strict exact match metric

**Output:**
- Detailed console output with formatted results
- JSON file with all metrics
- Statistical significance (p-values)
- Sample counts and success rates

**Usage:**
```bash
python scoring.py gold_file.json predictions.jsonl output_scores.json
```

### Utility Scripts

#### 4. **analyze_data.py** - Dataset Analysis Tool
Comprehensive analysis of the AmbiStory dataset.

**Analyses Provided:**
- Basic statistics (total samples, unique stories)
- Homonym frequency and distribution
- Rating statistics (mean, median, std dev)
- Agreement levels (high/low disagreement)
- Rating distribution histograms
- Story structure analysis (with/without endings)
- Word sense statistics
- Nonsensical annotation tracking
- Dataset comparison (train vs dev)
- Example samples display
- Challenging samples (high disagreement)

**Usage:**
```bash
# Basic analysis
python analyze_data.py

# With comparison and challenging samples
python analyze_data.py --compare --show-challenging
```

#### 5. **visualize_results.py** - Prediction Analysis Tool
Analyzes model predictions and provides detailed error analysis.

**Analyses Provided:**
- Overall performance metrics
- Error distribution by threshold
- Performance by gold rating (1-5)
- Performance by agreement level
- Best/worst performing homonyms
- Examples of large errors
- Confusion matrix
- Per-class accuracy

**Usage:**
```bash
python visualize_results.py gold.json predictions.jsonl --confusion-matrix
```

#### 6. **run_baseline_test.sh** - Quick Test Script
Automated test script for rapid validation.

**What It Does:**
1. Checks for API key
2. Creates directory structure
3. Runs baseline on 20 samples
4. Evaluates predictions
5. Displays results

**Usage:**
```bash
chmod +x run_baseline_test.sh
./run_baseline_test.sh
```

### Documentation

#### 7. **README.md** - Main Documentation
Comprehensive project documentation including:
- Task overview and description
- Quick start guide
- File structure
- Usage examples for all scripts
- Expected performance benchmarks
- Troubleshooting guide
- Advanced usage tips

#### 8. **USAGE.md** - Detailed Usage Guide
In-depth guide covering:
- Setup and installation
- Running the baseline models
- Evaluation procedures
- Output format specifications
- Tips for better performance
- Complete workflow examples
- Troubleshooting common issues

#### 9. **IMPLEMENTATION_SUMMARY.md** - This Document
Technical overview of the implementation.

### Configuration Files

#### 10. **requirements.txt** - Python Dependencies
All required packages with versions:
- scipy (for statistical tests)
- openai (for GPT models)
- anthropic (optional, for Claude)
- tqdm (for progress bars)
- numpy (for numerical operations)

#### 11. **.gitignore** - Git Configuration
Properly configured to ignore:
- Python cache files
- Virtual environments
- API keys and secrets
- Generated predictions and outputs
- IDE-specific files
- OS-specific files

## Architecture Overview

```
User Input (Story Context)
         ↓
    Prompt Engineering
         ↓
    LLM API Call
    (OpenAI/Anthropic)
         ↓
    Response Parsing
         ↓
    Rating Extraction (1-5)
         ↓
    Predictions (JSONL)
         ↓
    Evaluation Script
         ↓
    Metrics & Analysis
```

## Prompt Engineering Strategy

### Key Design Decisions

1. **Structured Format**: Clear sections for each story component
2. **Explicit Scale**: Detailed 5-point scale with descriptions
3. **Context Emphasis**: All story elements presented together
4. **Example Grounding**: Word sense definition + example sentence
5. **Reasoning Instructions**: Step-by-step evaluation process
6. **Format Control**: Explicit output format to ensure consistent parsing

### Prompt Components

```
1. Task Description
   ↓
2. Rating Scale (1-5 with descriptions)
   ↓
3. Precontext (narrative setup)
   ↓
4. Ambiguous Sentence (with target word)
   ↓
5. Ending (if present)
   ↓
6. Word Sense Information
   ↓
7. Reasoning Instructions
   ↓
8. Output Format Specification
```

## Evaluation Pipeline

```
Gold Data (JSON)       Predictions (JSONL)
       ↓                        ↓
       └────────────┬───────────┘
                    ↓
           Load & Parse Data
                    ↓
           Match by Sample ID
                    ↓
           Calculate Metrics:
           - Spearman Correlation
           - Accuracy (within stdev)
           - MAE, RMSE
           - Exact Accuracy
                    ↓
           Generate Report
                    ↓
      Console Output + JSON File
```

## Data Flow

### Input Data Structure (JSON)
```json
{
  "id": {
    "homonym": "word",
    "judged_meaning": "definition",
    "precontext": "three sentences",
    "sentence": "ambiguous sentence",
    "ending": "optional ending",
    "choices": [5, 4, 4, 3, 5],
    "average": 4.2,
    "stdev": 0.8,
    "sample_id": "unique_id",
    "example_sentence": "example usage"
  }
}
```

### Output Predictions Structure (JSONL)
```json
{"id": "0", "prediction": 4}
{"id": "1", "prediction": 3}
{"id": "2", "prediction": 5}
```

### Scores Output Structure (JSON)
```json
{
  "spearman_correlation": 0.65,
  "spearman_p_value": 0.000001,
  "accuracy_within_stdev": 0.78,
  "mean_absolute_error": 0.85,
  "root_mean_squared_error": 1.12,
  "exact_accuracy": 0.45,
  "num_samples": 2457,
  "num_correct_within_stdev": 1916
}
```

## Usage Workflows

### 1. Quick Test (2-3 minutes)
```bash
export OPENAI_API_KEY='your-key'
./run_baseline_test.sh
```

### 2. Full Development Set Evaluation (30-60 minutes)
```bash
# Run baseline
python baseline.py --data dev.json --output input/res/predictions.jsonl

# Evaluate
python scoring.py dev.json input/res/predictions.jsonl output/scores.json

# Analyze
python visualize_results.py dev.json input/res/predictions.jsonl --confusion-matrix
```

### 3. Model Comparison
```bash
# GPT-4o
python baseline.py --model gpt-4o --output input/res/gpt4o_pred.jsonl

# GPT-3.5-turbo
python baseline.py --model gpt-3.5-turbo --output input/res/gpt35_pred.jsonl

# Claude (if available)
python baseline_reasoning.py --provider anthropic --model claude-3-5-sonnet-20241022 --output input/res/claude_pred.jsonl

# Compare results
python scoring.py dev.json input/res/gpt4o_pred.jsonl output/gpt4o_scores.json
python scoring.py dev.json input/res/gpt35_pred.jsonl output/gpt35_scores.json
python scoring.py dev.json input/res/claude_pred.jsonl output/claude_scores.json
```

### 4. Data Exploration
```bash
# Analyze dataset
python analyze_data.py --compare --show-challenging

# Explore examples
python analyze_data.py --examples 10
```

## Expected Performance

Based on the task design and baseline implementation:

### Baseline Models (GPT-4o/Claude)
- **Spearman Correlation**: 0.50-0.70
- **Accuracy Within Std Dev**: 0.65-0.80
- **Mean Absolute Error**: 0.7-1.1

### Weaker Models (GPT-3.5-turbo)
- **Spearman Correlation**: 0.35-0.55
- **Accuracy Within Std Dev**: 0.55-0.70
- **Mean Absolute Error**: 0.9-1.3

### State-of-the-Art Goal
- **Spearman Correlation**: >0.75
- **Accuracy Within Std Dev**: >0.85

## Extensibility

The system is designed to be easily extensible:

### Adding New LLM Providers
1. Create a new Rater class in `baseline_reasoning.py`
2. Implement `get_rating()` method
3. Add to `get_rater()` factory function

### Modifying Prompts
- Edit `create_prompt()` in `baseline.py` or `baseline_reasoning.py`
- Test with `--max-samples` to validate changes
- Compare performance with original

### Adding New Metrics
- Edit `calculate_metrics()` in `scoring.py`
- Add new calculations
- Update output formatting

### Custom Analysis
- Use `analyze_data.py` as template
- Add new analysis functions
- Integrate into main() workflow

## Cost Estimation

### GPT-4o Pricing (approximate)
- Input: ~500 tokens/sample
- Output: ~50 tokens/sample
- Cost: ~$0.01-0.02 per sample
- **Full dev set (2,457 samples): ~$25-50**

### GPT-3.5-turbo Pricing
- Much cheaper: ~$2-5 for full dev set
- Lower accuracy

### Claude 3.5 Sonnet
- Similar cost to GPT-4o
- Comparable performance

## Limitations and Future Work

### Current Limitations
1. No fine-tuning (pure prompting approach)
2. No ensemble methods
3. No calibration techniques
4. Single prediction per sample (no uncertainty estimation)
5. No use of training data beyond exploration

### Potential Improvements
1. **Few-shot Learning**: Include examples in prompts
2. **Ensemble**: Combine multiple models
3. **Calibration**: Adjust predictions based on dev set
4. **Fine-tuning**: Train on training set
5. **Uncertainty Estimation**: Multiple samples per prediction
6. **Prompt Optimization**: Systematic prompt engineering
7. **Feature Engineering**: Extract features for traditional ML

## Testing and Validation

### Validation Strategy
1. Small test run (20 samples) to verify setup
2. Medium test (100 samples) to estimate performance
3. Full dev set for final evaluation
4. Analysis of errors and edge cases
5. Comparison across models

### Quality Checks
- ✅ All predictions are integers 1-5
- ✅ All sample IDs match gold data
- ✅ No missing predictions
- ✅ JSONL format is valid
- ✅ Metrics calculations are correct

## Conclusion

This implementation provides a complete, production-ready baseline system for the AmbiStory WSD task. It includes:

- **Two baseline models** with different complexity levels
- **Comprehensive evaluation** with multiple metrics
- **Extensive utilities** for data analysis and visualization
- **Clear documentation** for easy usage
- **Extensible architecture** for future improvements
- **Best practices** for reproducibility and maintainability

The system is ready to use for developing and evaluating WSD models, with clear paths for improvement and extension.

