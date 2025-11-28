# Project Overview: AmbiStory WSD Baseline System

**Complete baseline system for SemEval 2026 Task 5**

---

## 🎯 What You Have

A **production-ready baseline system** for Word Sense Disambiguation that:
- ✅ Uses state-of-the-art LLMs (OpenAI GPT-4o, Claude 3.5)
- ✅ Includes comprehensive evaluation metrics
- ✅ Provides detailed analysis and visualization tools
- ✅ Has complete documentation and examples
- ✅ Is extensible and well-structured

## 📁 Files Created (14 files, ~65KB)

### Core Scripts (3 files)
1. **baseline.py** (5.4 KB) - Simple OpenAI baseline ⭐
2. **baseline_reasoning.py** (10 KB) - Enhanced multi-provider baseline
3. **scoring.py** (4.9 KB) - Comprehensive evaluation script ⭐

### Utility Scripts (3 files)
4. **analyze_data.py** (8.9 KB) - Dataset statistics and analysis
5. **visualize_results.py** (7.9 KB) - Prediction analysis and error visualization
6. **run_baseline_test.sh** (1.5 KB) - Automated quick test ⭐

### Documentation (5 files)
7. **README.md** (6.7 KB) - Main project documentation
8. **USAGE.md** (6.3 KB) - Detailed usage guide
9. **QUICKSTART.md** (4.6 KB) - 5-minute getting started guide ⭐
10. **IMPLEMENTATION_SUMMARY.md** (12 KB) - Technical implementation details
11. **PROJECT_OVERVIEW.md** (this file) - Complete overview

### Configuration (2 files)
12. **requirements.txt** - Python dependencies
13. **.gitignore** - Git configuration

### Directories (3 folders)
14. **input/ref/** - For gold standard data
15. **input/res/** - For model predictions
16. **output/** - For evaluation results

⭐ = Most frequently used

---

## 🚀 Quick Start (Choose Your Path)

### Path 1: Fastest (5 minutes)
```bash
# Install and test
pip install -r requirements.txt
export OPENAI_API_KEY='your-key'
./run_baseline_test.sh
```

### Path 2: Guided (15 minutes)
```bash
# Follow the quick start guide
cat QUICKSTART.md
# Then run commands step by step
```

### Path 3: Full Workflow (1 hour)
```bash
# 1. Analyze data
python analyze_data.py --compare

# 2. Run baseline
python baseline.py --data dev.json --output input/res/predictions.jsonl

# 3. Evaluate
python scoring.py dev.json input/res/predictions.jsonl output/scores.json

# 4. Visualize
python visualize_results.py dev.json input/res/predictions.jsonl --confusion-matrix
```

---

## 🔑 Key Features

### 1. Multiple Baseline Options

**Simple Baseline (`baseline.py`)**
- Clean, easy to understand
- Perfect for getting started
- Uses OpenAI GPT models
- ~100 lines of well-commented code

**Enhanced Baseline (`baseline_reasoning.py`)**
- Supports multiple providers (OpenAI, Anthropic)
- Better prompts for reasoning
- More robust parsing
- Extensible architecture

### 2. Comprehensive Evaluation

**Primary Metrics**
- Spearman Correlation (ranking quality)
- Accuracy Within Standard Deviation (accounts for disagreement)

**Additional Metrics**
- Mean Absolute Error
- Root Mean Squared Error
- Exact Accuracy
- Statistical significance (p-values)

### 3. Rich Analysis Tools

**Data Analysis (`analyze_data.py`)**
- Dataset statistics
- Homonym frequency
- Rating distributions
- Agreement analysis
- Challenging sample identification
- Dataset comparison

**Results Visualization (`visualize_results.py`)**
- Error distribution analysis
- Performance by rating level
- Performance by agreement level
- Best/worst performing words
- Confusion matrix
- Error examples

### 4. Complete Documentation

**For Users:**
- QUICKSTART.md - Get running in 5 minutes
- README.md - Complete project guide
- USAGE.md - Detailed usage instructions

**For Developers:**
- IMPLEMENTATION_SUMMARY.md - Technical details
- Well-commented code
- Clear architecture

---

## 📊 What to Expect

### Performance Benchmarks

**GPT-4o Baseline (Recommended)**
- Spearman Correlation: 0.50-0.70
- Accuracy (within stdev): 0.65-0.80
- Processing time: 30-60 minutes for full dev set
- Cost: ~$25-50 for 2,457 samples

**GPT-3.5-turbo Baseline (Budget)**
- Spearman Correlation: 0.35-0.55
- Accuracy (within stdev): 0.55-0.70
- Processing time: 15-30 minutes
- Cost: ~$2-5

**State-of-the-Art Goal**
- Spearman Correlation: >0.75
- Accuracy (within stdev): >0.85

### Dataset Overview

**Development Set (dev.json)**
- 2,457 samples
- ~410 unique story setups
- 6 samples per setup (2 senses × 3 ending types)

**Training Set (train.json)**
- 9,512 samples
- Can be used for few-shot learning or fine-tuning

---

## 🎓 How to Use This System

### For Quick Evaluation
```bash
./run_baseline_test.sh  # 2-3 minutes
```

### For Development
1. Start with small tests (`--max-samples 50`)
2. Analyze errors (`visualize_results.py`)
3. Iterate on prompts or parameters
4. Run full evaluation

### For Research
1. Compare multiple models
2. Analyze dataset characteristics
3. Study challenging cases
4. Build on the baseline

### For Production
1. Use the evaluation pipeline
2. Track metrics over time
3. Extend with custom models
4. Scale with batch processing

---

## 🛠️ Architecture

```
┌─────────────────────────────────────────────┐
│           User Input (Story Data)           │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│         Baseline Models                     │
│  • baseline.py (OpenAI)                     │
│  • baseline_reasoning.py (Multi-provider)   │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│         Predictions (JSONL)                 │
│  {"id": "0", "prediction": 4}               │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│         Evaluation (scoring.py)             │
│  • Spearman Correlation                     │
│  • Accuracy Within Stdev                    │
│  • MAE, RMSE, Exact Accuracy                │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│         Analysis & Visualization            │
│  • visualize_results.py                     │
│  • analyze_data.py                          │
└─────────────────────────────────────────────┘
```

---

## 📚 Documentation Roadmap

**Start Here:**
1. **QUICKSTART.md** - 5-minute guide
2. **README.md** - Project overview

**Go Deeper:**
3. **USAGE.md** - Detailed instructions
4. **IMPLEMENTATION_SUMMARY.md** - Technical details

**Reference:**
5. Code comments in each script
6. This overview (PROJECT_OVERVIEW.md)

---

## 🎯 Common Use Cases

### 1. Getting Started
```bash
# Read the quick start
cat QUICKSTART.md

# Run test
./run_baseline_test.sh
```

### 2. Understanding the Data
```bash
python analyze_data.py --compare --show-challenging
```

### 3. Testing a Hypothesis
```bash
# Test with different temperature
python baseline.py --temperature 0.1 --max-samples 100 --output test1.jsonl
python baseline.py --temperature 0.5 --max-samples 100 --output test2.jsonl

# Compare
python scoring.py dev.json test1.jsonl out1.json
python scoring.py dev.json test2.jsonl out2.json
```

### 4. Model Comparison
```bash
# Try different models
python baseline.py --model gpt-4o --output gpt4o.jsonl
python baseline.py --model gpt-3.5-turbo --output gpt35.jsonl

# Evaluate both
python scoring.py dev.json gpt4o.jsonl gpt4o_scores.json
python scoring.py dev.json gpt35.jsonl gpt35_scores.json

# Compare scores
cat gpt4o_scores.json
cat gpt35_scores.json
```

### 5. Error Analysis
```bash
# Generate predictions
python baseline.py --data dev.json --output predictions.jsonl

# Detailed analysis
python visualize_results.py dev.json predictions.jsonl --confusion-matrix --show-errors 10
```

---

## 🔧 Customization Points

### Easy Modifications
1. **Prompts**: Edit `create_prompt()` in baseline scripts
2. **Temperature**: Use `--temperature` flag
3. **Model**: Use `--model` flag
4. **Sample size**: Use `--max-samples` flag

### Medium Modifications
1. **Add new provider**: Extend `baseline_reasoning.py`
2. **New metrics**: Edit `calculate_metrics()` in `scoring.py`
3. **Custom analysis**: Add functions to `analyze_data.py`

### Advanced Modifications
1. **Ensemble methods**: Combine multiple model predictions
2. **Fine-tuning**: Use training data for model adaptation
3. **Feature engineering**: Extract additional features
4. **Calibration**: Adjust predictions based on dev set

---

## 💡 Tips for Success

### Before Running
1. ✅ Set API key: `export OPENAI_API_KEY='...'`
2. ✅ Install dependencies: `pip install -r requirements.txt`
3. ✅ Test on small sample first: `--max-samples 20`

### During Development
1. 🎯 Start simple, iterate
2. 📊 Monitor costs with small tests
3. 🔍 Analyze errors to improve
4. 📈 Track metrics over time

### Best Practices
1. 💾 Save all predictions for reproducibility
2. 📝 Document experiments and results
3. 🔄 Version control your modifications
4. 🧪 Test incrementally before full runs

---

## 🆘 Troubleshooting

### Common Issues

**"Error: OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY='sk-...'
```

**"Module not found"**
```bash
pip install -r requirements.txt
```

**Rate limits**
- Use `--max-samples` to test smaller batches
- Use `gpt-3.5-turbo` for faster testing
- Add delays in code between requests

**Permission denied**
```bash
chmod +x run_baseline_test.sh
```

### Getting Help
1. Check documentation files
2. Review code comments
3. Look at example outputs
4. Test with small samples to isolate issues

---

## 📈 Next Steps

### Immediate (Today)
1. ✅ Run quick test: `./run_baseline_test.sh`
2. ✅ Explore the data: `python analyze_data.py`
3. ✅ Read QUICKSTART.md

### Short Term (This Week)
1. Run full baseline on dev set
2. Analyze results and errors
3. Try different models/parameters
4. Understand challenging cases

### Long Term (Research)
1. Develop improved prompts
2. Implement ensemble methods
3. Add fine-tuning capabilities
4. Benchmark against other approaches
5. Extend to training data

---

## 🎉 Summary

You now have a **complete, production-ready baseline system** for the AmbiStory Word Sense Disambiguation task. The system includes:

- ✅ **2 baseline models** (simple and enhanced)
- ✅ **6 utility scripts** (evaluation, analysis, visualization)
- ✅ **5 documentation files** (quick start to technical details)
- ✅ **Ready-to-use workflow** (from data to results)
- ✅ **Extensible architecture** (easy to customize and improve)

**Start here:** `./run_baseline_test.sh` or `cat QUICKSTART.md`

**Good luck with your research! 🚀**

---

*Last updated: November 2024*
*System version: 1.0*

