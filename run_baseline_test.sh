#!/bin/bash

# Quick test script for the baseline model
# This runs a small test to verify everything is working

echo "=========================================="
echo "AmbiStory WSD Baseline Model - Quick Test"
echo "=========================================="
echo ""

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

echo "✓ OpenAI API key found"
echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p input/ref input/res output
echo "✓ Directories created"
echo ""

# Run baseline on small sample
echo "Running baseline model on 20 samples (this may take 1-2 minutes)..."
python baseline.py \
    --data dev.json \
    --output input/res/test_predictions.jsonl \
    --model gpt-4o \
    --temperature 0.3 \
    --max-samples 20

echo ""
echo "✓ Predictions generated"
echo ""

# Run evaluation
echo "Evaluating predictions..."
python scoring.py \
    dev.json \
    input/res/test_predictions.jsonl \
    output/test_scores.json

echo ""
echo "✓ Evaluation complete"
echo ""

# Display results
echo "=========================================="
echo "TEST RESULTS:"
echo "=========================================="
cat output/test_scores.json
echo ""
echo "=========================================="
echo ""
echo "Test completed successfully!"
echo "To run on full dataset, use:"
echo "  python baseline.py --data dev.json --output input/res/predictions.jsonl"
echo ""

