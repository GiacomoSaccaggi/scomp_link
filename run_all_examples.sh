#!/bin/bash
# Run all scomp_link examples and save models to staging directory

echo "=========================================="
echo "Running All ScompLink Examples"
echo "=========================================="

# Create staging directory
mkdir -p staging
echo "✅ Created staging directory"

# Counter for successful runs
SUCCESS=0
TOTAL=0

# Function to run example
run_example() {
    local example=$1
    local name=$2
    TOTAL=$((TOTAL + 1))
    
    echo ""
    echo "=========================================="
    echo "Running: $name"
    echo "=========================================="
    
    if python3 "$example"; then
        echo "✅ $name completed successfully"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "❌ $name failed"
    fi
}

# Run core examples (no optional dependencies)
run_example "examples/example_01_numerical_small.py" "Example 01 - Numerical Small"
run_example "examples/example_02_numerical_medium_lasso.py" "Example 02 - Numerical Medium Lasso"
run_example "examples/example_03_numerical_mixed_features.py" "Example 03 - Numerical Mixed Features"
run_example "examples/example_04_classification_small.py" "Example 04 - Classification Small"
run_example "examples/example_05_classification_large.py" "Example 05 - Classification Large"
run_example "examples/example_06_clustering_known.py" "Example 06 - Clustering Known"
run_example "examples/example_07_clustering_unknown.py" "Example 07 - Clustering Unknown"
run_example "examples/example_08_numerical_very_large.py" "Example 08 - Numerical Very Large"

# Run NLP examples (requires NLP dependencies)
if python3 -c "import torch, transformers" 2>/dev/null; then
    echo ""
    echo "📝 NLP dependencies found, running text examples..."
    run_example "examples/example_09_text_classification.py" "Example 09 - Text Classification"
    run_example "examples/example_12_text_configuration.py" "Example 12 - Text Configuration"
    
    if python3 -c "from sentence_transformers import SentenceTransformer" 2>/dev/null; then
        run_example "examples/example_13_text_unsupervised.py" "Example 13 - Text Unsupervised"
    else
        echo "⚠️  Skipping Example 13 (requires sentence-transformers)"
    fi
else
    echo "⚠️  Skipping text examples (NLP dependencies not installed)"
    echo "   Install with: pip install .[nlp]"
fi

# Run image examples (requires image/CV dependencies)
if python3 -c "import tensorflow" 2>/dev/null; then
    echo ""
    echo "🖼️  Image/CV dependencies found, running image examples..."
    run_example "examples/example_10_image_classification.py" "Example 10 - Image Classification"
    run_example "examples/example_11_image_clustering.py" "Example 11 - Image Clustering"
else
    echo "⚠️  Skipping image examples (Image/CV dependencies not installed)"
    echo "   Install with: pip install .[img]"
fi

# Summary
echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "✅ Successful: $SUCCESS/$TOTAL"
echo "📂 Models saved in: ./staging/"
echo ""

# List saved models
echo "Saved models:"
find staging -name "*.pkl" -o -name "*.json" | sort

echo ""
echo "=========================================="
echo "Run complete!"
echo "=========================================="
