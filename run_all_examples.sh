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

# Run all examples
run_example "examples/example_01_numerical_small.py" "Example 01 - Numerical Small"
run_example "examples/example_02_numerical_medium_lasso.py" "Example 02 - Numerical Medium Lasso"
run_example "examples/example_03_numerical_mixed_features.py" "Example 03 - Numerical Mixed Features"
run_example "examples/example_04_classification_small.py" "Example 04 - Classification Small"
run_example "examples/example_05_classification_large.py" "Example 05 - Classification Large"
run_example "examples/example_06_clustering_known.py" "Example 06 - Clustering Known"
run_example "examples/example_07_clustering_unknown.py" "Example 07 - Clustering Unknown"
run_example "examples/example_08_numerical_very_large.py" "Example 08 - Numerical Very Large"
run_example "examples/example_09_text_classification.py" "Example 09 - Text Classification"
run_example "examples/example_10_image_classification.py" "Example 10 - Image Classification"
run_example "examples/example_11_image_clustering.py" "Example 11 - Image Clustering"
run_example "examples/example_12_text_configuration.py" "Example 12 - Text Configuration"
run_example "examples/example_13_text_unsupervised.py" "Example 13 - Text Unsupervised"
run_example "examples/example_14_ensemble_advanced_cv.py" "Example 14 - Ensemble & Advanced CV"
run_example "examples/example_15_anomaly_detection.py" "Example 15 - Anomaly Detection"
run_example "examples/example_16_ts_anomaly_detection.py" "Example 16 - Time Series Anomaly Detection"
run_example "examples/contrastive_text_example.py" "Contrastive Text Classification"

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
find staging -name "*.pkl" -o -name "*.json" -o -name "*.pt" | sort

echo ""
echo "=========================================="
echo "Run complete!"
echo "=========================================="
