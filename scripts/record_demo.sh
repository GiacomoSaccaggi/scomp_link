#!/bin/bash
# Record a terminal demo with asciinema
# Install: brew install asciinema
# Run: ./scripts/record_demo.sh
# Upload: asciinema upload demo.cast

set -e

echo "Recording scomp-link demo..."
echo "Press Ctrl+D when done"

asciinema rec --title "scomp-link: End-to-end ML in 30 seconds" demo.cast -c '
echo "# scomp-link — End-to-end ML toolkit"
echo ""
sleep 1

echo "# Step 1: Profile your data"
scomp-link describe --data examples/train_demo.csv --format table
sleep 2

echo ""
echo "# Step 2: Train + tune a model"
scomp-link tune --data examples/train_demo.csv --target price --task regression --method optuna --n-trials 10 --save-artifact demo_model.scomp --silent
echo "✅ Model trained and saved!"
sleep 2

echo ""
echo "# Step 3: Validate"
scomp-link validate --artifact demo_model.scomp --data examples/test_demo.csv --target price --format table --silent
sleep 2

echo ""
echo "# Step 4: Deploy as REST API"
echo "$ scomp-link serve --artifact demo_model.scomp --port 8080"
echo "Serving on http://localhost:8080/predict"
sleep 2

echo ""
echo "# 🎉 Done! From CSV to deployed model in 30 seconds."
echo "# pip install scomp-link"
sleep 3
'
