#!/bin/bash
# Record a terminal demo with asciinema
# Install: brew install asciinema
# Run: ./scripts/record_demo.sh
# Upload: asciinema upload demo.cast

set -e

echo "Recording scomp-link demo..."
echo "Press Ctrl+D when done"

asciinema rec --title "scomp-link: End-to-end ML in 30 seconds" docs/demo/demo.cast -c '
echo "# scomp-link — End-to-end ML toolkit"
echo ""
sleep 1

echo "# Step 1: Profile your data"
scomp-link describe --data tmp/train_demo.csv --format table
sleep 2

echo ""
echo "# Step 2: Train + tune a model"
scomp-link tune --data tmp/train_demo.csv --target price --task regression --method optuna --n-trials 10 --save-artifact tmp/model.scomp --silent
echo "✅ Model trained and saved!"
sleep 2

echo ""
echo "# Step 3: Validate on test data"
scomp-link validate --artifact tmp/model.scomp --data tmp/test_demo.csv --target price --format table --silent
sleep 2

echo ""
echo "# Step 4: Generate a report with code + diff"
echo "$ python -c \"..."
python3 -c "
from scomp_link import CodeStep, DiffStep, SectionStep, SaveStep
from scomp_link.utils.report_html import ScompLinkHTMLReport
report = ScompLinkHTMLReport(title=\"ML Pipeline Report\")
(
    SectionStep(\"Training Code\")
    >> CodeStep(\"scomp-link tune --data train.csv --target price --task regression --n-trials 10\", \"bash\", \"CLI Command\", \"Best R²: 0.84\")
    >> SectionStep(\"Config Changes\")
    >> DiffStep(\"n_trials: 10\\ntest_size: 0.2\", \"n_trials: 50\\ntest_size: 0.3\\nmethod: optuna\", \"yaml\", \"Hyperparameter Update\", \"v1\", \"v2\")
    >> SaveStep(\"docs/demo/report.html\")
).run(report)
print(\"✅ Report saved: docs/demo/report.html\")
"
sleep 2

echo ""
echo "# 🎉 Done! From CSV to model + report in 30 seconds."
echo "# pip install scomp-link"
sleep 3
'
