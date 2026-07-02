# Workflow Patterns

## Pattern 1: CSV → Trained Model → Deploy

```bash
# Understand the data
scomp-link describe --data customers.csv --format table

# Engineer features
scomp-link engineer --data customers.csv --target churn --interactions --log-transform --output features.csv

# Tune and save best model
scomp-link tune --data features.csv --target churn --task classification --method optuna --n-trials 100 --save-artifact churn_model.scomp

# Validate
scomp-link validate --artifact churn_model.scomp --data test.csv --target churn --report validation.html

# Deploy
scomp-link serve --artifact churn_model.scomp --port 8080
```

**Prediction from deployed model:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"age": 35, "income": 50000, "tenure": 24}]}'
```

## Pattern 2: Monitor Production Data

```bash
# Save reference data during training
cp train.csv reference_data.csv

# Weekly monitoring job
scomp-link monitor --reference reference_data.csv --current this_week.csv \
  --artifact model.scomp --target y --output weekly_report.html

# Quick drift check (no model needed)
scomp-link drift --reference reference_data.csv --current this_week.csv --plot drift.html

# Anomaly scan on incoming data
scomp-link anomaly --data this_week.csv --methods iforest,lof --contamination 0.03 --output anomalies.csv
```

## Pattern 3: Create Analytical Dashboard (Python)

```python
from scomp_link.utils.report_html import ScompLinkHTMLReport
from scomp_link.utils.plotly_utils import histogram, barchart, linechart, area_chart
from scomp_link.utils.highcharts import streamgraphs, calendar_heatmap
from scomp_link.utils.rawgraphs import treemap, sankey_diagram, sunburst
import pandas as pd

df = pd.read_csv('sales_data.csv')

report = ScompLinkHTMLReport(title='Sales Dashboard Q4 2025')

# Section 1: KPIs
report.open_section("Key Metrics")
kpi_df = pd.DataFrame([{
    'Total Revenue': f"${df['revenue'].sum():,.0f}",
    'Avg Order': f"${df['revenue'].mean():,.2f}",
    'Customers': df['customer_id'].nunique(),
}])
report.add_dataframe(kpi_df, "KPIs")
report.close_section()

# Section 2: Trends (Highcharts streamgraph)
report.open_section("Revenue by Category Over Time")
monthly = df.groupby(['month', 'category'])['revenue'].sum().unstack(fill_value=0)
dates = monthly.index.tolist()
series = {col: monthly[col].tolist() for col in monthly.columns}
html_stream = streamgraphs("Revenue by Category", dates, series, area=True)
report.html_report += html_stream
report.close_section()

# Section 3: Distribution (Plotly)
report.open_section("Order Value Distribution")
fig = histogram(df['revenue'].values, "Order Value ($)")
report.add_graph_to_report(fig, "Revenue Distribution")
report.close_section()

# Section 4: Hierarchy (RAWGraphs)
report.open_section("Revenue Breakdown")
cat_rev = df.groupby('category')['revenue'].sum()
svg = treemap(cat_rev.index.tolist(), [''] * len(cat_rev), cat_rev.values.tolist(), "Revenue Treemap")
report.add_rawgraphs_to_report(svg, "Treemap")
report.close_section()

# Section 5: Flow (RAWGraphs)
report.open_section("Customer Journey")
flows = [
    {'source': 'Homepage', 'target': 'Product', 'value': 1000},
    {'source': 'Product', 'target': 'Cart', 'value': 400},
    {'source': 'Cart', 'target': 'Checkout', 'value': 300},
    {'source': 'Checkout', 'target': 'Purchase', 'value': 250},
]
svg_sankey = sankey_diagram(
    ['Homepage', 'Product', 'Cart', 'Checkout', 'Purchase'],
    [{'source': f['source'], 'target': f['target'], 'value': f['value']} for f in flows],
    "Customer Flow"
)
report.add_rawgraphs_to_report(svg_sankey, "Sankey")
report.close_section()

report.save_html('sales_dashboard.html')
```

## Pattern 4: Compare Multiple Approaches

```bash
# Approach 1: Simple model
scomp-link run --data train.csv --target y --task regression --save-artifact simple.scomp --silent

# Approach 2: With feature engineering
scomp-link engineer --data train.csv --target y --interactions --log-transform --output eng.csv
scomp-link run --data eng.csv --target y --task regression --save-artifact engineered.scomp --silent

# Approach 3: Tuned
scomp-link tune --data eng.csv --target y --task regression --method optuna --n-trials 50 --save-artifact tuned.scomp --silent

# Compare all three
scomp-link compare --artifacts simple.scomp engineered.scomp tuned.scomp --plot comparison.html
```

## Pattern 5: Text Classification Pipeline

```bash
# Profile text data
scomp-link describe --data tickets.csv

# Quick TF-IDF approach (fast)
scomp-link text --data tickets.csv --text-col message --target category --method tfidf --save-artifact tfidf_model.scomp

# Validate
scomp-link validate --artifact tfidf_model.scomp --data test_tickets.csv --target category --format table
```

## Pattern 6: YAML-Driven Pipeline

```yaml
# pipeline.yaml — full automated workflow
data: data/train.csv
target: price
task: regression
name: house_price_predictor

engineer:
  interactions: true
  log_transform: true
  date_features: true
  target_encode: true

tune:
  method: optuna
  n_trials: 100

validate:
  test_data: data/test.csv
  report: reports/validation.html

save: models/house_price_v2.scomp
```

```bash
scomp-link pipeline --config pipeline.yaml
```

## Troubleshooting

### Model performs poorly
1. Check data quality: `scomp-link quality --data train.csv`
2. Look for drift: `scomp-link drift --reference train.csv --current test.csv`
3. Try feature engineering: `scomp-link engineer --data train.csv --target y --interactions --log-transform`
4. Tune more: increase `--n-trials`
5. Check fairness: `scomp-link fairness --data preds.csv --target y_true --predicted y_pred --sensitive group`

### Command fails with error
- `unsupported file format` → use .csv, .tsv, or .parquet files
- `target column not found` → check column name with `scomp-link describe --data file.csv`
- `artifact not found` → use absolute path or check working directory
- `could not convert string to float` → categoricals need encoding, add `--engineer` flag
- `Memory error` → reduce dataset size or use `--methods iforest,lof` (skip deep learning methods)

### Report is empty or broken
- Check that `open_section()` has matching `close_section()`
- Highcharts content must be appended with `report.html_report += html_string`
- Plotly figures must be passed to `add_graph_to_report(fig, title)`
- RAWGraphs SVGs go to `add_rawgraphs_to_report(svg, title)`
