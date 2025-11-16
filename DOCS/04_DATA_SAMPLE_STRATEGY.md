# Sample Data Strategy & Usage Plan

## 1. Dataset Overview

The project ships with a curated sample dataset located at `data/sample_sales.csv`. It represents a mid-sized consumer goods company operating across Bangladesh’s major regions. Each row is a transactional record with the following fields:

| Column | Description |
| --- | --- |
| `date` | Transaction date (daily granularity) |
| `product_category` | One of: Electronics, Furniture, Industrial, FMCG |
| `product_name` | Specific SKU identifier |
| `region` | Dhaka, Chattogram, Khulna, Sylhet, Rajshahi |
| `customer_segment` | Enterprise, SME, Retail |
| `sales_channel` | Direct, Distributor, Online |
| `quantity` | Units sold |
| `revenue` | Monetary value of the transaction |

## 2. Why We Chose This Dataset

1. **Realistic Scale** – 1,163 rows spanning the 2025 calendar year provide enough density for rolling analytics, anomaly detection, and forecasting while remaining lightweight for demos. The mix of regions and categories reflects Akij Group’s operating footprint.
2. **Representative Segmentation** – Regions, categories, segments, and channels reflect the client’s operational structure, ensuring insights feel familiar.
3. **Rich Variability** – Embedded patterns (seasonal lifts across 12 months, abrupt dips, channel differences) intentionally trigger diagnostic and prescriptive logic, showcasing the system’s full capabilities.
4. **Manageable Size** – Fits comfortably in memory for pandas operations; avoids external database setup while keeping scenarios realistic.
5. **Privacy Friendly** – Synthetic but statistically aligned with real sales behavior, so it’s safe to share and demo without exposing confidential records.

## 3. Execution Plan Using the Sample Data

1. **Initialization**
   - Streamlit/CLI loads `sample_sales.csv` via `DataLoader`, validating required columns.
   - Data is cached in `CoordinatorAgent.data` for reuse.
2. **Descriptive Stage**
   - Provides totals, AOV, top categories, region/channel breakdowns. Demonstrates baseline KPI visibility for management.
3. **Diagnostic Stage**
   - Detects engineered revenue dips (e.g., Electronics in Dhaka during early February). Highlights root causes meaningful to Akij (e.g., FMCG decline in Khulna).
4. **Predictive Stage**
   - Generates 30-day forecasts per region to illustrate future planning. Dataset includes upward and downward trends to validate accuracy.
5. **Prescriptive Stage**
   - Recommendations combine diagnostics & forecasting (e.g., “Increase promotions in Khulna” due to repeated dips). The sample’s data quirks ensure non-trivial advice.
6. **Charts & Narratives**
   - Visualizations (bar, line, heatmap) use aggregated sample data to show multi-modal insights (region severity, anomaly heatmap, forecast curve) across a full year.

## 4. Client-Facing Value

- **Demo Readiness**: Out-of-the-box experience; no extra data loading required. Perfect for Akij presentations or interviews.
- **Scenario Coverage**: Supports full question spectrum—simple KPIs, deep diagnostics, forecasting, and prescriptive “next steps”.
- **Extensibility**: Clients can replace `sample_sales.csv` with their actual data once comfortable. Schema validation ensures compatibility.

## 5. Next Steps After Client Adoption

1. Replace sample data with production exports (CSV, database connection, or API ingestion).
2. Update configuration if new columns appear (e.g., additional regions or channels).
3. Re-run analytics to generate client-specific insights; fine-tune thresholds (dip sensitivity, forecasting horizon) as needed.
4. Optionally automate data refresh with n8n workflow (provided in repo).

---
*Updated: November 2025*
