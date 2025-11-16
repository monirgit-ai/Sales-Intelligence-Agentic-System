# Analytics Deep Dive – How the System Thinks

## 1. Descriptive Analytics – “What Happened?”

**Goal:** Summarize historical performance and answer factual KPI questions.

**Data Inputs:**
- Columns: `date`, `product_category`, `region`, `sales_channel`, `customer_segment`, `quantity`, `revenue`.
- All operations run on a pandas DataFrame (optionally filtered by region/category before execution).

**Core Logic (`src/descriptive_analytics.py`):**
1. Validate presence of revenue and quantity columns.
2. Compute overall totals:
   - `total_revenue = sum(revenue)`
   - `total_quantity = sum(quantity)`
   - `average_order_value = total_revenue / total_quantity` (guarded against division by zero).
3. Add optional date range if date column exists.
4. Group aggregations using `DataFrame.groupby(...).agg(...)`:
   - `by_category`, `by_region`, `by_channel`, `by_segment` → each includes total revenue, average revenue, transaction counts, and quantities.
5. Identify `top_categories` by sorting aggregated revenue descending and calculating percentage share.

**Output Structure:**
```json
{
  "summary": {
    "total_revenue": 252034977.0,
    "total_quantity": 52068.0,
    "average_order_value": 4840.5,
    "number_of_records": 100000
  },
  "by_category": {...},
  "by_region": {...},
  "top_categories": [
    {"category": "Electronics", "total_revenue": 78091016.0, "percentage": 31.0},
    ...
  ]
}
```

**Answer Delivery:**
- `synthesize_answer_with_deepseek` prompts DeepSeek to produce either:
  - `simple` responses (direct value, e.g., “Khulna average order value is $5,192.70”).
  - `detailed` narratives (~120 words) summarizing highlights.
- Chart payload builder generates bar charts for regions/categories when appropriate.

---

## 2. Diagnostic Analytics – “Why It Happened?”

**Goal:** Detect and explain sudden drops or anomalies in performance.

**Data Inputs:** Same DataFrame; region/category filtering applied before module invocation.

**Core Logic (`src/diagnostic_analytics.py`):**
1. Require `date` and `revenue` columns; convert `date` to pandas datetime.
2. For each region:
   - Resample daily revenue (`resample('D').sum()`).
   - Require ≥14 days of data for stability.
   - Calculate rolling 7-day mean for current window (`rolling(window=7, min_periods=3)`).
   - Compare current vs. previous window averages; if percentage change ≤ −50%, flag as dip.
   - Capture metadata: region, date, current/previous averages, drop percentage.
   - Extract contributing categories during the dip window for narrative context.
3. Repeat similar dip detection per sales channel.
4. Detect anomalies using z-score threshold:
   - `z = (revenue - mean) / std`
   - Entries with `|z| > 3` classified as anomalies (up to 20 stored).
5. Compile human-readable insights from dip metadata (e.g., “Khulna region showed 77.8% drop... due to lower FMCG sales”).

**Output Structure:**
```json
{
  "insights": ["Dhaka region showed 55.8% drop..."],
  "region_dips": [
    {"region": "Dhaka", "date": "2025-02-03", "drop_percentage": -55.8, ...},
    ...
  ],
  "channel_dips": [...],
  "anomalies": [
    {"date": "2025-02-05", "revenue": 5000.0, "z_score": 3.5, "region": "Sylhet"}
  ],
  "summary": {
    "total_dips_detected": 522,
    "total_anomalies": 15
  }
}
```

**Answer Delivery:**
- For “why” questions, DeepSeek uses diagnostic insights to craft narrative explanations.
- When user asks “Why should Increase promotions in Khulna?”, `synthesize_recommendation_explanation` combines prescriptive recommendation details with diagnostic data.
- Charts: stacked bars for drop severity timeline, heatmap for anomaly counts by region/date.

---

## 3. Predictive Analytics – “What Will Happen?”

**Goal:** Forecast future revenue trends and provide ranges/alerts.

**Data Inputs:** `date`, `revenue`, optionally `region` for breakdown.

**Core Logic (`src/predictive_analytics.py`):**
1. Ensure required columns exist; convert dates, sort chronologically.
2. For each region (if present):
   - Resample to daily revenue (`resample('D').sum()`, fill missing days with zeros).
   - Use scikit-learn `LinearRegression` (if installed) or NumPy polyfit fallback.
   - Fit on index positions vs revenue values (`x = np.arange(len(series))`).
   - Forecast next 30 days (`future_x` continuation of index).
   - Calculate residual standard error from training residuals.
   - Build confidence bounds: `forecast ± 1.96 * std_error`.
   - Sum future daily values → monthly forecast, upper, lower.
   - Determine trend direction from slope sign.
3. Compute overall forecast using aggregated daily revenue similarly.

**Output Structure:**
```json
{
  "forecasts": {
    "Dhaka": {
      "monthly_forecast": 21265541.0,
      "monthly_upper_bound": 24500000.0,
      "monthly_lower_bound": 18000000.0,
      "daily_forecast": {
        "dates": ["2025-04-01", ...],
        "values": [650000.0, ...],
        "upper_bound": [...],
        "lower_bound": [...]
      },
      "trend": {"slope": 1200.0, "direction": "increasing"}
    },
    ...
  },
  "overall_forecast": {...},
  "summary": {
    "regions_forecasted": 5,
    "method": "sklearn_linear_regression",
    "forecast_horizon_days": 30
  }
}
```

**Answer Delivery:**
- DeepSeek returns forecasts in simple or detailed style (e.g., “Next month revenue expected ~ $3.64M, trend decreasing”).
- `build_chart_payloads` generates line charts for the forecasted region’s daily curve when `daily_forecast` present.

---

## 4. Prescriptive Analytics – “What Should We Do?”

**Goal:** Translate diagnostics and forecasts into recommended actions with priorities.

**Inputs:** Combination of descriptive, diagnostic, and predictive outputs plus raw filtered dataset.

**Core Logic (`src/prescriptive_analytics.py`):**
1. Aggregate region dips to compute severity stats:
   - Track number of dips, min/max/avg drop %, affected categories.
   - Build recommendation objects with: priority (high/medium/low), urgency, reasoning, expected impact, action text.
2. Channel dips → high-priority actions to investigate underperforming channels.
3. Predictive declining trends → medium-priority “boost sales” actions per region.
4. Descriptive channel performance → allocate resources to best-performing channels.
5. Ensure recommendations include reasoning narrative and stats for transparency.

**Output Structure:**
```json
{
  "recommendations": [
    {
      "priority": "high",
      "category": "Region Performance",
      "title": "Increase promotions in Khulna",
      "description": "Khulna region experienced...",
      "reasoning": "Based on 8 revenue drops...",
      "action": "Launch targeted marketing...",
      "expected_impact": "medium",
      "urgency": "high",
      "stats": {
        "num_dips": 8,
        "avg_drop_pct": 62.5,
        "max_drop_pct": 77.8,
        "affected_categories": ["FMCG", "Electronics"]
      }
    },
    ...
  ]
}
```

**Answer Delivery:**
- DeepSeek synthesizes recommendation-focused narratives, emphasizing priority, actions, and rationale.
- Follow-up “why” questions trigger targeted explanation synthesis linking dips/anomalies to the specific recommendation.

---

## 5. End-to-End Flow Summary

1. **Intent Understanding:** DeepSeek returns `query_spec` with analytics type, filters, metrics, answer style, intent summary.
2. **Data Filtering:** Region/category filters applied before module execution.
3. **Analytics Execution:** Selected module produces structured results as described above.
4. **Answer Synthesis:** DeepSeek uses tailored prompts (simple vs. detailed, category comparisons, recommendation explanations) to produce natural language output.
5. **Visualization:** Optional charts generated based on results (descriptive bars, predictive lines, diagnostic heatmaps).
6. **Response Packaging:** Chat message stores text, charts, raw JSON for transparency.

---
*Updated: November 2025*

