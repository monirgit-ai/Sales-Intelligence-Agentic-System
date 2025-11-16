# Technical Architecture – Sales Intelligence Agentic System

## 1. Solution Topology

```
┌─────────────────────────┐          ┌─────────────────────────┐
│ Streamlit UI (chat_demo)│  <---->  │ DeepSeek LLM (API)      │
│  - Conversation state   │          │  - Intent planning      │
│  - Chart rendering      │          │  - Answer synthesis     │
└──────────┬──────────────┘          └──────────┬──────────────┘
           │                                      │
           ▼                                      ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│ CoordinatorAgent        │  <---->  │ Analytics Modules       │
│  - Orchestration        │          │ 1. Descriptive (pandas) │
│  - Data filtering       │          │ 2. Diagnostic (dips)    │
│  - Raw result caching   │          │ 3. Predictive (forecast)│
│  - LLM prompt context   │          │ 4. Prescriptive (rules) │
└──────────┬──────────────┘          └──────────┬──────────────┘
           │                                      │
           ▼                                      ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│ Data Loader             │          │ FastAPI Service         │
│  - CSV ingestion        │          │  - /ask /analyze /health│
│  - Validation           │          │  - n8n integrations     │
└─────────────────────────┘          └─────────────────────────┘
```

## 2. Key Components

### 2.1 UI Layer (`ui/chat_demo.py`)
- Streamlit app orchestrating user interactions.
- Maintains session state (`messages`, `coordinator`, `data_loaded`).
- Invokes DeepSeek in two stages: intent understanding and answer synthesis.
- Renders markdown responses, raw JSON expanders, and Altair-based charts.
- Handles example question shortcuts, system status, and environment setup.

### 2.2 Coordinator Agent (`src/agents/coordinator_agent.py`)
- Central controller invoked by both UI and API layers.
- Responsibilities:
  - Load and cache sales dataset (`self.data`).
  - Execute analytics modules sequentially.
  - Store `self._last_raw_results` for LLM prompt context.
  - Generate summary insights and narrative via LangChain integration.
  - Provide fallbacks when LLM synthesis fails.
- Ensures data is only reloaded when necessary (preserves filtered datasets).

### 2.3 Analytics Modules
1. **Descriptive (`src/descriptive_analytics.py`)**
   - Aggregates totals by category, region, channel, segment.
   - Produces `summary`, `top_categories`, and grouped dictionaries.
2. **Diagnostic (`src/diagnostic_analytics.py`)**
   - Detects 50%+ drops via rolling averages per region/channel.
   - Generates regional insights, dip metadata, anomaly list.
3. **Predictive (`src/predictive_analytics.py`)**
   - Uses scikit-learn linear regression (fallback: NumPy) over daily revenue.
   - Forecasts 30-day horizon, calculates confidence bounds, trend direction.
4. **Prescriptive (`src/prescriptive_analytics.py`)**
   - Converts diagnostic/predictive signals into prioritized recommendations.
   - Aggregates dips per region, adds reasoning/action fields, urgency.

### 2.4 LLM Integration (`src/agents/langchain_integration.py`)
- Configures prompts for concise WhatsApp-style narratives.
- Formats analytics data for DeepSeek consumption.
- Limits tokens, enforces <120 word responses.
- Provides fallback synthesis when data missing or LLM unavailable.

### 2.5 Intent Planner & Synthesis (`ui/chat_demo.py`)
- `understand_intent_and_generate_query`: DeepSeek call returning JSON spec with analytics type, filters, metrics, answer style, intent summary.
- `synthesize_answer_with_deepseek`: applies analytics output, answer style (simple/detailed), and dynamic prompt templates (e.g., category comparison).
- `synthesize_recommendation_explanation`: explains prescriptive recommendations on “why” questions.

### 2.6 Charts
- `build_chart_payloads` translates analytic results into descriptive/diagnostic chart descriptors.
- `render_chart` renders Altair stacked bars, heatmaps, or Streamlit built-ins when Altair absent.
- Charts persist in chat history and replay for prior messages.

### 2.7 API Layer (`api/fastapi_endpoint.py`)
- FastAPI app exposing:
  - `GET /ask`, `POST /ask`: question answering using coordinator.
  - `GET /analyze`, `POST /analyze`: comprehensive run.
  - `GET /health`: health status.
- Shares coordinator instance to avoid repeated data loads.

## 3. Data Flow

1. User enters question → Streamlit captures input.
2. DeepSeek (intent stage) returns query spec (type, region, metrics, answer style).
3. UI filters dataset (region/category) and selects analytics module.
4. Coordinator executes analytics → returns structured results.
5. DeepSeek (synthesis stage) generates natural-language response + optional recommendation explanation.
6. UI displays text, charts, raw JSON, analytics badge.
7. Conversation stored in session state for context.

## 4. Deployment & Environment

- **Python**: 3.10+ recommended.
- **Dependencies**: `requirements.txt` covers Streamlit, pandas, scikit-learn, FastAPI, LangChain, DeepSeek connectors, Altair.
- **Env variables**: `DEEPSEEK_API_KEY` (primary), optional `OPENAI_API_KEY` for fallback.
- **Scripts**:
  - `run_streamlit_venv.ps1`: activates venv, sets API key, runs Streamlit.
  - `run_api.py`: launches FastAPI with uvicorn.
  - `run_coordinator_agent.py`: CLI trigger for coordinator run (if present).

## 5. Reliability & Error Handling

- LLM failures → fallback to rule-based summaries or descriptive formatting.
- JSON parsing guard rails for DeepSeek responses (code block stripping, try/except).
- Data filtering ensures missing region queries return explicit error message.
- Logging (INFO/ERROR) across modules for diagnosis.
- API includes health endpoint and exception handling via `HTTPException`.

## 6. Extensibility Points

- Add new analytics modules and register within coordinator.
- Extend intent planner with additional metrics or multi-filter support.
- Replace DeepSeek with alternative LLM by adjusting LangChain configuration.
- Enhance chart rendering to cover prescriptive outputs (priority heatmaps, etc.).
- Integrate authentication and rate limiting for production API usage.

---
*Updated: November 2025*

