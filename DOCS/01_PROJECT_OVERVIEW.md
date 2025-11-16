# Sales Intelligence Agentic System – Project Overview

## Purpose

The Sales Intelligence Agentic System equips sales teams with a natural-language interface for exploring historical performance, spotting issues, forecasting trends, and generating actionable recommendations. It combines structured analytics modules with a conversational Streamlit UI powered by DeepSeek LLM reasoning.

## High-Level Goals

- Provide enterprise stakeholders with quick answers to “what happened, why it happened, what will happen, and what should we do”.
- Deliver concise, business-ready insights via chat, while retaining drill-down access to raw analytics.
- Support client-facing presentations (e.g., Akij Group) and interview scenarios with clear documentation and reusable prompts.
- Offer an API surface (FastAPI) for automation, integrations (n8n), and partner tooling.

## Key Features

- **Conversational Analytics**: Streamlit chat UI, DeepSeek-powered intent understanding, structured query planning, and narrative answer synthesis.
- **Four Analytics Pillars**:
  - Descriptive: revenue, quantity, category/region breakdowns.
  - Diagnostic: rolling-window dip detection, anomaly spotting, root-cause narratives.
  - Predictive: linear-regression forecasts, regional projections, confidence bounds.
  - Prescriptive: prioritized recommendations with reasoning and expected impact.
- **Two-Stage LLM Flow**: DeepSeek first extracts query specs (analytics type, filters, metrics, answer style) then produces tailored responses.
- **Visual Enrichment**: Inline charts (bar, line, stacked, heatmap) for descriptive and diagnostic insights.
- **Robustness**: Fallback handling for API failures, JSON parsing, empty data, and optional OpenAI backup.
- **API Integration**: FastAPI endpoints (`/ask`, `/analyze`, `/health`) for external services and automation workflows.

## Architecture Snapshot

1. **Coordinator Agent** orchestrates analytic modules, caches raw results, and manages LLM synthesis.
2. **Analytics Modules** (`descriptive`, `diagnostic`, `predictive`, `prescriptive`) operate on filtered pandas data, returning structured dictionaries.
3. **UI Layer** (`ui/chat_demo.py`): Streamlit session state holds conversation, renders answers + charts, and handles DeepSeek calls.
4. **LLM Integration** (`src/agents/langchain_integration.py`): defines the concise narrative prompt, formatting helpers, and fallback logic.
5. **Data Layer** (`src/data_loader.py`): loads CSV inputs (sample data included) and validates schema.
6. **API Layer** (`api/fastapi_endpoint.py`): exposes analytics via REST.

## Current Capabilities

- Natural-language Q&A for KPI lookups, root-cause analysis, and forecasting.
- Region/category filtering with automatic detection in user queries.
- Simple vs. detailed answer styles (e.g., direct KPI vs. narrative) based on intent.
- Diagnostic charts summarizing dip severity and anomaly density.
- Streamlit/DeepSeek environment scripts (`run_streamlit_venv.ps1`) for local launch.

## Deliverables

| Component | Status |
|-----------|--------|
| Streamlit chat UI with DeepSeek | ✅ Complete |
| Analytics modules (4 types) | ✅ Complete |
| Intent-to-query planning | ✅ Complete |
| Chart rendering in chat | ✅ Descriptive & diagnostic |
| FastAPI endpoints | ✅ Complete |
| Sample sales dataset | ✅ Included |

## Next Steps

- Finalize rewritten documentation set (architecture, analytics deep dive, integration guides).
- Optional: add automated tests for API/LLM fallbacks.
- Package project for client delivery (clean repo, environment instructions, demo script).

---
*Updated: November 2025*


