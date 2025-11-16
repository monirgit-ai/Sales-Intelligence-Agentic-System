"""
Configuration module for the Sales Intelligence Agentic System.
Contains settings and constants used throughout the application.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_FILE = DATA_DIR / "sample_sales.csv"

# Analytics settings
DEFAULT_DATE_COLUMN = "date"
DEFAULT_REVENUE_COLUMN = "revenue"
DEFAULT_QUANTITY_COLUMN = "quantity"

# Hierarchical dimensions
DIMENSIONS = [
    "date",
    "product_category",
    "product_name",
    "region",
    "customer_segment",
    "sales_channel"
]

# Metrics
METRICS = [
    "quantity",
    "revenue"
]

# Forecasting settings
FORECAST_HORIZON = 30  # days
FORECAST_CONFIDENCE_INTERVAL = 0.95

# Agent settings
AGENT_MAX_ITERATIONS = 10
AGENT_VERBOSE = True

