"""
Agentic Intelligence Module
Provides agent-based architecture for autonomous sales intelligence analysis.
Designed to be integrated with LangChain for advanced agentic workflows.
"""

from .base_agent import BaseAgent
from .coordinator_agent import CoordinatorAgent

# Legacy agents - optional imports (using old class-based analytics)
try:
    from .analytical_agent import AnalyticalAgent
    from .forecasting_agent import ForecastingAgent
    __all__ = [
        "BaseAgent",
        "CoordinatorAgent",
        "AnalyticalAgent",
        "ForecastingAgent"
    ]
except ImportError:
    # If old agents can't be imported (due to analytics module changes), just export what we can
    __all__ = [
        "BaseAgent",
        "CoordinatorAgent"
    ]

