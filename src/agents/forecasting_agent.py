"""
Forecasting Agent
Specialized agent for predictive analytics and forecasting.
"""

from typing import Dict, Any, Optional
import pandas as pd

from .base_agent import BaseAgent
from ..predictive_analytics import PredictiveAnalytics

logger = __import__('logging').getLogger(__name__)


class ForecastingAgent(BaseAgent):
    """
    Agent specialized in predictive analytics and forecasting.
    Answers "What will happen?" questions.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the forecasting agent.
        
        Args:
            data: DataFrame containing sales data with date column
            max_iterations: Maximum number of iterations
            verbose: Whether to log detailed information
        """
        super().__init__(
            name="ForecastingAgent",
            description="Generates forecasts and predictive insights for sales data",
            max_iterations=max_iterations,
            verbose=verbose
        )
        self.data = data
        self.predictive = PredictiveAnalytics(data)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process forecasting queries.
        
        Args:
            input_data: Dictionary with 'query_type' and forecasting parameters
            
        Returns:
            Dictionary with forecasting results
        """
        self.increment_iteration()
        
        if not self.check_iteration_limit():
            return {"error": "Maximum iterations reached"}
        
        query_type = input_data.get("query_type", "moving_average")
        metric = input_data.get("metric", "revenue")
        
        try:
            if query_type == "moving_average":
                window = input_data.get("window", 7)
                horizon = input_data.get("horizon", 30)
                result = self.predictive.moving_average_forecast(
                    metric, window, horizon
                )
                self.log_action(
                    f"Moving average forecast (window={window}, horizon={horizon})",
                    "success"
                )
                
            elif query_type == "exponential_smoothing":
                alpha = input_data.get("alpha", 0.3)
                horizon = input_data.get("horizon", 30)
                result = self.predictive.exponential_smoothing_forecast(
                    metric, alpha, horizon
                )
                self.log_action(
                    f"Exponential smoothing forecast (alpha={alpha})",
                    "success"
                )
                
            elif query_type == "trend_analysis":
                result = self.predictive.trend_analysis(metric)
                self.log_action("Trend analysis", "success")
                
            elif query_type == "seasonal_patterns":
                result = self.predictive.seasonal_patterns(metric)
                self.log_action("Seasonal pattern analysis", "success")
                
            else:
                result = {"error": f"Unknown query type: {query_type}"}
                self.log_action(f"Process query: {query_type}", "error")
            
            return {
                "success": True,
                "query_type": query_type,
                "result": result,
                "agent": self.name
            }
            
        except Exception as e:
            error_msg = str(e)
            self.log_action(f"Process query: {query_type}", f"error: {error_msg}")
            return {
                "success": False,
                "query_type": query_type,
                "error": error_msg,
                "agent": self.name
            }
    
    def get_capabilities(self) -> list:
        """Get list of capabilities."""
        return [
            "moving_average_forecast",
            "exponential_smoothing_forecast",
            "trend_analysis",
            "seasonal_pattern_identification"
        ]

