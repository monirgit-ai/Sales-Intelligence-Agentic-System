"""
Analytical Agent
Specialized agent for performing descriptive and diagnostic analytics.
"""

from typing import Dict, Any, Optional
import pandas as pd

from .base_agent import BaseAgent
from ..descriptive_analytics import DescriptiveAnalytics
from ..diagnostic_analytics import DiagnosticAnalytics

logger = __import__('logging').getLogger(__name__)


class AnalyticalAgent(BaseAgent):
    """
    Agent specialized in descriptive and diagnostic analytics.
    Answers "What happened?" and "Why did it happen?" questions.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the analytical agent.
        
        Args:
            data: DataFrame containing sales data
            max_iterations: Maximum number of iterations
            verbose: Whether to log detailed information
        """
        super().__init__(
            name="AnalyticalAgent",
            description="Performs descriptive and diagnostic analytics on sales data",
            max_iterations=max_iterations,
            verbose=verbose
        )
        self.data = data
        self.descriptive = DescriptiveAnalytics(data)
        self.diagnostic = DiagnosticAnalytics(data)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process analytical queries.
        
        Args:
            input_data: Dictionary with 'query_type' and query parameters
            
        Returns:
            Dictionary with analytical results
        """
        self.increment_iteration()
        
        if not self.check_iteration_limit():
            return {"error": "Maximum iterations reached"}
        
        query_type = input_data.get("query_type", "summary")
        
        try:
            if query_type == "summary":
                result = self.descriptive.summary_statistics()
                self.log_action("Generate summary statistics", "success")
                
            elif query_type == "aggregate":
                dimension = input_data.get("dimension", "product_category")
                result = self.descriptive.aggregate_by_dimension(dimension).to_dict()
                self.log_action(f"Aggregate by {dimension}", "success")
                
            elif query_type == "time_series":
                freq = input_data.get("frequency", "D")
                result = self.descriptive.time_series_summary(freq).to_dict()
                self.log_action(f"Time series summary ({freq})", "success")
                
            elif query_type == "top_performers":
                dimension = input_data.get("dimension", "product_name")
                metric = input_data.get("metric", "revenue")
                top_n = input_data.get("top_n", 10)
                result = self.descriptive.top_performers(
                    dimension, metric, top_n
                ).to_dict()
                self.log_action(f"Top performers by {dimension}", "success")
                
            elif query_type == "correlation":
                dimensions = input_data.get("dimensions")
                result = self.diagnostic.correlation_analysis(dimensions).to_dict()
                self.log_action("Correlation analysis", "success")
                
            elif query_type == "variance":
                metric = input_data.get("metric", "revenue")
                dimension = input_data.get("dimension", "product_category")
                result = self.diagnostic.variance_analysis(metric, dimension)
                self.log_action(f"Variance analysis ({dimension})", "success")
                
            elif query_type == "anomaly_detection":
                metric = input_data.get("metric", "revenue")
                threshold = input_data.get("threshold", 2.0)
                result = self.diagnostic.anomaly_detection(metric, threshold).to_dict('records')
                self.log_action("Anomaly detection", f"Found {len(result)} anomalies")
                
            elif query_type == "segment_performance":
                segment = input_data.get("segment_dimension", "customer_segment")
                metric = input_data.get("metric", "revenue")
                result = self.diagnostic.segment_performance(segment, metric)
                self.log_action(f"Segment performance ({segment})", "success")
                
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
            "summary_statistics",
            "aggregation_by_dimension",
            "time_series_analysis",
            "top_performers_identification",
            "correlation_analysis",
            "variance_analysis",
            "anomaly_detection",
            "segment_performance_analysis"
        ]

