"""
Coordinator Agent
Orchestrates all analytics modules and generates comprehensive reports
with natural language synthesis via LangChain (placeholder).
"""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import logging

from ..data_loader import DataLoader
from ..descriptive_analytics import run as run_descriptive
from ..diagnostic_analytics import run as run_diagnostic
from ..predictive_analytics import run as run_predictive
from ..prescriptive_analytics import run as run_prescriptive

logger = logging.getLogger(__name__)

# Optional LangChain integration
try:
    from .langchain_integration import synthesize_with_langchain_llmchain
    LANGCHAIN_INTEGRATION_AVAILABLE = True
except ImportError:
    LANGCHAIN_INTEGRATION_AVAILABLE = False
    logger.info("LangChain integration not available (optional dependency)")


class CoordinatorAgent:
    """
    Coordinator agent that orchestrates all analytics modules:
    - Descriptive Analytics
    - Diagnostic Analytics
    - Predictive Analytics
    - Prescriptive Analytics
    
    Generates both raw analytics results and natural language summaries.
    """
    
    def __init__(
        self,
        data_file_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the coordinator agent.
        
        Args:
            data_file_path: Optional path to CSV file. If None, uses default sample_sales.csv
            verbose: Whether to log detailed information
        """
        self.data_file_path = data_file_path
        self.verbose = verbose
        if data_file_path is None:
            self.data_loader = DataLoader()
        else:
            self.data_loader = DataLoader(Path(data_file_path))
        self.data: Optional[pd.DataFrame] = None
        
        if self.verbose:
            logger.info("CoordinatorAgent initialized")
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load sales data using DataLoader.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            Exception: If data loading fails
        """
        try:
            logger.info("Loading sales data...")
            self.data = self.data_loader.load()
            
            # Validate data
            validation = self.data_loader.validate_data()
            if not validation.get("valid", False):
                logger.warning(f"Data validation issues: {validation.get('missing_columns', [])}")
            
            logger.info(f"Successfully loaded {len(self.data)} records")
            return self.data
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _run_descriptive_analytics(self) -> Dict[str, Any]:
        """
        Execute descriptive analytics.
        
        Returns:
            Descriptive analytics results
        """
        try:
            logger.info("Running descriptive analytics...")
            results = run_descriptive(self.data)
            logger.info("Descriptive analytics completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in descriptive analytics: {e}")
            return {"error": str(e), "stage": "descriptive"}
    
    def _run_diagnostic_analytics(self) -> Dict[str, Any]:
        """
        Execute diagnostic analytics.
        
        Returns:
            Diagnostic analytics results
        """
        try:
            logger.info("Running diagnostic analytics...")
            results = run_diagnostic(self.data)
            logger.info("Diagnostic analytics completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in diagnostic analytics: {e}")
            return {"error": str(e), "stage": "diagnostic"}
    
    def _run_predictive_analytics(self) -> Dict[str, Any]:
        """
        Execute predictive analytics.
        
        Returns:
            Predictive analytics results
        """
        try:
            logger.info("Running predictive analytics...")
            results = run_predictive(self.data)
            logger.info("Predictive analytics completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in predictive analytics: {e}")
            return {"error": str(e), "stage": "predictive"}
    
    def _run_prescriptive_analytics(
        self,
        descriptive_results: Dict[str, Any],
        diagnostic_results: Dict[str, Any],
        predictive_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute prescriptive analytics using results from other modules.
        
        Args:
            descriptive_results: Results from descriptive analytics
            diagnostic_results: Results from diagnostic analytics
            predictive_results: Results from predictive analytics
            
        Returns:
            Prescriptive analytics results
        """
        try:
            logger.info("Running prescriptive analytics...")
            results = run_prescriptive(
                self.data,
                diagnostic_results=diagnostic_results,
                predictive_results=predictive_results,
                descriptive_results=descriptive_results
            )
            logger.info("Prescriptive analytics completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in prescriptive analytics: {e}")
            return {"error": str(e), "stage": "prescriptive"}
    
    def _generate_summary_insights(
        self,
        descriptive_results: Dict[str, Any],
        diagnostic_results: Dict[str, Any],
        predictive_results: Dict[str, Any],
        prescriptive_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate summary insights dictionary for LLM synthesis.
        
        Args:
            descriptive_results: Descriptive analytics results
            diagnostic_results: Diagnostic analytics results
            predictive_results: Predictive analytics results
            prescriptive_results: Prescriptive analytics results
            
        Returns:
            Dictionary with key insights for LLM processing
        """
        summary = {
            "overall_summary": {},
            "key_findings": [],
            "recommendations_summary": []
        }
        
        # Extract overall summary from descriptive
        if "summary" in descriptive_results and "error" not in descriptive_results:
            summary["overall_summary"] = {
                "total_revenue": descriptive_results["summary"].get("total_revenue", 0),
                "total_quantity": descriptive_results["summary"].get("total_quantity", 0),
                "average_order_value": descriptive_results["summary"].get("average_order_value", 0)
            }
        
        # Extract key findings from diagnostic
        if "insights" in diagnostic_results and "error" not in diagnostic_results:
            summary["key_findings"] = diagnostic_results["insights"][:5]  # Top 5 insights
        
        # Extract recommendations
        if "recommendations" in prescriptive_results and "error" not in prescriptive_results:
            high_priority = prescriptive_results.get("categorized", {}).get("high_priority", [])
            summary["recommendations_summary"] = [
                {
                    "title": rec.get("title", ""),
                    "category": rec.get("category", ""),
                    "priority": rec.get("priority", "")
                }
                for rec in high_priority[:5]  # Top 5 recommendations
            ]
        
        # Add predictive trends
        if "overall_forecast" in predictive_results and "error" not in predictive_results:
            forecast = predictive_results["overall_forecast"]
            if forecast:
                summary["forecast_trend"] = {
                    "monthly_forecast": forecast.get("monthly_forecast", 0),
                    "trend_direction": forecast.get("trend", {}).get("direction", "unknown")
                }
        
        return summary
    
    def _synthesize_narrative(self, summary_insights: Dict[str, Any]) -> str:
        """
        Generate natural language narrative using LangChain LLMChain.
        
        Uses LangChain LLMChain for actual LLM reasoning calls when available.
        Falls back to rule-based synthesis if LangChain unavailable.
        
        Args:
            summary_insights: Summary insights dictionary
            
        Returns:
            Natural language report
        """
        # Try LangChain LLMChain integration first (explicit LangChain usage)
        if LANGCHAIN_INTEGRATION_AVAILABLE:
            try:
                # Get raw results for LangChain context
                raw_results = getattr(self, '_last_raw_results', {})
                
                # Validate we have data
                if not raw_results or not raw_results.get("descriptive"):
                    logger.warning("No raw results available for LangChain synthesis")
                    raise ValueError("No analytics results available")
                
                # Prepare combined results for LangChain
                combined_results = {
                    "descriptive": raw_results.get("descriptive", {}),
                    "diagnostic": raw_results.get("diagnostic", {}),
                    "predictive": raw_results.get("predictive", {}),
                    "prescriptive": raw_results.get("prescriptive", {})
                }
                
                # Call LangChain LLMChain (makes actual LLM API call)
                # Automatically detects DeepSeek or OpenAI based on API keys
                narrative = synthesize_with_langchain_llmchain(combined_results)
                
                # Check if we got a real LLM response (not fallback)
                if narrative and not narrative.startswith("Sales intelligence analysis completed"):
                    logger.info("✓ Generated narrative using LangChain LLMChain (LLM reasoning)")
                    return narrative
                else:
                    logger.warning("LangChain returned fallback, using internal synthesis")
            except Exception as e:
                logger.warning(f"LangChain synthesis error: {e}. Using fallback.")
        
        # Fallback to rule-based synthesis
        logger.info("Using internal summarizer (LangChain not available or failed)")
        
        narrative_parts = []
        
        # Build narrative from insights
        overall = summary_insights.get("overall_summary", {})
        if overall:
            narrative_parts.append(
                f"Sales Performance Summary: Total revenue reached ${overall.get('total_revenue', 0):,.2f} "
                f"with {overall.get('total_quantity', 0):,.0f} units sold, resulting in an average order value "
                f"of ${overall.get('average_order_value', 0):,.2f}."
            )
        
        findings = summary_insights.get("key_findings", [])
        if findings:
            narrative_parts.append("\nKey Findings:")
            for i, finding in enumerate(findings, 1):
                narrative_parts.append(f"  {i}. {finding}")
        
        forecast = summary_insights.get("forecast_trend", {})
        if forecast:
            direction = forecast.get("trend_direction", "unknown")
            monthly = forecast.get("monthly_forecast", 0)
            narrative_parts.append(
                f"\nForecast: Next month revenue is projected at ${monthly:,.2f} "
                f"with a {direction} trend."
            )
        
        recommendations = summary_insights.get("recommendations_summary", [])
        if recommendations:
            narrative_parts.append("\nPriority Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                narrative_parts.append(
                    f"  {i}. [{rec.get('priority', '').upper()}] {rec.get('title', '')} "
                    f"({rec.get('category', '')})"
                )
        
        # If no insights, provide default message
        if not narrative_parts:
            narrative_parts.append(
                "Sales intelligence analysis completed. Please review the detailed "
                "analytics results for specific insights."
            )
        
        narrative = "\n".join(narrative_parts)
        return narrative
    
    def run_all(self) -> Dict[str, Any]:
        """
        Execute all analytics modules sequentially and generate comprehensive report.
        
        Returns:
            Dictionary containing:
                - raw_results: Complete analytics results from all modules
                - summary_insights: Condensed insights for LLM processing
                - narrative_report: Natural language summary report
                - execution_summary: Status of each stage
        """
        execution_summary = {
            "stages": {},
            "overall_status": "pending",
            "errors": []
        }
        
        try:
            # Stage 1: Load data
            logger.info("="*70)
            logger.info("Starting comprehensive sales intelligence analysis")
            logger.info("="*70)
            
            try:
                # Only load data if it's not already set (for filtered queries)
                if self.data is None:
                    self._load_data()
                else:
                    logger.info(f"Using existing data: {len(self.data)} records")
                execution_summary["stages"]["data_loading"] = "success"
            except Exception as e:
                execution_summary["stages"]["data_loading"] = "failed"
                execution_summary["errors"].append(f"Data loading: {str(e)}")
                logger.error(f"Failed to load data: {e}")
                raise
            
            # Stage 2: Descriptive Analytics
            try:
                descriptive_results = self._run_descriptive_analytics()
                execution_summary["stages"]["descriptive"] = "success" if "error" not in descriptive_results else "failed"
                if "error" in descriptive_results:
                    execution_summary["errors"].append(f"Descriptive: {descriptive_results['error']}")
            except Exception as e:
                descriptive_results = {"error": str(e)}
                execution_summary["stages"]["descriptive"] = "failed"
                execution_summary["errors"].append(f"Descriptive: {str(e)}")
                logger.error(f"Descriptive analytics failed: {e}")
            
            # Stage 3: Diagnostic Analytics
            try:
                diagnostic_results = self._run_diagnostic_analytics()
                execution_summary["stages"]["diagnostic"] = "success" if "error" not in diagnostic_results else "failed"
                if "error" in diagnostic_results:
                    execution_summary["errors"].append(f"Diagnostic: {diagnostic_results['error']}")
            except Exception as e:
                diagnostic_results = {"error": str(e)}
                execution_summary["stages"]["diagnostic"] = "failed"
                execution_summary["errors"].append(f"Diagnostic: {str(e)}")
                logger.error(f"Diagnostic analytics failed: {e}")
            
            # Stage 4: Predictive Analytics
            try:
                predictive_results = self._run_predictive_analytics()
                execution_summary["stages"]["predictive"] = "success" if "error" not in predictive_results else "failed"
                if "error" in predictive_results:
                    execution_summary["errors"].append(f"Predictive: {predictive_results['error']}")
            except Exception as e:
                predictive_results = {"error": str(e)}
                execution_summary["stages"]["predictive"] = "failed"
                execution_summary["errors"].append(f"Predictive: {str(e)}")
                logger.error(f"Predictive analytics failed: {e}")
            
            # Stage 5: Prescriptive Analytics
            try:
                prescriptive_results = self._run_prescriptive_analytics(
                    descriptive_results,
                    diagnostic_results,
                    predictive_results
                )
                execution_summary["stages"]["prescriptive"] = "success" if "error" not in prescriptive_results else "failed"
                if "error" in prescriptive_results:
                    execution_summary["errors"].append(f"Prescriptive: {prescriptive_results['error']}")
            except Exception as e:
                prescriptive_results = {"error": str(e)}
                execution_summary["stages"]["prescriptive"] = "failed"
                execution_summary["errors"].append(f"Prescriptive: {str(e)}")
                logger.error(f"Prescriptive analytics failed: {e}")
            
            # Stage 6: Merge all results (before summary generation so it's available for narrative)
            raw_results = {
                "descriptive": descriptive_results,
                "diagnostic": diagnostic_results,
                "predictive": predictive_results,
                "prescriptive": prescriptive_results
            }
            
            # Store raw results for LangChain context (must be before narrative synthesis)
            self._last_raw_results = raw_results
            
            # Stage 7: Generate summary insights
            try:
                summary_insights = self._generate_summary_insights(
                    descriptive_results,
                    diagnostic_results,
                    predictive_results,
                    prescriptive_results
                )
                execution_summary["stages"]["summary_generation"] = "success"
            except Exception as e:
                summary_insights = {}
                execution_summary["stages"]["summary_generation"] = "failed"
                execution_summary["errors"].append(f"Summary generation: {str(e)}")
                logger.error(f"Summary generation failed: {e}")
            
            # Stage 8: Synthesize narrative (now _last_raw_results is available)
            try:
                narrative_report = self._synthesize_narrative(summary_insights)
                execution_summary["stages"]["narrative_synthesis"] = "success"
            except Exception as e:
                narrative_report = "Error generating narrative report."
                execution_summary["stages"]["narrative_synthesis"] = "failed"
                execution_summary["errors"].append(f"Narrative synthesis: {str(e)}")
                logger.error(f"Narrative synthesis failed: {e}")
            
            # Determine overall status
            failed_stages = [stage for stage, status in execution_summary["stages"].items() if status == "failed"]
            if not failed_stages:
                execution_summary["overall_status"] = "success"
            elif "data_loading" in failed_stages:
                execution_summary["overall_status"] = "critical_failure"
            else:
                execution_summary["overall_status"] = "partial_success"
            
            logger.info("="*70)
            logger.info("Analysis complete")
            logger.info(f"Overall status: {execution_summary['overall_status']}")
            logger.info("="*70)
            
            return {
                "raw_results": raw_results,
                "summary_insights": summary_insights,
                "narrative_report": narrative_report,
                "execution_summary": execution_summary
            }
            
        except Exception as e:
            logger.error(f"Critical error in run_all(): {e}")
            execution_summary["overall_status"] = "critical_failure"
            execution_summary["errors"].append(f"Critical: {str(e)}")
            
            return {
                "raw_results": {},
                "summary_insights": {},
                "narrative_report": f"Analysis failed with critical error: {str(e)}",
                "execution_summary": execution_summary,
                "error": str(e)
            }


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize coordinator agent
    coordinator = CoordinatorAgent(verbose=True)
    
    # Run all analytics
    print("\n" + "="*70)
    print("COORDINATOR AGENT - COMPREHENSIVE ANALYSIS")
    print("="*70 + "\n")
    
    results = coordinator.run_all()
    
    # Print execution summary
    print("\nExecution Summary:")
    print(f"  Overall Status: {results['execution_summary']['overall_status']}")
    for stage, status in results['execution_summary']['stages'].items():
        print(f"  {stage}: {status}")
    
    if results['execution_summary']['errors']:
        print("\nErrors:")
        for error in results['execution_summary']['errors']:
            print(f"  - {error}")
    
    # Print narrative report
    print("\n" + "="*70)
    print("NARRATIVE REPORT")
    print("="*70)
    print(results['narrative_report'])
    print("="*70)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("QUICK SUMMARY")
    print("="*70)
    
    summary = results.get('summary_insights', {})
    overall = summary.get('overall_summary', {})
    if overall:
        print(f"\nTotal Revenue: ${overall.get('total_revenue', 0):,.2f}")
        print(f"Total Quantity: {overall.get('total_quantity', 0):,.0f}")
        print(f"Average Order Value: ${overall.get('average_order_value', 0):,.2f}")
    
    print("="*70 + "\n")
