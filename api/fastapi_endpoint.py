"""
FastAPI Endpoint for Sales Intelligence Agentic System
Provides REST API for n8n integration and external access.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from src.agents.coordinator_agent import CoordinatorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sales Intelligence Agentic System API",
    description="REST API for Sales Intelligence Analytics",
    version="1.0.0"
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global coordinator agent instance
coordinator: Optional[CoordinatorAgent] = None


def get_coordinator() -> CoordinatorAgent:
    """Get or initialize coordinator agent."""
    global coordinator
    if coordinator is None:
        logger.info("Initializing coordinator agent...")
        coordinator = CoordinatorAgent(verbose=False)
        # Ensure data is loaded
        if coordinator.data is None:
            coordinator._load_data()
    return coordinator


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str
    data_file_path: Optional[str] = None


class AnalysisRequest(BaseModel):
    """Request model for comprehensive analysis."""
    data_file_path: Optional[str] = None


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Sales Intelligence Agentic System API",
        "version": "1.0.0",
        "endpoints": {
            "GET /ask": "Ask a question and get intelligent response",
            "GET /analyze": "Run comprehensive analysis",
            "GET /health": "Health check endpoint"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        coord = get_coordinator()
        return {
            "status": "healthy",
            "coordinator_initialized": coord is not None,
            "data_loaded": coord.data is not None if coord else False
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/ask")
def ask(question: str):
    """
    Ask a question and get intelligent response.
    
    Args:
        question: User's question about sales data
        
    Returns:
        Dictionary with answer and metadata
    """
    try:
        logger.info(f"Received question: {question}")
        
        # Get coordinator
        coord = get_coordinator()
        
        # Classify question and run appropriate analytics
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['everything', 'all', 'complete', 'full', 'comprehensive', 'overview']):
            # Run comprehensive analysis
            results = coord.run_all()
            answer = results.get("narrative_report", "Analysis completed.")
            analytics_type = "comprehensive"
            raw_results = results.get("raw_results", {})
        else:
            # For specific questions, run targeted analytics
            from src.descriptive_analytics import run as run_descriptive
            from src.diagnostic_analytics import run as run_diagnostic
            from src.predictive_analytics import run as run_predictive
            from src.prescriptive_analytics import run as run_prescriptive
            
            if any(word in question_lower for word in ['forecast', 'predict', 'future', 'next month', 'will', 'trend']):
                results = run_predictive(coord.data)
                analytics_type = "predictive"
                answer = f"Forecast: Next month revenue is projected at ${results.get('overall_forecast', {}).get('monthly_forecast', 0):,.2f}"
            elif any(word in question_lower for word in ['recommend', 'action', 'suggest', 'what should', 'how to', 'improve']):
                desc_results = run_descriptive(coord.data)
                diag_results = run_diagnostic(coord.data)
                pred_results = run_predictive(coord.data)
                results = run_prescriptive(coord.data, diag_results, pred_results, desc_results)
                analytics_type = "prescriptive"
                recommendations = results.get("recommendations", [])
                answer = f"Recommendations:\n" + "\n".join([
                    f"- {r['title']} ({r['priority']})" for r in recommendations[:5]
                ])
            elif any(word in question_lower for word in ['why', 'cause', 'reason', 'anomaly', 'dip', 'drop', 'problem']):
                results = run_diagnostic(coord.data)
                analytics_type = "diagnostic"
                insights = results.get("insights", [])
                answer = "\n".join(insights[:5]) if insights else "No significant issues detected."
            else:
                # Default to descriptive
                results = run_descriptive(coord.data)
                analytics_type = "descriptive"
                summary = results.get("summary", {})
                answer = f"Total Revenue: ${summary.get('total_revenue', 0):,.2f}\nTotal Quantity: {summary.get('total_quantity', 0):,.0f}"
            
            raw_results = results
        
        return {
            "answer": answer,
            "analytics_type": analytics_type,
            "question": question,
            "raw_results": raw_results
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/ask")
def ask_post(request: QuestionRequest):
    """
    Ask a question via POST request.
    
    Args:
        request: QuestionRequest with question and optional data file path
        
    Returns:
        Dictionary with answer and metadata
    """
    try:
        # If custom data file provided, create new coordinator
        if request.data_file_path:
            coord = CoordinatorAgent(data_file_path=request.data_file_path, verbose=False)
            if coord.data is None:
                coord._load_data()
        else:
            coord = get_coordinator()
        
        # Process question (similar to GET endpoint)
        question_lower = request.question.lower()
        
        if any(word in question_lower for word in ['everything', 'all', 'complete', 'full', 'comprehensive']):
            results = coord.run_all()
            answer = results.get("narrative_report", "Analysis completed.")
            analytics_type = "comprehensive"
        else:
            # Use simple classification for specific questions
            from src.descriptive_analytics import run as run_descriptive
            results = run_descriptive(coord.data)
            summary = results.get("summary", {})
            answer = f"Sales Summary:\n- Total Revenue: ${summary.get('total_revenue', 0):,.2f}\n- Total Quantity: {summary.get('total_quantity', 0):,.0f}\n- Average Order Value: ${summary.get('average_order_value', 0):,.2f}"
            analytics_type = "descriptive"
        
        return {
            "answer": answer,
            "analytics_type": analytics_type,
            "question": request.question
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze")
def analyze():
    """
    Run comprehensive analysis.
    
    Returns:
        Dictionary with full analysis results including narrative report
    """
    try:
        logger.info("Running comprehensive analysis...")
        coord = get_coordinator()
        results = coord.run_all()
        
        return {
            "narrative_report": results.get("narrative_report", ""),
            "summary_insights": results.get("summary_insights", {}),
            "execution_summary": results.get("execution_summary", {}),
            "raw_results": results.get("raw_results", {})
        }
        
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
def analyze_post(request: AnalysisRequest):
    """
    Run comprehensive analysis with optional custom data file.
    
    Args:
        request: AnalysisRequest with optional data file path
        
    Returns:
        Dictionary with full analysis results
    """
    try:
        if request.data_file_path:
            coord = CoordinatorAgent(data_file_path=request.data_file_path, verbose=False)
            if coord.data is None:
                coord._load_data()
        else:
            coord = get_coordinator()
        
        results = coord.run_all()
        
        return {
            "narrative_report": results.get("narrative_report", ""),
            "summary_insights": results.get("summary_insights", {}),
            "execution_summary": results.get("execution_summary", {}),
            "raw_results": results.get("raw_results", {})
        }
        
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "fastapi_endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

