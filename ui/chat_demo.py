"""
Streamlit Chat Demo UI
Interactive chat interface for the Sales Intelligence Agentic System.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import logging
import os
import re
from typing import Dict, Any, List, Optional

try:
    import altair as alt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    alt = None

# Set DeepSeek API key if not already set
if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "sk-64543b775f5546f0ba0e313366a1b550"

# Import coordinator agent directly (bypassing __init__.py to avoid legacy agent imports)
from src.agents.coordinator_agent import CoordinatorAgent
from src.descriptive_analytics import run as run_descriptive
from src.diagnostic_analytics import run as run_diagnostic
from src.predictive_analytics import run as run_predictive
from src.prescriptive_analytics import run as run_prescriptive
from src.agents.langchain_integration import synthesize_with_langchain_llmchain

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Sales Intelligence Agentic System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "coordinator" not in st.session_state:
    st.session_state.coordinator = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


def initialize_coordinator():
    """Initialize the coordinator agent."""
    try:
        if st.session_state.coordinator is None:
            with st.spinner("Loading coordinator agent and data..."):
                st.session_state.coordinator = CoordinatorAgent(verbose=False)
                # Load data on initialization
                if st.session_state.coordinator.data is None:
                    st.session_state.coordinator._load_data()
                st.session_state.data_loaded = True
            return True
        return True
    except Exception as e:
        st.error(f"Error initializing coordinator: {e}")
        return False


def extract_region(question: str, data) -> str:
    """
    Extract region name from question if mentioned.
    
    Args:
        question: User's question
        data: DataFrame to check available regions
        
    Returns:
        Region name if found, None otherwise
    """
    if data is None or 'region' not in data.columns or len(data) == 0:
        return None
    
    question_lower = question.lower()
    # Get unique regions with original casing
    available_regions = data['region'].dropna().unique().tolist()
    
    # Check if any region name is mentioned in the question
    for region in available_regions:
        region_lower = region.lower()
        # Use word boundary for better matching
        pattern = r'\b' + re.escape(region_lower) + r'\b'
        if re.search(pattern, question_lower):
            return region  # Return with original casing
    
    return None


def understand_intent_and_generate_query(question: str, available_regions: List[str], available_categories: List[str]) -> Dict[str, Any]:
    """
    Use DeepSeek to understand user's intent and generate a structured query specification.
    
    This ensures DeepSeek guides the system to find exactly the data needed for the answer.
    
    Args:
        question: User's question
        available_regions: List of available regions in the data
        available_categories: List of available categories in the data
        
    Returns:
        Dictionary with query specification:
        {
            "analytics_type": "descriptive" | "diagnostic" | "predictive" | "prescriptive" | "comprehensive",
            "region": region_name or None,
            "category": category_name or None,
            "specific_metrics": ["average_order_value", "total_revenue", etc.],
            "answer_style": "simple" | "detailed",
            "question_intent": "What the user is really asking for"
        }
    """
    try:
        from langchain_openai import ChatOpenAI
        import json
        import os
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key not found")
        
        # Build prompt for intent understanding
        prompt_text = (
            f"You are an intelligent query planner for a Sales Intelligence System.\n\n"
            f"User Question: {question}\n\n"
            f"Available Regions: {', '.join(available_regions)}\n"
            f"Available Categories: {', '.join(available_categories)}\n\n"
            f"Your task: Analyze the user's question and generate a structured query specification.\n\n"
            f"Instructions:\n"
            f"1. Determine the analytics type needed:\n"
            f"   - 'descriptive': Questions asking 'what happened' or about specific metrics (revenue, sales, average order value, etc.)\n"
            f"   - 'diagnostic': Questions asking 'why' something happened or about problems/issues\n"
            f"   - 'predictive': Questions asking 'what will happen' or about forecasts\n"
            f"   - 'prescriptive': Questions asking 'what should we do' or for recommendations\n"
            f"   - 'comprehensive': Questions asking for 'everything' or 'overall' analysis\n\n"
            f"2. Extract filters:\n"
            f"   - Identify if a specific region is mentioned (from available regions)\n"
            f"   - Identify if a specific category is mentioned (from available categories)\n"
            f"   - Set to null if not mentioned\n\n"
                         f"3. Identify specific metrics requested:\n"
             f"   - Examples: 'average_order_value', 'total_revenue', 'total_quantity', 'revenue_by_category', 'top_categories', 'revenue_by_region', 'top_regions', etc.\n"
             f"   - If asking for a single metric (like 'average order value'), list only that metric\n"
             f"   - If asking 'which region has highest/lowest', include 'revenue_by_region' or 'top_regions'\n"
             f"   - If asking 'which category has highest/lowest', include 'revenue_by_category' or 'top_categories'\n"
             f"   - If asking generally, list relevant metrics\n\n"
                         f"4. Determine answer style:\n"
             f"   - 'simple': Direct answer to a specific factual question (e.g., 'What is the average order value?', 'How many units were sold overall?', 'What is the total revenue?')\n"
             f"   - 'detailed': Requires explanation, analysis, or multiple insights (e.g., 'Why did sales drop?', 'What should we do?')\n"
             f"   - Note: Questions asking 'how many', 'how much', 'what is', 'what was' are typically 'simple' even if they contain 'overall' or 'all'\n\n"
            f"5. Understand the question intent: Summarize what the user is really asking for\n\n"
            f"Respond ONLY with a valid JSON object in this exact format:\n"
            f'{{"analytics_type": "descriptive", "region": "Khulna" or null, "category": null, "specific_metrics": ["average_order_value"], "answer_style": "simple", "question_intent": "User wants to know the average order value for Khulna region"}}\n\n'
            f"Your JSON response:"
        )
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.1,  # Low temperature for consistent parsing
            max_tokens=200,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
        # Generate query specification
        response = llm.invoke(prompt_text)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON from response (may have markdown code blocks)
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        query_spec = json.loads(response_text)
        
        return query_spec
        
    except Exception as e:
        logger.warning(f"DeepSeek intent understanding failed: {e}")
        # Fallback to basic classification
        return {
            "analytics_type": classify_question(question),
            "region": None,
            "category": None,
            "specific_metrics": [],
            "answer_style": "detailed",
            "question_intent": question
        }


def needs_natural_language_answer(question: str) -> bool:
    """
    Check if question needs a natural language answer (not just raw data).
    
    Args:
        question: User's question
        
    Returns:
        True if question asks for natural language explanation
    """
    question_lower = question.lower()
    # Questions that ask "which", "what", "who", "how many", etc. typically want natural language
    natural_language_indicators = [
        'which', 'what', 'who', 'how many', 'how much',
        'less', 'lowest', 'lowest', 'least', 'minimum',
        'more', 'highest', 'most', 'maximum', 'best', 'worst',
        'tell me', 'explain', 'describe', 'show me'
    ]
    return any(indicator in question_lower for indicator in natural_language_indicators)


def is_why_about_recommendation(question: str) -> bool:
    """
    Check if question is asking "why" about a recommendation.
    
    Args:
        question: User's question
        
    Returns:
        True if asking why about a recommendation
    """
    question_lower = question.lower()
    # Check for "why" and recommendation-related keywords
    has_why = any(word in question_lower for word in ['why', 'why should', 'why do we', 'reason'])
    has_recommendation = any(word in question_lower for word in ['increase', 'promotions', 'recommend', 'action', 'should'])
    return has_why and has_recommendation


def extract_recommendation_info(question: str, data) -> tuple:
    """
    Extract recommendation title and region from question.
    
    Args:
        question: User's question
        data: DataFrame to check available regions
        
    Returns:
        Tuple of (recommendation_title, region) or (None, None)
    """
    region = extract_region(question, data)
    question_lower = question.lower()
    
    # Try to extract recommendation title pattern
    # Common patterns: "Why should Increase promotions in X?"
    if 'increase promotions' in question_lower:
        return ("Increase promotions", region)
    elif 'increase' in question_lower and 'promotion' in question_lower:
        return ("Increase promotions", region)
    
    return (None, region)


def classify_question(question: str) -> str:
    """
    Classify user question to determine which analytics to run.
    
    Args:
        question: User's question
        
    Returns:
        Analytics type: 'descriptive', 'diagnostic', 'predictive', 'prescriptive', 'comprehensive'
    """
    question_lower = question.lower()
    
    # Check for simple factual questions first (questions asking "how many", "what is", etc.)
    # These should be descriptive even if they contain "overall"
    simple_factual_patterns = ['how many', 'how much', 'what is', 'what was', 'what are the']
    is_simple_factual = any(pattern in question_lower for pattern in simple_factual_patterns)
    
    # Comprehensive analysis keywords (check first, before descriptive)
    # "overall" should trigger comprehensive for questions like "overall sales", but NOT for simple factual questions
    comprehensive_keywords = ['everything', 'complete', 'full', 'comprehensive', 'overview', 'summary']
    has_comprehensive = any(word in question_lower for word in comprehensive_keywords)
    
    # "overall" and "all" should trigger comprehensive only if not a simple factual question
    if 'overall' in question_lower or 'all' in question_lower:
        if not is_simple_factual:
            return 'comprehensive'
    elif has_comprehensive:
        return 'comprehensive'
    
    # Descriptive keywords (removed 'overall' since it's in comprehensive)
    if any(word in question_lower for word in ['what happened', 'total', 'summary', 'statistics', 'aggregate', 'top', 'best', 'sales', 'revenue']):
        return 'descriptive'
    
    # Diagnostic keywords (includes "why" questions about recommendations)
    if any(word in question_lower for word in ['why', 'cause', 'reason', 'anomaly', 'dip', 'drop', 'problem', 'issue', 'diagnose']):
        # If asking "why" about a recommendation, still return diagnostic to show reasoning
        return 'diagnostic'
    
    # Predictive keywords
    if any(word in question_lower for word in ['forecast', 'predict', 'future', 'next month', 'next week', 'will', 'trend']):
        return 'predictive'
    
    # Prescriptive keywords
    if any(word in question_lower for word in ['recommend', 'action', 'suggest', 'what should', 'how to', 'improve', 'optimize']):
        return 'prescriptive'
    
    # Default to descriptive for sales/revenue questions
    if any(word in question_lower for word in ['sales', 'revenue', 'month', 'region']):
        return 'descriptive'
    
    # Default to comprehensive for general questions
    return 'comprehensive'


def format_descriptive_results(results: Dict[str, Any]) -> str:
    """Format descriptive analytics results for display."""
    output = []
    
    summary = results.get("summary", {})
    if summary:
        output.append("### 📈 Summary Statistics")
        output.append(f"- **Total Revenue:** ${summary.get('total_revenue', 0):,.2f}")
        output.append(f"- **Total Quantity:** {summary.get('total_quantity', 0):,.0f}")
        output.append(f"- **Average Order Value:** ${summary.get('average_order_value', 0):,.2f}")
        output.append("")
    
    top_categories = results.get("top_categories", [])
    if top_categories:
        output.append("### 🏆 Top Categories by Revenue")
        for i, cat in enumerate(top_categories[:5], 1):
            output.append(f"{i}. **{cat['category']}:** ${cat['total_revenue']:,.2f} ({cat['percentage']:.1f}%)")
        output.append("")
    
    by_region = results.get("by_region", {})
    if by_region:
        output.append("### 🌍 By Region")
        sorted_regions = sorted(
            by_region.items(),
            key=lambda x: x[1].get("total_revenue", 0),
            reverse=True
        )
        for region, data in sorted_regions[:5]:
            output.append(f"- **{region}:** ${data.get('total_revenue', 0):,.2f}")
        output.append("")
    
    return "\n".join(output)


def format_diagnostic_results(results: Dict[str, Any]) -> str:
    """Format diagnostic analytics results for display."""
    output = []
    
    insights = results.get("insights", [])
    if insights:
        output.append("### 🔍 Diagnostic Insights")
        for i, insight in enumerate(insights[:10], 1):
            output.append(f"{i}. {insight}")
        output.append("")
    
    region_dips = results.get("region_dips", [])
    if region_dips:
        output.append("### ⚠️ Region Dips Detected")
        for dip in region_dips[:5]:
            output.append(f"- **{dip['region']}:** {abs(dip['drop_percentage']):.1f}% drop on {dip['date']}")
        output.append("")
    
    channel_dips = results.get("channel_dips", [])
    if channel_dips:
        output.append("### ⚠️ Channel Dips Detected")
        for dip in channel_dips[:5]:
            output.append(f"- **{dip['channel']}:** {abs(dip['drop_percentage']):.1f}% drop on {dip['date']}")
        output.append("")
    
    if not insights and not region_dips and not channel_dips:
        output.append("No significant issues detected in the analysis period.")
    
    return "\n".join(output)


def format_predictive_results(results: Dict[str, Any]) -> str:
    """Format predictive analytics results for display."""
    output = []
    
    overall_forecast = results.get("overall_forecast")
    if overall_forecast:
        output.append("### 📊 Overall Forecast")
        output.append(f"- **Next Month Forecast:** ${overall_forecast.get('monthly_forecast', 0):,.2f}")
        output.append(f"- **Confidence Range:** ${overall_forecast.get('monthly_lower_bound', 0):,.2f} - ${overall_forecast.get('monthly_upper_bound', 0):,.2f}")
        trend = overall_forecast.get("trend", {})
        output.append(f"- **Trend:** {trend.get('direction', 'unknown').title()}")
        output.append("")
    
    forecasts = results.get("forecasts", {})
    if forecasts:
        output.append("### 🌍 Regional Forecasts")
        for region, forecast_data in list(forecasts.items())[:5]:
            output.append(f"\n**{region}:**")
            output.append(f"- Forecast: ${forecast_data.get('monthly_forecast', 0):,.2f}")
            output.append(f"- Trend: {forecast_data.get('trend', {}).get('direction', 'unknown').title()}")
        output.append("")
    
    return "\n".join(output)


def synthesize_answer_with_deepseek(question: str, analytics_type: str, results: Dict[str, Any], region: str = None, query_spec: Dict[str, Any] = None) -> str:
    """
    Use DeepSeek to generate a natural language answer for ANY type of analytics question.
    
    Args:
        question: User's question
        analytics_type: Type of analytics ('descriptive', 'diagnostic', 'predictive', 'prescriptive')
        results: Analytics results dictionary
        region: Optional region name if question is region-specific
        query_spec: Optional query specification from intent understanding (contains specific_metrics, answer_style, etc.)
        
        Returns:
        Natural language answer, or None if synthesis fails
    """
    try:
        from langchain_openai import ChatOpenAI
        import os
        import json
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key not found")
        
        # Format data based on analytics type
        # If query_spec is provided, extract ONLY the specific metrics requested
        data_summary = ""
        region_context = f" for {region} region" if region else " overall"
        specific_metrics = query_spec.get("specific_metrics", []) if query_spec else []
        answer_style = query_spec.get("answer_style", "detailed") if query_spec else "detailed"
        
        # Check if question is asking about regional comparisons
        question_lower = question.lower()
        is_regional_comparison = any(word in question_lower for word in ['which region', 'which regions', 'highest', 'lowest', 'best region', 'worst region', 'top region'])
        is_category_comparison = any(word in question_lower for word in ['which category', 'which categories', 'highest category', 'lowest category', 'top category'])
        
        if analytics_type == 'descriptive':
             summary = results.get("summary", {})
             top_categories = results.get("top_categories", [])
             by_region = results.get("by_region", {})
             
             # Always include regional data for regional comparison questions
             if is_regional_comparison and by_region:
                 if 'revenue_by_region' not in specific_metrics and 'top_regions' not in specific_metrics:
                     specific_metrics = specific_metrics + ['revenue_by_region']
             
             # Always include category data for category comparison questions
             if is_category_comparison and top_categories:
                 if 'revenue_by_category' not in specific_metrics and 'top_categories' not in specific_metrics:
                     specific_metrics = specific_metrics + ['revenue_by_category']
             
             # If specific metrics requested, extract only those
             if specific_metrics:
                 if 'average_order_value' in specific_metrics:
                     data_summary += f"Average Order Value: ${summary.get('average_order_value', 0):,.2f}\n"
                 if 'total_revenue' in specific_metrics:
                     data_summary += f"Total Revenue: ${summary.get('total_revenue', 0):,.2f}\n"
                 if 'total_quantity' in specific_metrics or 'units_sold' in specific_metrics:
                     data_summary += f"Total Quantity: {summary.get('total_quantity', 0):,.0f}\n"
                 if 'revenue_by_category' in specific_metrics or 'top_categories' in specific_metrics:
                     if top_categories:
                         data_summary += "\nCategories by Revenue:\n"
                         for cat in top_categories[:5]:
                             data_summary += f"- {cat['category']}: ${cat['total_revenue']:,.2f} ({cat['percentage']:.1f}%)\n"
                 if 'revenue_by_region' in specific_metrics:
                     if by_region:
                         data_summary += "\nRevenue by Region:\n"
                         sorted_regions = sorted(
                             by_region.items(),
                             key=lambda x: x[1].get("total_revenue", 0),
                             reverse=True
                         )
                         for reg, data in sorted_regions[:5]:
                             data_summary += f"- {reg}: ${data.get('total_revenue', 0):,.2f}\n"
             else:
                 # If no specific metrics, include all
                 data_summary = f"Total Revenue: ${summary.get('total_revenue', 0):,.2f}\n"
                 data_summary += f"Total Quantity: {summary.get('total_quantity', 0):,.0f}\n"
                 data_summary += f"Average Order Value: ${summary.get('average_order_value', 0):,.2f}\n\n"
                 
                 if top_categories:
                     data_summary += "Categories by Revenue:\n"
                     for cat in top_categories:
                         data_summary += f"- {cat['category']}: ${cat['total_revenue']:,.2f} ({cat['percentage']:.1f}%)\n"
                     data_summary += "\n"
                 
                 if by_region:
                     data_summary += "Revenue by Region:\n"
                     sorted_regions = sorted(
                         by_region.items(),
                         key=lambda x: x[1].get("total_revenue", 0),
                         reverse=True
                     )
                     for reg, data in sorted_regions[:5]:
                         data_summary += f"- {reg}: ${data.get('total_revenue', 0):,.2f}\n"
        
        elif analytics_type == 'diagnostic':
            insights = results.get("insights", [])
            region_dips = results.get("region_dips", [])
            channel_dips = results.get("channel_dips", [])
            
            if insights:
                data_summary += "Key Issues Detected:\n"
                for i, insight in enumerate(insights[:10], 1):
                    data_summary += f"{i}. {insight}\n"
                data_summary += "\n"
            
            if region_dips:
                data_summary += "Region Revenue Dips:\n"
                for dip in region_dips[:10]:
                    data_summary += f"- {dip['region']}: {abs(dip.get('drop_percentage', 0)):.1f}% drop on {dip.get('date', 'unknown')}\n"
                data_summary += "\n"
            
            if channel_dips:
                data_summary += "Channel Revenue Dips:\n"
                for dip in channel_dips[:10]:
                    data_summary += f"- {dip['channel']}: {abs(dip.get('drop_percentage', 0)):.1f}% drop on {dip.get('date', 'unknown')}\n"
                data_summary += "\n"
            
            if not insights and not region_dips and not channel_dips:
                data_summary = "No significant issues detected in the analysis period."
        
        elif analytics_type == 'predictive':
            overall_forecast = results.get("overall_forecast")
            forecasts = results.get("forecasts", {})
            
            if overall_forecast:
                data_summary += f"Overall Forecast:\n"
                data_summary += f"- Next Month: ${overall_forecast.get('monthly_forecast', 0):,.2f}\n"
                data_summary += f"- Confidence Range: ${overall_forecast.get('monthly_lower_bound', 0):,.2f} - ${overall_forecast.get('monthly_upper_bound', 0):,.2f}\n"
                trend = overall_forecast.get("trend", {})
                data_summary += f"- Trend: {trend.get('direction', 'unknown').title()}\n\n"
            
            if forecasts:
                data_summary += "Regional Forecasts:\n"
                for reg, forecast_data in list(forecasts.items())[:5]:
                    data_summary += f"- {reg}: ${forecast_data.get('monthly_forecast', 0):,.2f} ({forecast_data.get('trend', {}).get('direction', 'unknown')} trend)\n"
        
        elif analytics_type == 'prescriptive':
            recommendations = results.get("recommendations", [])
            
            if recommendations:
                high_priority = [r for r in recommendations if r.get("priority") == "high"]
                medium_priority = [r for r in recommendations if r.get("priority") == "medium"]
                low_priority = [r for r in recommendations if r.get("priority") == "low"]
                
                data_summary += "Recommendations:\n\n"
                
                if high_priority:
                    data_summary += "High Priority:\n"
                    for i, rec in enumerate(high_priority[:5], 1):
                        data_summary += f"{i}. {rec.get('title', 'N/A')}\n"
                        data_summary += f"   {rec.get('description', '')}\n"
                        data_summary += f"   Action: {rec.get('action', '')}\n\n"
                
                if medium_priority:
                    data_summary += "Medium Priority:\n"
                    for i, rec in enumerate(medium_priority[:3], 1):
                        data_summary += f"{i}. {rec.get('title', 'N/A')} - {rec.get('action', '')}\n"
                    data_summary += "\n"
                
                if low_priority:
                    data_summary += "Low Priority:\n"
                    for i, rec in enumerate(low_priority[:3], 1):
                        data_summary += f"{i}. {rec.get('title', 'N/A')} - {rec.get('action', '')}\n"
            else:
                data_summary = "No specific recommendations at this time."
        
        # Build prompt
        analytics_type_name = analytics_type.title()
        
        # Use answer_style from query_spec if available, otherwise detect from question
        if query_spec and query_spec.get("answer_style") == "simple":
            is_simple_answer = True
        else:
            # Fallback detection
            question_lower = question.lower()
            is_simple_answer = any(pattern in question_lower for pattern in [
                'what is', 'what was', 'how many', 'how much',
                'what are the', 'show me the', 'tell me the'
            ]) and not any(comprehensive_word in question_lower for comprehensive_word in [
                'overall', 'comprehensive', 'everything', 'all', 'complete', 'summary'
            ])
        
        if is_simple_answer:
            # For simple factual questions, give direct, concise answers
            # Special handling for category comparison questions
            if is_category_comparison:
                prompt_text = (
                    f"You are a Sales Intelligence Analyst. Answer the user's specific question directly and concisely.\n\n"
                    f"User Question: {question}\n\n"
                    f"Sales Data{region_context}:\n{data_summary}\n\n"
                    f"Instructions:\n"
                    f"- Look at the Categories by Revenue data provided above\n"
                    f"- Identify which category has the HIGHEST total revenue (if asking about highest/best/top)\n"
                    f"- Identify which category has the LOWEST total revenue (if asking about lowest/worst/less)\n"
                    f"- Answer directly with the category name and its revenue amount\n"
                    f"- Keep the answer brief (1-2 sentences maximum)\n"
                    f"- Use a friendly, conversational tone\n"
                    f"- Example: 'Electronics has the highest sales with $X revenue' or 'FMCG has the lowest sales with $Y revenue'\n\n"
                    f"Your direct answer:"
                )
            # Special handling for regional comparison questions
            elif is_regional_comparison:
                prompt_text = (
                    f"You are a Sales Intelligence Analyst. Answer the user's specific question directly and concisely.\n\n"
                    f"User Question: {question}\n\n"
                    f"Sales Data{region_context}:\n{data_summary}\n\n"
                    f"Instructions:\n"
                    f"- Look at the Revenue by Region data provided above\n"
                    f"- Identify which region has the HIGHEST total revenue (if asking about highest/best/top)\n"
                    f"- Identify which region has the LOWEST total revenue (if asking about lowest/worst)\n"
                    f"- Answer directly with the region name and its revenue amount\n"
                    f"- Keep the answer brief (1-2 sentences maximum)\n"
                    f"- Use a friendly, conversational tone\n"
                    f"- Example: 'Dhaka has the highest sales with $X revenue' or 'Khulna has the lowest sales with $Y revenue'\n\n"
                    f"Your direct answer:"
                )
            else:
                prompt_text = (
                    f"You are a Sales Intelligence Analyst. Answer the user's specific question directly and concisely.\n\n"
                    f"User Question: {question}\n\n"
                    f"Sales Data{region_context}:\n{data_summary}\n\n"
                    f"Instructions:\n"
                    f"- Answer ONLY what the user asked - be direct and specific\n"
                    f"- Provide the exact value/number from the data\n"
                    f"- Keep the answer brief (1-3 sentences maximum)\n"
                    f"- If the question asks for a specific metric (like average order value, total revenue, etc.), provide that exact number\n"
                    f"- Use a friendly, conversational tone\n"
                    f"- Do not provide extra information unless the question asks for it\n\n"
                    f"Your direct answer:"
                )
        else:
            # For complex questions, provide more detailed analysis
            prompt_text = (
                f"You are a Sales Intelligence Analyst answering questions about sales data.\n\n"
                f"User Question: {question}\n\n"
                f"Analytics Type: {analytics_type_name} Analysis{region_context}\n\n"
                f"Analytics Results:\n{data_summary}\n\n"
                f"Instructions:\n"
                f"- Answer the user's question in clear, natural language (under 150 words)\n"
                f"- Be specific and mention actual numbers, percentages, and data from the results\n"
                f"- Use a conversational, friendly tone\n"
                f"- Focus on answering what the user asked\n"
                f"- If the question asks about 'less' or 'lowest', identify and mention the lowest values\n"
                f"- If the question asks about 'more' or 'highest', identify and mention the highest values\n"
                f"- Make the answer engaging and easy to understand\n\n"
                f"Your answer:"
            )
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=300,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
        # Generate answer
        response = llm.invoke(prompt_text)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return answer
        
    except Exception as e:
        logger.warning(f"DeepSeek synthesis failed: {e}")
        return None


def synthesize_descriptive_answer(question: str, results: Dict[str, Any], region: str = None) -> str:
    """
    Use DeepSeek to generate a natural language answer for descriptive questions.
    
    Args:
        question: User's question
        results: Descriptive analytics results
        region: Optional region name if question is region-specific
        
    Returns:
        Natural language answer
    """
    try:
        from langchain_openai import ChatOpenAI
        import os
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key not found")
        
        # Format the data for the prompt
        summary = results.get("summary", {})
        top_categories = results.get("top_categories", [])
        by_region = results.get("by_region", {})
        
        # Build data summary for the prompt
        data_summary = f"Total Revenue: ${summary.get('total_revenue', 0):,.2f}\n"
        data_summary += f"Total Quantity: {summary.get('total_quantity', 0):,.0f}\n"
        data_summary += f"Average Order Value: ${summary.get('average_order_value', 0):,.2f}\n\n"
        
        if top_categories:
            data_summary += "Categories by Revenue:\n"
            for cat in top_categories:
                data_summary += f"- {cat['category']}: ${cat['total_revenue']:,.2f} ({cat['percentage']:.1f}%)\n"
            data_summary += "\n"
        
        if by_region:
            data_summary += "Revenue by Region:\n"
            sorted_regions = sorted(
                by_region.items(),
                key=lambda x: x[1].get("total_revenue", 0),
                reverse=True
            )
            for reg, data in sorted_regions[:5]:
                data_summary += f"- {reg}: ${data.get('total_revenue', 0):,.2f}\n"
        
        # Build prompt
        region_context = f" for {region} region" if region else " overall"
        prompt_text = (
            f"You are a Sales Intelligence Analyst answering questions about sales data.\n\n"
            f"User Question: {question}\n\n"
            f"Sales Data{region_context}:\n{data_summary}\n\n"
            f"Answer the user's question in clear, natural language (under 100 words). "
            f"Be specific and mention actual numbers and percentages from the data. "
            f"If the question asks about 'less' or 'lowest', identify the category/region with the lowest revenue. "
            f"If the question asks about 'more' or 'highest', identify the category/region with the highest revenue. "
            f"Use a conversational, friendly tone.\n\n"
            f"Your answer:"
        )
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=200,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
        # Generate answer
        response = llm.invoke(prompt_text)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return answer
        
    except Exception as e:
        logger.warning(f"DeepSeek synthesis failed: {e}")
        # Fallback to formatted results
        return None


def synthesize_recommendation_explanation(
    recommendation: Dict[str, Any],
    diagnostic_results: Dict[str, Any],
    region: str
) -> str:
    """
    Use DeepSeek to synthesize a natural language explanation for a recommendation.
    
    Args:
        recommendation: The recommendation dictionary
        diagnostic_results: Diagnostic analytics results for the region
        region: Region name
        
    Returns:
        Natural language explanation
    """
    try:
        # Prepare context for DeepSeek
        recommendation_title = recommendation.get("title", "")
        recommendation_reasoning = recommendation.get("reasoning", "")
        recommendation_description = recommendation.get("description", "")
        stats = recommendation.get("stats", {})
        
        # Format diagnostic data
        region_dips = [d for d in diagnostic_results.get("region_dips", []) if d.get("region") == region]
        insights = [i for i in diagnostic_results.get("insights", []) if region.lower() in i.lower()]
        
        # Build prompt for DeepSeek
        prompt_data = {
            "recommendation": recommendation_title,
            "reasoning": recommendation_reasoning,
            "description": recommendation_description,
            "num_dips": stats.get("num_dips", 0),
            "avg_drop": stats.get("avg_drop_pct", 0),
            "max_drop": stats.get("max_drop_pct", 0),
            "affected_categories": ", ".join(stats.get("affected_categories", [])),
            "region_dips_count": len(region_dips),
            "top_insights": "\n".join(insights[:5])
        }
        
        # Create a combined results dict for DeepSeek
        combined_results = {
            "recommendation_context": prompt_data,
            "diagnostic": {
                "region_dips": region_dips[:10],
                "insights": insights[:10]
            }
        }
        
        # Try to use DeepSeek for synthesis
        try:
            # Use a specialized prompt for recommendation explanations
            explanation = synthesize_recommendation_with_deepseek(combined_results, region, recommendation_title)
            if explanation and not explanation.startswith("Error"):
                return explanation
        except Exception as e:
            logger.warning(f"DeepSeek synthesis failed: {e}")
        
        # Fallback to formatted text
        output = []
        output.append(f"### 💡 Why {recommendation_title} in {region}?")
        output.append("")
        output.append(recommendation_reasoning or recommendation_description)
        output.append("")
        
        if stats:
            output.append(f"**Key Statistics:**")
            output.append(f"- {stats.get('num_dips', 0)} significant revenue drops detected")
            output.append(f"- Average drop: {stats.get('avg_drop_pct', 0):.1f}%")
            output.append(f"- Maximum drop: {stats.get('max_drop_pct', 0):.1f}%")
            if stats.get('affected_categories'):
                output.append(f"- Affected categories: {', '.join(stats.get('affected_categories', [])[:3])}")
            output.append("")
        
        if region_dips:
            output.append(f"**Recent Revenue Dips in {region}:**")
            for dip in region_dips[:5]:
                output.append(f"- {abs(dip.get('drop_percentage', 0)):.1f}% drop on {dip.get('date', 'unknown')}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Error synthesizing recommendation explanation: {e}")
        return f"Error generating explanation: {str(e)}"


def synthesize_recommendation_with_deepseek(combined_results: Dict[str, Any], region: str, recommendation_title: str) -> str:
    """
    Use DeepSeek to generate a natural language explanation for a recommendation.
    
    Args:
        combined_results: Combined recommendation and diagnostic data
        region: Region name
        recommendation_title: Title of the recommendation
        
    Returns:
        Natural language explanation
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        import os
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key not found")
        
        rec_context = combined_results.get("recommendation_context", {})
        diag_data = combined_results.get("diagnostic", {})
        
        # Build specialized prompt for recommendation explanation
        prompt_text = (
            f"You are a Sales Intelligence Analyst explaining why a recommendation is needed.\n\n"
            f"User Question: Why should {recommendation_title} in {region}?\n\n"
            f"Recommendation Details:\n"
            f"- Title: {rec_context.get('recommendation', 'N/A')}\n"
            f"- Reasoning: {rec_context.get('reasoning', 'N/A')}\n"
            f"- Description: {rec_context.get('description', 'N/A')}\n"
            f"- Number of revenue dips: {rec_context.get('num_dips', 0)}\n"
            f"- Average drop: {rec_context.get('avg_drop', 0):.1f}%\n"
            f"- Maximum drop: {rec_context.get('max_drop', 0):.1f}%\n"
            f"- Affected categories: {rec_context.get('affected_categories', 'N/A')}\n\n"
            f"Diagnostic Data:\n"
            f"- Region dips: {len(diag_data.get('region_dips', []))} detected\n"
            f"- Key insights:\n{rec_context.get('top_insights', 'N/A')}\n\n"
            f"Write a clear, engaging explanation (under 150 words) that:\n"
            f"1. Explains WHY this recommendation is needed for {region}\n"
            f"2. Uses the specific statistics (number of dips, drop percentages)\n"
            f"3. Mentions affected categories\n"
            f"4. Explains the business impact\n"
            f"5. Uses a conversational, easy-to-understand tone\n\n"
            f"Format:\n"
            f"📍 **Why {recommendation_title} in {region}?**\n\n"
            f"[Your explanation here]"
        )
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=300,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        
        # Generate explanation
        response = llm.invoke(prompt_text)
        explanation = response.content if hasattr(response, 'content') else str(response)
        
        return explanation
        
    except Exception as e:
        logger.warning(f"DeepSeek synthesis error: {e}")
        raise


def format_prescriptive_results(results: Dict[str, Any]) -> str:
    """Format prescriptive analytics results for display."""
    output = []
    
    recommendations = results.get("recommendations", [])
    if recommendations:
        output.append("### 💡 Recommendations")
        
        high_priority = [r for r in recommendations if r.get("priority") == "high"]
        medium_priority = [r for r in recommendations if r.get("priority") == "medium"]
        low_priority = [r for r in recommendations if r.get("priority") == "low"]
        
        if high_priority:
            output.append("\n#### 🔴 High Priority")
            for i, rec in enumerate(high_priority[:5], 1):
                output.append(f"{i}. **{rec.get('title', 'N/A')}**")
                output.append(f"   - Category: {rec.get('category', 'N/A')}")
                output.append(f"   - {rec.get('description', '')}")
                output.append(f"   - **Action:** {rec.get('action', '')}")
                output.append("")
        
        if medium_priority:
            output.append("\n#### 🟡 Medium Priority")
            for i, rec in enumerate(medium_priority[:3], 1):
                output.append(f"{i}. **{rec.get('title', 'N/A')}** - {rec.get('action', '')}")
        
        if low_priority:
            output.append("\n#### 🟢 Low Priority")
            for i, rec in enumerate(low_priority[:3], 1):
                output.append(f"{i}. **{rec.get('title', 'N/A')}** - {rec.get('action', '')}")
    else:
        output.append("No specific recommendations at this time.")
    
    return "\n".join(output)


def build_chart_payloads(
    analytics_type: str,
    results: Dict[str, Any],
    query_spec: Optional[Dict[str, Any]] = None,
    region: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Create lightweight chart descriptors for chat rendering."""
    charts: List[Dict[str, Any]] = []

    if not results:
        return charts

    specific_metrics = set(query_spec.get("specific_metrics", [])) if isinstance(query_spec, dict) else set()
    answer_style = query_spec.get("answer_style") if isinstance(query_spec, dict) else None
    if answer_style == "simple":
        simple_scalar_metrics = {
            "total_revenue",
            "total_quantity",
            "total_units",
            "units_sold",
            "average_order_value",
            "avg_order_value"
        }
        if specific_metrics and specific_metrics.issubset(simple_scalar_metrics):
            return charts

    if analytics_type == "descriptive":
        by_region = results.get("by_region", {})
        if by_region and not region:
            # Show overall revenue by region when no region filter is applied or metric requires it
            include_region_chart = (
                not specific_metrics
                or bool({"revenue_by_region", "top_regions"} & specific_metrics)
            )
            if include_region_chart:
                data = [
                    {
                        "label": reg,
                        "value": float(info.get("total_revenue", 0))
                    }
                    for reg, info in sorted(
                        by_region.items(),
                        key=lambda x: x[1].get("total_revenue", 0),
                        reverse=True
                    )
                ]
                if data:
                    charts.append({
                        "type": "bar",
                        "title": "Revenue by Region",
                        "data": data,
                        "label_field": "label",
                        "value_field": "value",
                        "height": 320
                    })

        top_categories = results.get("top_categories", [])
        include_category_chart = False
        if top_categories:
            if specific_metrics:
                include_category_chart = bool({"revenue_by_category", "top_categories"} & specific_metrics)
            else:
                include_category_chart = True

        if include_category_chart and top_categories:
            data = [
                {
                    "label": cat.get("category", "Unknown"),
                    "value": float(cat.get("total_revenue", 0))
                }
                for cat in top_categories[:10]
            ]
            if data:
                title_suffix = f" - {region}" if region else ""
                charts.append({
                    "type": "bar",
                    "title": f"Top Categories by Revenue{title_suffix}",
                    "data": data,
                    "label_field": "label",
                    "value_field": "value",
                    "height": 320
                })

    elif analytics_type == "diagnostic":
        region_dips = results.get("region_dips", [])
        if region_dips:
            stacked_data = []
            severity_by_region = {}
            for dip in region_dips[:100]:
                drop_value = abs(float(dip.get("drop_percentage", 0)))
                stacked_data.append({
                    "date": dip.get("date"),
                    "group": dip.get("region", "Unknown"),
                    "severity": drop_value
                })
                region = dip.get("region", "Unknown")
                summary_entry = severity_by_region.setdefault(region, {"label": region, "value": 0.0, "count": 0})
                summary_entry["value"] += drop_value
                summary_entry["count"] += 1

            if stacked_data:
                charts.append({
                    "type": "stacked_bar",
                    "title": "Revenue Drop Severity by Region (7-day dips)",
                    "data": stacked_data,
                    "x_field": "date",
                    "y_field": "severity",
                    "color_field": "group",
                    "x_title": "Dip Date",
                    "y_title": "Drop % (absolute)",
                    "color_title": "Region",
                    "height": 320
                })

            if severity_by_region:
                severity_data = [
                    {
                        "label": region,
                        "value": round(entry["value"], 1)
                    }
                    for region, entry in sorted(
                        severity_by_region.items(),
                        key=lambda item: item[1]["value"],
                        reverse=True
                    )
                ]
                charts.append({
                    "type": "bar",
                    "title": "Total Dip Severity by Region",
                    "data": severity_data,
                    "label_field": "label",
                    "value_field": "value",
                    "height": 260
                })

        channel_dips = results.get("channel_dips", [])
        if channel_dips:
            channel_data = []
            for dip in channel_dips[:80]:
                channel_data.append({
                    "date": dip.get("date"),
                    "group": dip.get("channel", "Unknown"),
                    "severity": abs(float(dip.get("drop_percentage", 0)))
                })

            if channel_data:
                charts.append({
                    "type": "stacked_bar",
                    "title": "Drop Severity by Sales Channel",
                    "data": channel_data,
                    "x_field": "date",
                    "y_field": "severity",
                    "color_field": "group",
                    "x_title": "Dip Date",
                    "y_title": "Drop % (absolute)",
                    "color_title": "Channel",
                    "height": 300
                })

        anomalies = results.get("anomalies", [])
        if anomalies:
            heatmap_counts: Dict[tuple, int] = {}
            for anomaly in anomalies[:200]:
                region_name = anomaly.get("region", "Unknown") or "Unknown"
                date_key = str(anomaly.get("date", ""))[:10]
                key = (region_name, date_key)
                heatmap_counts[key] = heatmap_counts.get(key, 0) + 1

            heatmap_data = [
                {
                    "region": key[0],
                    "date": key[1],
                    "count": count
                }
                for key, count in heatmap_counts.items()
            ]

            if heatmap_data:
                charts.append({
                    "type": "heatmap",
                    "title": "Anomaly Counts by Region and Date",
                    "data": heatmap_data,
                    "x_field": "date",
                    "y_field": "region",
                    "color_field": "count",
                    "x_title": "Date",
                    "y_title": "Region",
                    "color_title": "Anomaly Count",
                    "color_scheme": "oranges",
                    "height": 320
                })


    elif analytics_type == "predictive":
        forecasts = results.get("forecasts", {})
        target_region = None
        if isinstance(query_spec, dict) and query_spec.get("region"):
            target_region = query_spec["region"]
        if not target_region:
            target_region = region

        if target_region and target_region in forecasts:
            forecast = forecasts[target_region].get("daily_forecast", {})
            dates = forecast.get("dates", [])
            values = forecast.get("values", [])
            if dates and values:
                data = [
                    {
                        "label": dates[i],
                        "value": float(values[i])
                    }
                    for i in range(min(len(dates), len(values)))
                ]
                charts.append({
                    "type": "line",
                    "title": f"Daily Revenue Forecast - {target_region}",
                    "data": data,
                    "label_field": "label",
                    "value_field": "value",
                    "height": 320
                })

    return charts


def render_chart(chart_info: Dict[str, Any]) -> None:
    """Render a chart inside a chat message based on stored metadata."""
    data = chart_info.get("data")
    if not data:
        return

    df = pd.DataFrame(data)
    label_field = chart_info.get("label_field", "label")
    value_field = chart_info.get("value_field", "value")

    chart_type = chart_info.get("type", "bar")

    if chart_type in {"bar", "line"} and (label_field not in df.columns or value_field not in df.columns):
        return

    if chart_info.get("title"):
        st.markdown(f"**{chart_info['title']}**")

    height = chart_info.get("height", 320)

    if chart_type == "stacked_bar" and alt:
        x_field = chart_info.get("x_field")
        y_field = chart_info.get("y_field") or value_field
        color_field = chart_info.get("color_field")
        if not all(field in df.columns for field in [x_field, y_field, color_field] if field):
            return

        # Attempt to parse dates for x-axis if they look like dates
        if x_field and chart_info.get("x_parse_dates", True):
            try:
                df[x_field] = pd.to_datetime(df[x_field])
                x_encoding = alt.X(f"{x_field}:T", title=chart_info.get("x_title", x_field.title()))
            except Exception:
                x_encoding = alt.X(f"{x_field}:N", title=chart_info.get("x_title", x_field.title()))
        else:
            x_encoding = alt.X(f"{x_field}:N", title=chart_info.get("x_title", x_field.title()))

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=x_encoding,
                y=alt.Y(f"{y_field}:Q", title=chart_info.get("y_title", chart_info.get("value_label", y_field.title()))),
                color=alt.Color(f"{color_field}:N", title=chart_info.get("color_title", color_field.title() if color_field else "Group")),
                tooltip=[x_field, color_field, y_field]
            )
            .properties(height=height, width=chart_info.get("width", 520))
        )
        st.altair_chart(chart, use_container_width=True)
        return

    if chart_type == "heatmap" and alt:
        x_field = chart_info.get("x_field")
        y_field = chart_info.get("y_field")
        color_field = chart_info.get("color_field") or value_field
        if not all(field in df.columns for field in [x_field, y_field, color_field] if field):
            return

        if x_field and chart_info.get("x_parse_dates", True):
            try:
                df[x_field] = pd.to_datetime(df[x_field])
                x_encoding = alt.X(f"{x_field}:T", title=chart_info.get("x_title", x_field.title()))
            except Exception:
                x_encoding = alt.X(f"{x_field}:N", title=chart_info.get("x_title", x_field.title()))
        else:
            x_encoding = alt.X(f"{x_field}:N", title=chart_info.get("x_title", x_field.title()))

        chart = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=x_encoding,
                y=alt.Y(f"{y_field}:N", title=chart_info.get("y_title", y_field.title() if y_field else "")),
                color=alt.Color(
                    f"{color_field}:Q",
                    title=chart_info.get("color_title", color_field.title()),
                    scale=alt.Scale(scheme=chart_info.get("color_scheme", "reds"))
                ),
                tooltip=[x_field, y_field, color_field]
            )
            .properties(height=height, width=chart_info.get("width", 520))
        )
        st.altair_chart(chart, use_container_width=True)
        return

    # Fallback to simple charts if Altair not available or unsupported type
    if label_field in df.columns and value_field in df.columns:
        df_plot = df.sort_values(value_field, ascending=False).set_index(label_field)
        if chart_type == "line":
            st.line_chart(df_plot[value_field], height=height)
        else:
            st.bar_chart(df_plot[value_field], height=height)
    else:
        st.info("Chart data unavailable.")


def process_question(question: str) -> Dict[str, Any]:
    """Process user question and return appropriate analytics results."""
    coordinator = st.session_state.coordinator
    
    # STEP 1: Use DeepSeek to understand intent and generate query specification
    try:
        # Get available regions and categories
        if coordinator.data is not None:
            available_regions = coordinator.data['region'].dropna().unique().tolist() if 'region' in coordinator.data.columns else []
            available_categories = coordinator.data['category'].dropna().unique().tolist() if 'category' in coordinator.data.columns else []
        else:
            available_regions = []
            available_categories = []
        
        # Use DeepSeek to understand intent
        with st.spinner("Understanding your question..."):
            query_spec = understand_intent_and_generate_query(question, available_regions, available_categories)
    except Exception as e:
        logger.warning(f"Intent understanding failed, using fallback: {e}")
        query_spec = None
    
    # Extract region from query_spec if available, otherwise fallback to extract_region
    if query_spec and query_spec.get("region"):
        region = query_spec["region"]
    else:
        region = extract_region(question, coordinator.data)
 
    # Normalize region to a single string
    if isinstance(region, list):
        region = region[0] if region else None
    if isinstance(region, dict):
        region = region.get("name") or region.get("value")
    if region is not None and not isinstance(region, str):
        region = str(region)

    # Filter data by region if specified
    data_to_use = coordinator.data
    if region:
        data_to_use = coordinator.data[coordinator.data['region'] == region].copy()
        if len(data_to_use) == 0:
            return {
                "type": "error",
                "response": f"No data found for region: {region}",
                "raw_results": {}
            }
    
    # Use analytics_type from query_spec if available, otherwise classify
    if query_spec and query_spec.get("analytics_type"):
        analytics_type = query_spec["analytics_type"]
    else:
        analytics_type = classify_question(question)
    
    try:
        if analytics_type == 'comprehensive':
            with st.spinner(f"Running comprehensive analysis{' for ' + region if region else ''}..."):
                # Use filtered data if region specified, otherwise use all data
                if region:
                    # Create a temporary coordinator with filtered data
                    temp_coordinator = CoordinatorAgent(verbose=False)
                    temp_coordinator.data = data_to_use
                    # Ensure data_loader is set to avoid errors
                    if temp_coordinator.data_loader is None:
                        from src.data_loader import DataLoader
                        temp_coordinator.data_loader = DataLoader()
                    results = temp_coordinator.run_all()
                    # Add region info to narrative
                    narrative = results.get("narrative_report", "")
                    if narrative:
                        results["narrative_report"] = f"📍 **Analysis for {region} Region**\n\n{narrative}"
                else:
                    # Use existing coordinator for overall analysis
                    results = coordinator.run_all()
                return {
                    "type": "comprehensive",
                    "response": results.get("narrative_report", "Analysis completed."),
                    "raw_results": results
                }
        
        elif analytics_type == 'descriptive':
            with st.spinner(f"Running descriptive analytics{' for ' + region if region else ''}..."):
                results = run_descriptive(data_to_use)
 
                # Always use DeepSeek to generate natural language answer with query_spec
                natural_answer = synthesize_answer_with_deepseek(question, 'descriptive', results, region, query_spec)
                if natural_answer:
                    # Add region context if applicable
                    if region:
                        response_text = f"📍 **{region} Region Analysis**\n\n{natural_answer}"
                    else:
                        response_text = natural_answer
 
                    charts = build_chart_payloads('descriptive', results, query_spec, region)
                    return {
                        "type": "descriptive",
                        "response": response_text,
                        "raw_results": results,
                        "deepseek_used": True,
                        "charts": charts
                    }
 
                # Fallback to formatted results if DeepSeek fails
                formatted = format_descriptive_results(results)
                if region:
                    formatted = f"📍 **Sales Data for {region} Region**\n\n{formatted}"
                charts = build_chart_payloads('descriptive', results, query_spec, region)
                return {
                    "type": "descriptive",
                    "response": formatted,
                    "raw_results": results,
                    "deepseek_used": False,
                    "charts": charts
                }
        
        elif analytics_type == 'diagnostic':
            # Check if asking "why" about a recommendation
            if is_why_about_recommendation(question):
                rec_title, rec_region = extract_recommendation_info(question, coordinator.data)
                if rec_title and rec_region:
                    with st.spinner(f"Generating explanation for {rec_title} in {rec_region}..."):
                        # Get prescriptive recommendations
                        desc_results = run_descriptive(data_to_use)
                        diag_results = run_diagnostic(data_to_use)
                        pred_results = run_predictive(data_to_use)
                        presc_results = run_prescriptive(
                            data_to_use,
                            diagnostic_results=diag_results,
                            predictive_results=pred_results,
                            descriptive_results=desc_results
                        )
                        
                        # Find matching recommendation
                        recommendations = presc_results.get("recommendations", [])
                        matching_rec = None
                        rec_title_lower = rec_title.lower()
                        rec_region_lower = rec_region.lower()
                        
                        for rec in recommendations:
                            rec_title_full = rec.get("title", "").lower()
                            # Check if this recommendation matches (title contains both the action and region)
                            if rec_title_lower in rec_title_full and rec_region_lower in rec_title_full:
                                matching_rec = rec
                                break
                        
                        # If not found, try matching by region only
                        if not matching_rec:
                            for rec in recommendations:
                                rec_title_full = rec.get("title", "").lower()
                                if rec_region_lower in rec_title_full:
                                    matching_rec = rec
                                    break
                        
                        if matching_rec:
                            # Use DeepSeek to synthesize explanation
                            explanation = synthesize_recommendation_explanation(
                                matching_rec,
                                diag_results,
                                rec_region
                            )
                            return {
                                "type": "diagnostic",
                                "response": explanation,
                                "raw_results": {
                                    "recommendation": matching_rec,
                                    "diagnostic": diag_results
                                },
                                "deepseek_used": True
                            }
                        else:
                            # Fallback to regular diagnostic if recommendation not found
                            formatted = format_diagnostic_results(diag_results)
                            formatted = f"📍 **Diagnostic Analysis for {rec_region} Region**\n\n{formatted}\n\n*Note: Could not find matching recommendation, showing diagnostic data.*"
                            return {
                                "type": "diagnostic",
                                "response": formatted,
                                "raw_results": diag_results
                            }
            
            # Regular diagnostic analysis
            with st.spinner(f"Running diagnostic analytics{' for ' + region if region else ''}..."):
                results = run_diagnostic(data_to_use)
                
                # Always use DeepSeek to generate natural language answer
                natural_answer = synthesize_answer_with_deepseek(question, 'diagnostic', results, region, query_spec)
                if natural_answer:
                    if region:
                        response_text = f"📍 **Diagnostic Analysis for {region} Region**\n\n{natural_answer}"
                    else:
                        response_text = natural_answer
                    
                    return {
                        "type": "diagnostic",
                        "response": response_text,
                        "raw_results": results,
                        "deepseek_used": True
                    }
                
                # Fallback to formatted results if DeepSeek fails
                formatted = format_diagnostic_results(results)
                if region:
                    formatted = f"📍 **Diagnostic Analysis for {region} Region**\n\n{formatted}"
                return {
                    "type": "diagnostic",
                    "response": formatted,
                    "raw_results": results,
                    "deepseek_used": False
                }
        
        elif analytics_type == 'predictive':
            with st.spinner(f"Running predictive analytics{' for ' + region if region else ''}..."):
                results = run_predictive(data_to_use)
                
                # Always use DeepSeek to generate natural language answer
                natural_answer = synthesize_answer_with_deepseek(question, 'predictive', results, region, query_spec)
                if natural_answer:
                    if region:
                        response_text = f"📍 **Forecast for {region} Region**\n\n{natural_answer}"
                    else:
                        response_text = natural_answer
 
                    charts = build_chart_payloads('predictive', results, query_spec, region)
                    return {
                        "type": "predictive",
                        "response": response_text,
                        "raw_results": results,
                        "deepseek_used": True,
                        "charts": charts
                    }
 
                # Fallback to formatted results if DeepSeek fails
                formatted = format_predictive_results(results)
                if region:
                    formatted = f"📍 **Forecast for {region} Region**\n\n{formatted}"
                charts = build_chart_payloads('predictive', results, query_spec, region)
                return {
                    "type": "predictive",
                    "response": formatted,
                    "raw_results": results,
                    "deepseek_used": False,
                    "charts": charts
                }
        
        elif analytics_type == 'prescriptive':
            with st.spinner(f"Running prescriptive analytics{' for ' + region if region else ''}..."):
                # Need descriptive and diagnostic for prescriptive
                desc_results = run_descriptive(data_to_use)
                diag_results = run_diagnostic(data_to_use)
                pred_results = run_predictive(data_to_use)
                
                results = run_prescriptive(
                    data_to_use,
                    diagnostic_results=diag_results,
                    predictive_results=pred_results,
                    descriptive_results=desc_results
                )
                
                # Always use DeepSeek to generate natural language answer
                natural_answer = synthesize_answer_with_deepseek(question, 'prescriptive', results, region, query_spec)
                if natural_answer:
                    if region:
                        response_text = f"📍 **Recommendations for {region} Region**\n\n{natural_answer}"
                    else:
                        response_text = natural_answer
                    
                    return {
                        "type": "prescriptive",
                        "response": response_text,
                        "raw_results": results,
                        "deepseek_used": True
                    }
                
                # Fallback to formatted results if DeepSeek fails
                formatted = format_prescriptive_results(results)
                if region:
                    formatted = f"📍 **Recommendations for {region} Region**\n\n{formatted}"
                return {
                    "type": "prescriptive",
                    "response": formatted,
                    "raw_results": results,
                    "deepseek_used": False
                }
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return {
            "type": "error",
            "response": f"Error processing your question: {str(e)}",
            "raw_results": {}
        }


def main():
    """Main Streamlit app."""
    # Title and header
    st.title("📊 Sales Intelligence Agentic System")
    st.markdown("Ask questions about your sales data and get intelligent insights!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Initialize coordinator
        if st.button("Initialize System", type="primary"):
            if initialize_coordinator():
                st.success("✅ System initialized successfully!")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Hello! I'm your Sales Intelligence Assistant. Ask me questions about:\n- What happened (descriptive analytics)\n- Why it happened (diagnostic analytics)\n- What will happen (predictive analytics)\n- What should we do (prescriptive analytics)\n\nOr ask for a comprehensive analysis!"
                })
        
        if st.session_state.data_loaded:
            st.success("✅ Data loaded")
            
            # Show data summary
            if st.session_state.coordinator and st.session_state.coordinator.data is not None:
                data = st.session_state.coordinator.data
                st.info(f"📊 **{len(data)}** records loaded")
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "What is the total revenue?",
            "Which region has the highest sales?",
            "Why did sales drop in Rajshahi?",
            "What is the forecast for next month?",
            "What actions should we take?",
            "Give me a comprehensive analysis"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q}", use_container_width=True):
                st.session_state.user_input = q
        
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        if st.session_state.coordinator:
            st.success("Coordinator Agent: Ready")
            # Check if DeepSeek is available
            if os.getenv("DEEPSEEK_API_KEY"):
                st.success("🤖 DeepSeek AI: Active")
                st.info("💡 **Tip:** All questions are answered using DeepSeek AI for natural language responses!")
            else:
                st.warning("🤖 DeepSeek AI: Not configured")
        else:
            st.warning("Coordinator Agent: Not initialized")
    
    # Main chat interface
    if not st.session_state.coordinator:
        st.warning("⚠️ Please initialize the system from the sidebar first.")
        st.info("Click 'Initialize System' to load the coordinator agent and sales data.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Render any charts associated with the message
            charts = message.get("charts")
            if charts:
                for chart in charts:
                    render_chart(chart)
            
            # Show raw results in expander if available
            if "raw_results" in message and message["raw_results"]:
                with st.expander("📋 View Raw Data"):
                    st.json(message["raw_results"])
    
    # User input
    user_input = st.chat_input("Ask a question about your sales data...")
    
    # Handle example questions from sidebar
    if "user_input" in st.session_state:
        user_input = st.session_state.user_input
        del st.session_state.user_input
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process question
        response = process_question(user_input)
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["response"],
            "raw_results": response.get("raw_results", {}),
            "analytics_type": response.get("type", "unknown"),
            "charts": response.get("charts", [])
        })
        
        # Display response
        with st.chat_message("assistant"):
            # Format long text with proper paragraph spacing
            formatted_response = response["response"].replace("\n", "  \n")
            st.markdown(formatted_response)

            # Render charts for the current response if available
            if response.get("charts"):
                for chart in response["charts"]:
                    render_chart(chart)
            
            # Show analytics type badge
            analytics_type = response.get("type", "unknown")
            type_colors = {
                "descriptive": "🔵",
                "diagnostic": "🟡",
                "predictive": "🟢",
                "prescriptive": "🟣",
                "comprehensive": "🔴",
                "error": "❌"
            }
            
            # Check if DeepSeek was used
            deepseek_used = response.get("deepseek_used", False)
            
            # Also check for comprehensive analysis
            if analytics_type == "comprehensive" and not deepseek_used:
                raw_results = response.get("raw_results", {})
                narrative = raw_results.get("narrative_report", "") or response.get("response", "")
                # DeepSeek-generated narratives don't start with the fallback message
                if narrative and not narrative.startswith("Sales intelligence analysis completed"):
                    deepseek_used = True
            
            badge_text = f"{type_colors.get(analytics_type, '📊')} {analytics_type.title()} Analytics"
            if deepseek_used:
                badge_text += " | 🤖 Powered by DeepSeek AI"
            st.caption(badge_text)
            
            # Show raw results in expander
            if response.get("raw_results"):
                with st.expander("📋 View Raw Data"):
                    st.json(response["raw_results"])
        
        # Rerun to update chat
        st.rerun()


if __name__ == "__main__":
    main()
