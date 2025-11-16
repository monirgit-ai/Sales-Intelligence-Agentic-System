"""
LangChain Integration Module
Demonstrates LLM-based reasoning and contextual intelligence for narrative synthesis.
Uses LangChain LLMChain for actual LLM reasoning calls.
"""

from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Try to import LangChain components
# Support both old and new LangChain import paths
try:
    # LangChain 1.0+ (new structure)
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        LANGCHAIN_NEW_VERSION = True
    except ImportError:
        LANGCHAIN_NEW_VERSION = False
    
    # Try old structure for backward compatibility
    try:
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate as OldPromptTemplate
        from langchain.llms import OpenAI
        from langchain.chat_models import ChatOpenAI as OldChatOpenAI
        LANGCHAIN_OLD_VERSION = True
    except ImportError:
        LANGCHAIN_OLD_VERSION = False
    
    if LANGCHAIN_NEW_VERSION or LANGCHAIN_OLD_VERSION:
        LANGCHAIN_AVAILABLE = True
    else:
        LANGCHAIN_AVAILABLE = False
        raise ImportError("Could not import LangChain")
        
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_NEW_VERSION = False
    LANGCHAIN_OLD_VERSION = False
    logger.warning("LangChain not installed. Install with: pip install langchain openai")


def synthesize_with_langchain_llmchain(
    combined_results: Dict[str, Any],
    llm_provider: Optional[str] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Uses LangChain LLMChain for contextual reasoning summary.
    
    This is the explicit LangChain integration that makes actual LLM calls.
    Supports OpenAI, DeepSeek, and other OpenAI-compatible providers.
    
    Args:
        combined_results: Dictionary with descriptive, diagnostic, predictive, and prescriptive results
        llm_provider: Provider name ("openai", "deepseek", or None for auto-detect)
        model_name: Model name to use (defaults based on provider)
        
    Returns:
        Natural language narrative report from LLM
    """
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain not available. Using internal summarizer.")
        return _fallback_synthesis(combined_results)
    
    # Detect provider from environment or use parameter
    if llm_provider is None:
        # Check for DeepSeek API key first, then OpenAI
        if os.getenv("DEEPSEEK_API_KEY"):
            llm_provider = "deepseek"
        elif os.getenv("OPENAI_API_KEY"):
            llm_provider = "openai"
        else:
            logger.warning("No API key found for OpenAI or DeepSeek. Using fallback.")
            return _fallback_synthesis(combined_results)
    
    try:
        # Initialize LLM based on provider
        llm = None
        
        if llm_provider.lower() == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                logger.error("DEEPSEEK_API_KEY not set")
                return _fallback_synthesis(combined_results)
            
            # DeepSeek endpoint and model
            api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
            default_model = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            
            try:
                # Use ChatOpenAI with DeepSeek endpoint (OpenAI-compatible)
                if LANGCHAIN_NEW_VERSION:
                    # LangChain 1.0+ uses model parameter instead of model_name
                    llm = ChatOpenAI(
                        model=default_model,
                        temperature=0.2,
                        max_tokens=250,  # Reduced for concise 120-word output
                        api_key=api_key,
                        base_url=api_base
                    )
                elif LANGCHAIN_OLD_VERSION:
                    # LangChain 0.x uses model_name and openai_api_key parameters
                    llm = OldChatOpenAI(
                        model_name=default_model,
                        temperature=0.2,
                        max_tokens=250,  # Reduced for concise 120-word output
                        openai_api_key=api_key,
                        openai_api_base=api_base
                    )
                logger.info(f"Initialized DeepSeek LLM: {default_model}")
            except Exception as e:
                logger.error(f"DeepSeek initialization failed: {e}")
                return _fallback_synthesis(combined_results)
        
        elif llm_provider.lower() == "openai":
            # Standard OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not set")
                return _fallback_synthesis(combined_results)
            
            default_model = model_name or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            
            try:
                if LANGCHAIN_NEW_VERSION:
                    llm = ChatOpenAI(
                        model=default_model,
                        temperature=0.2,
                        max_tokens=1000
                    )
                elif LANGCHAIN_OLD_VERSION:
                    llm = OldChatOpenAI(
                        model_name=default_model,
                        temperature=0.2,
                        max_tokens=1000
                    )
                logger.info(f"Initialized OpenAI LLM: {default_model}")
            except Exception as e:
                logger.warning(f"ChatOpenAI initialization failed: {e}. Trying OpenAI...")
                try:
                    if LANGCHAIN_OLD_VERSION:
                        llm = OpenAI(model_name=default_model, temperature=0.2)
                        logger.info(f"Initialized OpenAI LLM (legacy): {default_model}")
                    else:
                        raise Exception("Legacy OpenAI not available in LangChain 1.0")
                except Exception as e2:
                    logger.error(f"OpenAI initialization failed: {e2}")
                    return _fallback_synthesis(combined_results)
        
        else:
            logger.error(f"Unknown LLM provider: {llm_provider}")
            return _fallback_synthesis(combined_results)
        
        if llm is None:
            return _fallback_synthesis(combined_results)
        
        # Define prompt template for concise, actionable sales analysis
        prompt_template_text = (
            "You are an AI Sales Analyst.\n\n"
            "Summarize the analytics results briefly and clearly. "
            "Write like you're sending insights to a busy manager on WhatsApp.\n\n"
            "Rules:\n"
            "- Keep total length under 120 words.\n"
            "- Use short, direct sentences.\n"
            "- Avoid formal report tone.\n"
            "- Focus on key facts and next steps.\n"
            "- Include emojis only when it improves readability (📊, ⚠️, 💡).\n"
            "- Use the EXACT values from the data provided below - do not say 'Data Missing'.\n"
            "- Structure output in this exact format:\n\n"
            "📊 **Sales Summary**\n"
            "- Total revenue: [use the exact revenue value from Descriptive Data]\n"
            "- Units sold: [use the exact units value from Descriptive Data]\n"
            "- Avg order: [use the exact average order value from Descriptive Data]\n"
            "- Top region: [use the exact top region from Descriptive Data], Top category: [use the exact top category from Descriptive Data]\n\n"
            "⚠️ **Key Issues**\n"
            "[summarize key issues from Diagnostic Data - if no issues, say 'No significant issues detected']\n\n"
            "💡 **Next Steps**\n"
            "- [use the first recommendation from Prescriptive Data]\n"
            "- [use the second recommendation if available]\n"
            "- [use the third recommendation if available]\n\n"
            "IMPORTANT: The data below contains all the values you need. Extract them directly:\n\n"
            "Descriptive Data:\n{desc}\n\n"
            "Diagnostic Data:\n{diag}\n\n"
            "Predictive Data:\n{pred}\n\n"
            "Prescriptive Data:\n{pres}\n\n"
            "Now generate the summary using the EXACT values from the data above. Do not use placeholders or say 'Data Missing'."
        )
        
        # Prepare inputs from combined results with new format
        desc, diag, pred, pres = _format_for_new_template(combined_results)
        
        # Check if we have actual data to work with
        if desc == "No descriptive data available. Please check data source.":
            logger.warning("No descriptive data available for DeepSeek synthesis")
            return _fallback_synthesis(combined_results)
        
        # Run the chain (actual LLM call)
        logger.info("Making LLM call via LangChain...")
        
        if LANGCHAIN_NEW_VERSION:
            # LangChain 1.0+: Use .invoke() directly with prompt and variables
            prompt = PromptTemplate(
                template=prompt_template_text,
                input_variables=["desc", "diag", "pred", "pres"]
            )
            chain = prompt | llm
            narrative = chain.invoke({
                "desc": desc,
                "diag": diag,
                "pred": pred,
                "pres": pres
            })
            # Extract content from message if it's an AIMessage
            if hasattr(narrative, 'content'):
                narrative = narrative.content
            narrative = str(narrative).strip()
        elif LANGCHAIN_OLD_VERSION:
            # LangChain 0.x: Use LLMChain
            prompt = OldPromptTemplate(
                input_variables=["desc", "diag", "pred", "pres"],
                template=prompt_template_text
            )
            chain = LLMChain(prompt=prompt, llm=llm)
            narrative = chain.run(
                desc=desc,
                diag=diag,
                pred=pred,
                pres=pres
            )
            narrative = narrative.strip()
        else:
            logger.error("LangChain not properly initialized")
            return _fallback_synthesis(combined_results)
        
        logger.info("Successfully generated narrative using LangChain LLM")
        return narrative
        
    except Exception as e:
        logger.error(f"LangChain synthesis error: {e}")
        return _fallback_synthesis(combined_results)


def _format_for_new_template(combined_results: Dict[str, Any]) -> tuple:
    """
    Format analytics results for the new concise template.
    
    Returns:
        Tuple of (desc, diag, pred, pres) formatted strings
    """
    descriptive = combined_results.get("descriptive", {})
    diagnostic = combined_results.get("diagnostic", {})
    predictive = combined_results.get("predictive", {})
    prescriptive = combined_results.get("prescriptive", {})
    
    # Format descriptive data
    desc_parts = []
    if descriptive and isinstance(descriptive, dict):
        summary = descriptive.get("summary", {})
        if summary and isinstance(summary, dict):
            total_revenue = summary.get('total_revenue', 0)
            total_units = summary.get('total_quantity', 0)
            avg_order = summary.get('average_order_value', 0)
            
            # Only add if we have actual data
            if total_revenue > 0 or total_units > 0:
                desc_parts.append(f"Total Revenue: ${total_revenue:,.2f}")
                desc_parts.append(f"Total Units: {total_units:,.0f}")
                desc_parts.append(f"Average Order Value: ${avg_order:,.2f}")
        
        # Get top region and category
        by_region = descriptive.get("by_region", {})
        if by_region and isinstance(by_region, dict) and len(by_region) > 0:
            try:
                top_region = max(by_region.items(), key=lambda x: x[1].get("total_revenue", 0) if isinstance(x[1], dict) else 0)
                if top_region and top_region[0]:
                    desc_parts.append(f"Top Region: {top_region[0]}")
            except (ValueError, TypeError):
                pass
        
        top_categories = descriptive.get("top_categories", [])
        if top_categories and isinstance(top_categories, list) and len(top_categories) > 0:
            top_category = top_categories[0].get("category", "N/A") if isinstance(top_categories[0], dict) else "N/A"
            if top_category != "N/A":
                desc_parts.append(f"Top Category: {top_category}")
    
    desc = "\n".join(desc_parts) if desc_parts else "No descriptive data available. Please check data source."
    
    # Format diagnostic data with overall context
    diag_parts = []
    if diagnostic:
        # Get overall summary first
        total_dips = diagnostic.get("summary", {}).get("total_dips_detected", 0)
        total_anomalies = diagnostic.get("summary", {}).get("total_anomalies", 0)
        
        if total_dips > 0 or total_anomalies > 0:
            diag_parts.append(f"Overall: {total_dips} significant dips and {total_anomalies} anomalies detected across all regions.")
        
        # Get insights but prioritize diversity (different regions)
        insights = diagnostic.get("insights", [])
        region_dips = diagnostic.get("region_dips", [])
        
        # Group insights by region to show diversity
        if region_dips:
            # Show top dip from each region (up to 3 regions)
            regions_shown = set()
            for dip in region_dips[:5]:  # Look at top 5
                region = dip.get('region', 'Unknown')
                if region not in regions_shown and len(regions_shown) < 3:
                    regions_shown.add(region)
                    diag_parts.append(f"{region}: {abs(dip.get('drop_percentage', 0)):.1f}% revenue drop on {dip.get('date', 'unknown date')}")
        
        # If we have insights but didn't show region-specific, show top insights
        if insights and len(diag_parts) == 1:  # Only overall summary
            for insight in insights[:2]:  # Top 2 insights
                diag_parts.append(insight)
    
    diag = "\n".join(diag_parts) if diag_parts else "No significant issues detected across all regions."
    
    # Format predictive data
    pred_parts = []
    if predictive:
        overall_forecast = predictive.get("overall_forecast")
        if overall_forecast:
            monthly = overall_forecast.get("monthly_forecast", 0)
            trend = overall_forecast.get("trend", {}).get("direction", "unknown")
            pred_parts.append(f"Next Month Forecast: ${monthly:,.2f}")
            pred_parts.append(f"Trend: {trend}")
    
    pred = "\n".join(pred_parts) if pred_parts else "No forecast data available."
    
    # Format prescriptive data
    pres_parts = []
    if prescriptive:
        recommendations = prescriptive.get("recommendations", [])[:3]
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                title = rec.get('title', 'N/A')
                priority = rec.get('priority', '').upper()
                pres_parts.append(f"{i}. {title} ({priority} priority)")
    
    pres = "\n".join(pres_parts) if pres_parts else "No recommendations available."
    
    return desc, diag, pred, pres


def _format_descriptive(descriptive: Dict[str, Any]) -> str:
    """Format descriptive results for LLM prompt."""
    if not descriptive:
        return "No descriptive data available."
    
    summary = descriptive.get("summary", {})
    top_categories = descriptive.get("top_categories", [])[:3]
    
    result = f"Total Revenue: ${summary.get('total_revenue', 0):,.2f}, "
    result += f"Total Quantity: {summary.get('total_quantity', 0):,.0f}, "
    result += f"Average Order Value: ${summary.get('average_order_value', 0):,.2f}. "
    
    if top_categories:
        result += f"Top categories: {', '.join([c['category'] for c in top_categories])}."
    
    return result


def _format_diagnostic(diagnostic: Dict[str, Any]) -> str:
    """Format diagnostic results for LLM prompt."""
    if not diagnostic:
        return "No diagnostic data available."
    
    insights = diagnostic.get("insights", [])[:5]
    region_dips = diagnostic.get("region_dips", [])[:3]
    
    result = ""
    if insights:
        result += "Key findings: " + "; ".join(insights) + ". "
    if region_dips:
        result += f"Region dips detected: {len(region_dips)} significant drops."
    
    return result if result else "No significant anomalies detected."


def _format_predictive(predictive: Dict[str, Any]) -> str:
    """Format predictive results for LLM prompt."""
    if not predictive:
        return "No forecast data available."
    
    overall_forecast = predictive.get("overall_forecast")
    if overall_forecast:
        monthly = overall_forecast.get("monthly_forecast", 0)
        trend = overall_forecast.get("trend", {}).get("direction", "unknown")
        return f"Next month forecast: ${monthly:,.2f} with {trend} trend."
    
    return "Forecast analysis pending."


def _format_prescriptive(prescriptive: Dict[str, Any]) -> str:
    """Format prescriptive results for LLM prompt."""
    if not prescriptive:
        return "No recommendations available."
    
    recommendations = prescriptive.get("recommendations", [])[:5]
    if recommendations:
        result = "Priority actions: "
        result += "; ".join([
            f"{r.get('title', 'N/A')} ({r.get('priority', 'unknown')})"
            for r in recommendations
        ])
        return result
    
    return "No specific recommendations at this time."


def _fallback_synthesis(combined_results: Dict[str, Any]) -> str:
    """
    Fallback synthesis when LangChain unavailable.
    
    Args:
        combined_results: Combined analytics results
        
    Returns:
        Natural language narrative
    """
    logger.info("Using internal summarizer (LangChain not available)")
    
    narrative_parts = []
    
    # Extract from combined results
    descriptive = combined_results.get("descriptive", {})
    diagnostic = combined_results.get("diagnostic", {})
    predictive = combined_results.get("predictive", {})
    prescriptive = combined_results.get("prescriptive", {})
    
    # Descriptive summary
    if descriptive and "summary" in descriptive:
        summary = descriptive["summary"]
        narrative_parts.append(
            f"Sales Performance Summary: Total revenue reached ${summary.get('total_revenue', 0):,.2f} "
            f"with {summary.get('total_quantity', 0):,.0f} units sold, resulting in an average order value "
            f"of ${summary.get('average_order_value', 0):,.2f}."
        )
    
    # Diagnostic findings
    if diagnostic and "insights" in diagnostic:
        findings = diagnostic["insights"][:3]
        if findings:
            narrative_parts.append("\nKey Findings:")
            for finding in findings:
                narrative_parts.append(f"  • {finding}")
    
    # Predictive forecast
    if predictive and "overall_forecast" in predictive:
        forecast = predictive["overall_forecast"]
        if forecast:
            monthly = forecast.get("monthly_forecast", 0)
            trend = forecast.get("trend", {}).get("direction", "unknown")
            narrative_parts.append(
                f"\nForecast: Next month revenue is projected at ${monthly:,.2f} "
                f"with a {trend} trend."
            )
    
    # Prescriptive recommendations
    if prescriptive and "recommendations" in prescriptive:
        recommendations = prescriptive["recommendations"][:5]
        if recommendations:
            narrative_parts.append("\nPriority Recommendations:")
            for rec in recommendations:
                narrative_parts.append(
                    f"  • [{rec.get('priority', '').upper()}] {rec.get('title', '')}"
                )
    
    if not narrative_parts:
        narrative_parts.append(
            "Sales intelligence analysis completed. Please review the detailed "
            "analytics results for specific insights."
        )
    
    return "\n".join(narrative_parts)


# Legacy function for backward compatibility
def synthesize_with_langchain(
    summary_insights: Dict[str, Any],
    raw_results: Dict[str, Any],
    llm_chain: Optional[Any] = None
) -> str:
    """
    Synthesize narrative using LangChain (legacy interface).
    
    Args:
        summary_insights: Summary insights dictionary
        raw_results: Full analytics results
        llm_chain: Optional pre-initialized LLMChain (not used in new implementation)
        
    Returns:
        Natural language narrative report
    """
    # Convert to new format
    combined_results = {
        "descriptive": raw_results.get("descriptive", {}),
        "diagnostic": raw_results.get("diagnostic", {}),
        "predictive": raw_results.get("predictive", {}),
        "prescriptive": raw_results.get("prescriptive", {})
    }
    
    return synthesize_with_langchain_llmchain(combined_results)
