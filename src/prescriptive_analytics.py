"""
Prescriptive Analytics Module
Answers: "What should we do?" - Generates actionable recommendations
based on diagnostic and predictive analytics results.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
import logging

from .config import DEFAULT_REVENUE_COLUMN

logger = logging.getLogger(__name__)


def _generate_llm_suggestion(context: str) -> str:
    """
    Placeholder for LLM-based suggestion generation.
    Later connected to LangChain for advanced recommendations.
    
    Args:
        context: Context string describing the situation
        
    Returns:
        Generated suggestion text
    """
    # Placeholder - will be replaced with LangChain integration
    return f"[LLM Placeholder] Based on: {context[:100]}..."


def run(
    df: pd.DataFrame,
    diagnostic_results: Optional[Dict[str, Any]] = None,
    predictive_results: Optional[Dict[str, Any]] = None,
    descriptive_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run prescriptive analytics to generate actionable recommendations.
    
    Args:
        df: DataFrame containing sales data
        diagnostic_results: Results from diagnostic analytics
        predictive_results: Results from predictive analytics
        descriptive_results: Results from descriptive analytics
        
    Returns:
        Dictionary containing:
            - recommendations: List of actionable recommendations
            - priority_actions: High priority actions
            - summary: Summary of recommendations
    """
    logger.info("Running prescriptive analytics...")
    
    recommendations = []
    priority_actions = []
    
    # Process diagnostic insights (dips and anomalies)
    if diagnostic_results:
        insights = diagnostic_results.get("insights", [])
        region_dips = diagnostic_results.get("region_dips", [])
        channel_dips = diagnostic_results.get("channel_dips", [])
        
        # Aggregate dips by region to avoid duplicate recommendations
        from collections import defaultdict
        region_dips_aggregated = defaultdict(list)
        for dip in region_dips:
            region_dips_aggregated[dip["region"]].append(dip)
        
        # Generate recommendations for each region (one per region with aggregated stats)
        for region, dips_list in region_dips_aggregated.items():
            # Calculate aggregate statistics
            num_dips = len(dips_list)
            avg_drop = sum(abs(d["drop_percentage"]) for d in dips_list) / num_dips
            max_drop = max(abs(d["drop_percentage"]) for d in dips_list)
            min_drop = min(abs(d["drop_percentage"]) for d in dips_list)
            
            # Identify affected categories
            affected_categories = set()
            for dip in dips_list:
                # Extract category from insights if available
                for insight in insights:
                    if region in insight and "due to lower" in insight:
                        category_part = insight.split("due to lower")[-1].split(" sales")[0].strip()
                        if category_part:
                            affected_categories.add(category_part)
            
            # Build comprehensive description
            if num_dips == 1:
                description = f"{region} region experienced a significant {max_drop:.1f}% revenue drop."
            else:
                description = f"{region} region experienced {num_dips} significant revenue drops, ranging from {min_drop:.1f}% to {max_drop:.1f}% (average {avg_drop:.1f}% drop)."
            
            if affected_categories:
                categories_str = ", ".join(list(affected_categories)[:3])  # Show up to 3 categories
                description += f" Primary impact on {categories_str} sales."
            
            # Build action with reasoning
            action = f"Launch targeted marketing campaigns and promotional offers in {region} to recover sales."
            if affected_categories:
                categories_str = ", ".join(list(affected_categories)[:2])
                action = f"Focus promotional efforts in {region} on {categories_str} categories to address {num_dips} revenue dips detected."
            
            # Determine urgency based on number and severity of dips
            urgency = "high" if num_dips >= 5 or max_drop >= 70 else "medium"
            priority = "high" if num_dips >= 3 or max_drop >= 65 else "medium"
            
            recommendation = {
                "priority": priority,
                "category": "Region Performance",
                "title": f"Increase promotions in {region}",
                "description": description,
                "reasoning": f"Based on {num_dips} revenue drops detected (avg {avg_drop:.1f}% drop), immediate action needed to stabilize {region} sales performance.",
                "action": action,
                "expected_impact": "medium" if num_dips < 5 else "high",
                "urgency": urgency,
                "stats": {
                    "num_dips": num_dips,
                    "avg_drop_pct": round(avg_drop, 1),
                    "max_drop_pct": round(max_drop, 1),
                    "affected_categories": list(affected_categories)[:5]
                }
            }
            recommendations.append(recommendation)
            if priority == "high":
                priority_actions.append(recommendation)
        
        # Generate recommendations for channel dips
        for dip in channel_dips:
            channel = dip["channel"]
            drop_pct = abs(dip["drop_percentage"])
            
            recommendation = {
                "priority": "high",
                "category": "Channel Optimization",
                "title": f"Investigate and optimize {channel} channel",
                "description": f"{channel} channel showed {drop_pct:.1f}% revenue drop around {dip['date']}.",
                "action": f"Analyze root causes for {channel} channel decline and implement corrective measures.",
                "expected_impact": "high",
                "urgency": "high"
            }
            recommendations.append(recommendation)
            priority_actions.append(recommendation)
    
    # Process predictive insights
    if predictive_results:
        forecasts = predictive_results.get("forecasts", {})
        overall_forecast = predictive_results.get("overall_forecast")
        
        # Identify regions with declining trends
        declining_regions = []
        for region, forecast_data in forecasts.items():
            trend = forecast_data.get("trend", {})
            if trend.get("direction") == "decreasing":
                declining_regions.append({
                    "region": region,
                    "forecast": forecast_data.get("monthly_forecast", 0),
                    "trend": trend
                })
        
        for region_info in declining_regions:
            recommendation = {
                "priority": "medium",
                "category": "Regional Strategy",
                "title": f"Boost sales in {region_info['region']}",
                "description": f"Forecast shows declining trend in {region_info['region']} region.",
                "action": f"Implement growth initiatives and special promotions in {region_info['region']}.",
                "expected_impact": "medium",
                "urgency": "medium"
            }
            recommendations.append(recommendation)
    
    # Analyze channel performance
    if "sales_channel" in df.columns and descriptive_results:
        by_channel = descriptive_results.get("by_channel", {})
        
        if by_channel:
            # Find best and worst performing channels
            channel_revenue = {
                channel: data["total_revenue"]
                for channel, data in by_channel.items()
            }
            
            best_channel = max(channel_revenue, key=channel_revenue.get)
            worst_channel = min(channel_revenue, key=channel_revenue.get)
            
            # Recommend shifting resources to best channel
            recommendation = {
                "priority": "medium",
                "category": "Resource Allocation",
                "title": f"Shift more resources to {best_channel} channel",
                "description": f"{best_channel} is the top-performing channel with ${channel_revenue[best_channel]:,.2f} in revenue.",
                "action": f"Reallocate marketing budget and sales resources to {best_channel} channel for better ROI.",
                "expected_impact": "high",
                "urgency": "low"
            }
            recommendations.append(recommendation)
    
    # Analyze product category performance
    if "product_category" in df.columns and descriptive_results:
        top_categories = descriptive_results.get("top_categories", [])
        
        if top_categories:
            top_category = top_categories[0]["category"]
            recommendation = {
                "priority": "low",
                "category": "Product Strategy",
                "title": f"Maintain focus on {top_category}",
                "description": f"{top_category} is the top revenue-generating category.",
                "action": f"Ensure adequate inventory and marketing support for {top_category} products.",
                "expected_impact": "medium",
                "urgency": "low"
            }
            recommendations.append(recommendation)
    
    # Generate LLM-enhanced recommendations (placeholder)
    if diagnostic_results and len(priority_actions) > 0:
        # Example of LLM placeholder usage
        context = f"Detected {len(diagnostic_results.get('region_dips', []))} region dips and {len(diagnostic_results.get('channel_dips', []))} channel dips."
        llm_suggestion = _generate_llm_suggestion(context)
        
        recommendations.append({
            "priority": "medium",
            "category": "Strategic Planning",
            "title": "Comprehensive Recovery Strategy",
            "description": llm_suggestion,
            "action": "Review and implement comprehensive recovery strategy based on diagnostic insights.",
            "expected_impact": "high",
            "urgency": "medium",
            "source": "llm_enhanced"
        })
    
    # Remove duplicates by title + category
    unique = {(r["title"], r["category"]): r for r in recommendations}
    recommendations = list(unique.values())
    
    # Update priority_actions to remove duplicates as well
    priority_actions = [
        r for r in priority_actions 
        if (r["title"], r["category"]) in unique
    ]
    
    # Categorize recommendations
    high_priority = [r for r in recommendations if r.get("priority") == "high"]
    medium_priority = [r for r in recommendations if r.get("priority") == "medium"]
    low_priority = [r for r in recommendations if r.get("priority") == "low"]
    
    results = {
        "recommendations": recommendations,
        "priority_actions": priority_actions,
        "categorized": {
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority
        },
        "summary": {
            "total_recommendations": len(recommendations),
            "high_priority_count": len(high_priority),
            "medium_priority_count": len(medium_priority),
            "low_priority_count": len(low_priority)
        }
    }
    
    logger.info(f"Prescriptive analytics completed: {len(recommendations)} recommendations generated")
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.data_loader import DataLoader
    from src.descriptive_analytics import run as run_descriptive
    from src.diagnostic_analytics import run as run_diagnostic
    from src.predictive_analytics import run as run_predictive
    
    # Load sample data
    loader = DataLoader()
    data = loader.load()
    
    # Run all analytics
    print("Running analytics modules...")
    descriptive_results = run_descriptive(data)
    diagnostic_results = run_diagnostic(data)
    predictive_results = run_predictive(data)
    
    # Run prescriptive analytics
    results = run(data, diagnostic_results, predictive_results, descriptive_results)
    
    # Print results
    print("\n" + "="*70)
    print("PRESCRIPTIVE ANALYTICS RESULTS")
    print("="*70)
    
    print(f"\nSummary:")
    print(f"  Total Recommendations: {results['summary']['total_recommendations']}")
    print(f"  High Priority: {results['summary']['high_priority_count']}")
    print(f"  Medium Priority: {results['summary']['medium_priority_count']}")
    print(f"  Low Priority: {results['summary']['low_priority_count']}")
    
    print(f"\nHigh Priority Actions:")
    for i, action in enumerate(results['categorized']['high_priority'], 1):
        print(f"\n  {i}. {action['title']}")
        print(f"     Category: {action['category']}")
        print(f"     Description: {action['description']}")
        print(f"     Action: {action['action']}")
        print(f"     Expected Impact: {action['expected_impact']}")
    
    print(f"\nAll Recommendations ({len(results['recommendations'])}):")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. [{rec['priority'].upper()}] {rec['title']}")
    
    print("="*70 + "\n")
