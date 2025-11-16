"""
Diagnostic Analytics Module
Answers: "Why did it happen?" - Detects anomalies, dips, and investigates
causes of performance changes using rolling mean comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

from .config import DEFAULT_DATE_COLUMN, DEFAULT_REVENUE_COLUMN

logger = logging.getLogger(__name__)


def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run diagnostic analytics to detect dips and anomalies.
    
    Args:
        df: DataFrame containing sales data with date column
        
    Returns:
        Dictionary containing:
            - insights: List of textual diagnostic insights
            - region_dips: Dips detected by region
            - channel_dips: Dips detected by sales channel
            - anomalies: Detected anomalies
    """
    logger.info("Running diagnostic analytics...")
    
    # Validate required columns
    if DEFAULT_DATE_COLUMN not in df.columns:
        raise ValueError(f"Date column '{DEFAULT_DATE_COLUMN}' is required for diagnostic analytics")
    if DEFAULT_REVENUE_COLUMN not in df.columns:
        raise ValueError(f"Revenue column '{DEFAULT_REVENUE_COLUMN}' is required")
    
    # Prepare data
    data = df.copy()
    data[DEFAULT_DATE_COLUMN] = pd.to_datetime(data[DEFAULT_DATE_COLUMN], errors='coerce')
    data = data.sort_values(DEFAULT_DATE_COLUMN)
    
    insights = []
    region_dips = []
    channel_dips = []
    anomalies = []
    
    # Detect dips by region
    if "region" in data.columns:
        for region in data["region"].unique():
            region_data = data[data["region"] == region].copy()
            region_data = region_data.set_index(DEFAULT_DATE_COLUMN).resample('D')[DEFAULT_REVENUE_COLUMN].sum()
            
            if len(region_data) < 14:  # Need at least 2 weeks
                continue
            
            # Calculate rolling mean (7-day window)
            rolling_mean = region_data.rolling(window=7, min_periods=3).mean()
            
            # Find periods with 50%+ drop compared to previous period
            for i in range(7, len(region_data)):
                current_period = region_data.iloc[max(0, i-7):i].mean()
                previous_period = region_data.iloc[max(0, i-14):max(0, i-7)].mean() if i >= 14 else current_period
                
                if previous_period > 0:
                    drop_pct = ((current_period - previous_period) / previous_period) * 100
                    
                    if drop_pct <= -50:  # 50% or more drop
                        dip_date = region_data.index[i]
                        region_dips.append({
                            "region": region,
                            "date": str(dip_date),
                            "drop_percentage": round(drop_pct, 1),
                            "current_period_avg": float(current_period),
                            "previous_period_avg": float(previous_period)
                        })
                        
                        # Check which categories contributed to the dip
                        dip_start = dip_date - pd.Timedelta(days=7)
                        dip_data = data[
                            (data[DEFAULT_DATE_COLUMN] >= dip_start) & 
                            (data[DEFAULT_DATE_COLUMN] <= dip_date) &
                            (data["region"] == region)
                        ]
                        
                        if "product_category" in dip_data.columns:
                            category_breakdown = dip_data.groupby("product_category")[DEFAULT_REVENUE_COLUMN].sum()
                            if len(category_breakdown) > 0:
                                top_category = category_breakdown.idxmin() if len(category_breakdown) > 0 else "Unknown"
                                insights.append(
                                    f"{region} region showed {abs(drop_pct):.1f}% drop in revenue around "
                                    f"{dip_date.strftime('%Y-%m-%d')} due to lower {top_category} sales."
                                )
    
    # Detect dips by sales channel
    if "sales_channel" in data.columns:
        for channel in data["sales_channel"].unique():
            channel_data = data[data["sales_channel"] == channel].copy()
            channel_data = channel_data.set_index(DEFAULT_DATE_COLUMN).resample('D')[DEFAULT_REVENUE_COLUMN].sum()
            
            if len(channel_data) < 14:
                continue
            
            for i in range(7, len(channel_data)):
                current_period = channel_data.iloc[max(0, i-7):i].mean()
                previous_period = channel_data.iloc[max(0, i-14):max(0, i-7)].mean() if i >= 14 else current_period
                
                if previous_period > 0:
                    drop_pct = ((current_period - previous_period) / previous_period) * 100
                    
                    if drop_pct <= -50:
                        dip_date = channel_data.index[i]
                        channel_dips.append({
                            "channel": channel,
                            "date": str(dip_date),
                            "drop_percentage": round(drop_pct, 1),
                            "current_period_avg": float(current_period),
                            "previous_period_avg": float(previous_period)
                        })
                        
                        insights.append(
                            f"{channel} channel showed {abs(drop_pct):.1f}% drop in revenue around "
                            f"{dip_date.strftime('%Y-%m-%d')}."
                        )
    
    # Detect anomalies using z-score method
    overall_mean = data[DEFAULT_REVENUE_COLUMN].mean()
    overall_std = data[DEFAULT_REVENUE_COLUMN].std()
    
    if overall_std > 0:
        data['z_score'] = (data[DEFAULT_REVENUE_COLUMN] - overall_mean) / overall_std
        anomaly_threshold = 3.0  # 3 standard deviations
        
        anomalous_records = data[abs(data['z_score']) > anomaly_threshold]
        
        for _, row in anomalous_records.iterrows():
            anomalies.append({
                "date": str(row[DEFAULT_DATE_COLUMN]),
                "revenue": float(row[DEFAULT_REVENUE_COLUMN]),
                "z_score": float(row['z_score']),
                "region": row.get("region", "Unknown"),
                "category": row.get("product_category", "Unknown")
            })
    
    # Generate additional insights
    if not insights:
        insights.append("No significant dips detected (>50% drop) in the analyzed period.")
    
    if len(anomalies) > 0:
        insights.append(f"Detected {len(anomalies)} revenue anomalies using z-score method (threshold: 3.0).")
    
    results = {
        "insights": insights,
        "region_dips": region_dips,
        "channel_dips": channel_dips,
        "anomalies": anomalies[:20],  # Limit to top 20 anomalies
        "summary": {
            "total_dips_detected": len(region_dips) + len(channel_dips),
            "total_anomalies": len(anomalies)
        }
    }
    
    logger.info(f"Diagnostic analytics completed: {len(insights)} insights, {len(region_dips)} region dips, {len(channel_dips)} channel dips")
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.data_loader import DataLoader
    
    # Load sample data
    loader = DataLoader()
    data = loader.load()
    
    # Run diagnostic analytics
    results = run(data)
    
    # Print results
    print("\n" + "="*70)
    print("DIAGNOSTIC ANALYTICS RESULTS")
    print("="*70)
    
    print(f"\nSummary:")
    print(f"  Total Dips Detected: {results['summary']['total_dips_detected']}")
    print(f"  Total Anomalies: {results['summary']['total_anomalies']}")
    
    print(f"\nDiagnostic Insights:")
    for i, insight in enumerate(results['insights'], 1):
        print(f"  {i}. {insight}")
    
    if results['region_dips']:
        print(f"\nRegion Dips ({len(results['region_dips'])}):")
        for dip in results['region_dips'][:5]:
            print(f"  - {dip['region']}: {dip['drop_percentage']:.1f}% drop on {dip['date']}")
    
    if results['channel_dips']:
        print(f"\nChannel Dips ({len(results['channel_dips'])}):")
        for dip in results['channel_dips'][:5]:
            print(f"  - {dip['channel']}: {dip['drop_percentage']:.1f}% drop on {dip['date']}")
    
    if results['anomalies']:
        print(f"\nTop 3 Anomalies:")
        for i, anomaly in enumerate(results['anomalies'][:3], 1):
            print(f"  {i}. Date: {anomaly['date']}, Revenue: ${anomaly['revenue']:,.2f}, "
                  f"Z-score: {anomaly['z_score']:.2f}, Region: {anomaly['region']}")
    
    print("="*70 + "\n")
