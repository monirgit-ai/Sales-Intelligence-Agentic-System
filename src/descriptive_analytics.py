"""
Descriptive Analytics Module
Answers: "What happened?" - Analyzes historical sales data to understand
patterns, trends, and key performance indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

from .config import DEFAULT_DATE_COLUMN, DEFAULT_REVENUE_COLUMN, DEFAULT_QUANTITY_COLUMN

logger = logging.getLogger(__name__)


def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run descriptive analytics on sales data.
    
    Args:
        df: DataFrame containing sales data with columns: date, product_category,
            product_name, region, customer_segment, sales_channel, quantity, revenue
            
    Returns:
        Dictionary containing:
            - summary: Overall statistics (total revenue, quantity, AOV)
            - by_category: Grouped by product_category
            - by_region: Grouped by region
            - by_channel: Grouped by sales_channel
            - by_segment: Grouped by customer_segment
            - top_categories: Top performing categories
    """
    logger.info("Running descriptive analytics...")
    
    # Validate required columns
    required_cols = [DEFAULT_REVENUE_COLUMN, DEFAULT_QUANTITY_COLUMN]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Overall summary statistics
    total_revenue = float(df[DEFAULT_REVENUE_COLUMN].sum())
    total_quantity = float(df[DEFAULT_QUANTITY_COLUMN].sum())
    avg_order_value = total_revenue / total_quantity if total_quantity > 0 else 0
    
    summary = {
        "total_revenue": total_revenue,
        "total_quantity": total_quantity,
        "average_order_value": float(avg_order_value),
        "number_of_records": len(df)
    }
    
    # Add date range if available
    if DEFAULT_DATE_COLUMN in df.columns:
        df_dates = pd.to_datetime(df[DEFAULT_DATE_COLUMN], errors='coerce')
        summary["date_range"] = {
            "start": str(df_dates.min()),
            "end": str(df_dates.max())
        }
    
    # Group by product_category
    by_category = {}
    if "product_category" in df.columns:
        category_stats = df.groupby("product_category").agg({
            DEFAULT_REVENUE_COLUMN: ['sum', 'mean', 'count'],
            DEFAULT_QUANTITY_COLUMN: ['sum', 'mean']
        }).round(2)
        by_category = {
            cat: {
                "total_revenue": float(category_stats.loc[cat, (DEFAULT_REVENUE_COLUMN, 'sum')]),
                "avg_revenue": float(category_stats.loc[cat, (DEFAULT_REVENUE_COLUMN, 'mean')]),
                "transaction_count": int(category_stats.loc[cat, (DEFAULT_REVENUE_COLUMN, 'count')]),
                "total_quantity": float(category_stats.loc[cat, (DEFAULT_QUANTITY_COLUMN, 'sum')]),
                "avg_quantity": float(category_stats.loc[cat, (DEFAULT_QUANTITY_COLUMN, 'mean')])
            }
            for cat in category_stats.index
        }
    
    # Group by region
    by_region = {}
    if "region" in df.columns:
        region_stats = df.groupby("region").agg({
            DEFAULT_REVENUE_COLUMN: ['sum', 'mean', 'count'],
            DEFAULT_QUANTITY_COLUMN: ['sum', 'mean']
        }).round(2)
        by_region = {
            region: {
                "total_revenue": float(region_stats.loc[region, (DEFAULT_REVENUE_COLUMN, 'sum')]),
                "avg_revenue": float(region_stats.loc[region, (DEFAULT_REVENUE_COLUMN, 'mean')]),
                "transaction_count": int(region_stats.loc[region, (DEFAULT_REVENUE_COLUMN, 'count')]),
                "total_quantity": float(region_stats.loc[region, (DEFAULT_QUANTITY_COLUMN, 'sum')]),
                "avg_quantity": float(region_stats.loc[region, (DEFAULT_QUANTITY_COLUMN, 'mean')])
            }
            for region in region_stats.index
        }
    
    # Group by sales_channel
    by_channel = {}
    if "sales_channel" in df.columns:
        channel_stats = df.groupby("sales_channel").agg({
            DEFAULT_REVENUE_COLUMN: ['sum', 'mean', 'count'],
            DEFAULT_QUANTITY_COLUMN: ['sum', 'mean']
        }).round(2)
        by_channel = {
            channel: {
                "total_revenue": float(channel_stats.loc[channel, (DEFAULT_REVENUE_COLUMN, 'sum')]),
                "avg_revenue": float(channel_stats.loc[channel, (DEFAULT_REVENUE_COLUMN, 'mean')]),
                "transaction_count": int(channel_stats.loc[channel, (DEFAULT_REVENUE_COLUMN, 'count')]),
                "total_quantity": float(channel_stats.loc[channel, (DEFAULT_QUANTITY_COLUMN, 'sum')]),
                "avg_quantity": float(channel_stats.loc[channel, (DEFAULT_QUANTITY_COLUMN, 'mean')])
            }
            for channel in channel_stats.index
        }
    
    # Group by customer_segment
    by_segment = {}
    if "customer_segment" in df.columns:
        segment_stats = df.groupby("customer_segment").agg({
            DEFAULT_REVENUE_COLUMN: ['sum', 'mean', 'count'],
            DEFAULT_QUANTITY_COLUMN: ['sum', 'mean']
        }).round(2)
        by_segment = {
            segment: {
                "total_revenue": float(segment_stats.loc[segment, (DEFAULT_REVENUE_COLUMN, 'sum')]),
                "avg_revenue": float(segment_stats.loc[segment, (DEFAULT_REVENUE_COLUMN, 'mean')]),
                "transaction_count": int(segment_stats.loc[segment, (DEFAULT_REVENUE_COLUMN, 'count')]),
                "total_quantity": float(segment_stats.loc[segment, (DEFAULT_QUANTITY_COLUMN, 'sum')]),
                "avg_quantity": float(segment_stats.loc[segment, (DEFAULT_QUANTITY_COLUMN, 'mean')])
            }
            for segment in segment_stats.index
        }
    
    # Top performing categories
    top_categories = []
    if "product_category" in df.columns:
        category_revenue = df.groupby("product_category")[DEFAULT_REVENUE_COLUMN].sum().sort_values(ascending=False)
        top_categories = [
            {
                "category": cat,
                "total_revenue": float(revenue),
                "percentage": float((revenue / total_revenue * 100) if total_revenue > 0 else 0)
            }
            for cat, revenue in category_revenue.head(10).items()
        ]
    
    results = {
        "summary": summary,
        "by_category": by_category,
        "by_region": by_region,
        "by_channel": by_channel,
        "by_segment": by_segment,
        "top_categories": top_categories
    }
    
    logger.info("Descriptive analytics completed successfully")
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
    
    # Run descriptive analytics
    results = run(data)
    
    # Print summary
    print("\n" + "="*70)
    print("DESCRIPTIVE ANALYTICS RESULTS")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Total Revenue: ${results['summary']['total_revenue']:,.2f}")
    print(f"  Total Quantity: {results['summary']['total_quantity']:,.0f}")
    print(f"  Average Order Value: ${results['summary']['average_order_value']:,.2f}")
    print(f"  Number of Records: {results['summary']['number_of_records']}")
    
    print(f"\nTop 3 Categories by Revenue:")
    for i, cat in enumerate(results['top_categories'][:3], 1):
        print(f"  {i}. {cat['category']}: ${cat['total_revenue']:,.2f} ({cat['percentage']:.1f}%)")
    
    print(f"\nBy Region (Top 3):")
    region_revenue = sorted(
        [(k, v['total_revenue']) for k, v in results['by_region'].items()],
        key=lambda x: x[1],
        reverse=True
    )[:3]
    for i, (region, revenue) in enumerate(region_revenue, 1):
        print(f"  {i}. {region}: ${revenue:,.2f}")
    
    print("="*70 + "\n")
