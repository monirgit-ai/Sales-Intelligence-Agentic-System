"""
Predictive Analytics Module
Answers: "What will happen?" - Forecasts future sales using regression
and time series models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta
import logging

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Using simpler forecasting methods.")

from .config import DEFAULT_DATE_COLUMN, DEFAULT_REVENUE_COLUMN

logger = logging.getLogger(__name__)


def _simple_linear_forecast(series: pd.Series, periods: int = 30) -> Dict[str, Any]:
    """
    Simple linear regression forecast using numpy.
    
    Args:
        series: Time series data
        periods: Number of periods to forecast ahead
        
    Returns:
        Dictionary with forecast values and dates
    """
    x = np.arange(len(series))
    y = series.values
    
    # Linear regression: y = mx + b
    slope, intercept = np.polyfit(x, y, 1)
    
    # Forecast future values
    future_x = np.arange(len(series), len(series) + periods)
    future_y = slope * future_x + intercept
    
    # Calculate confidence intervals (simplified)
    residuals = y - (slope * x + intercept)
    std_error = np.std(residuals)
    
    return {
        "forecast_values": future_y.tolist(),
        "slope": float(slope),
        "intercept": float(intercept),
        "std_error": float(std_error)
    }


def _sklearn_forecast(series: pd.Series, periods: int = 30) -> Dict[str, Any]:
    """
    Forecast using scikit-learn linear regression.
    
    Args:
        series: Time series data
        periods: Number of periods to forecast ahead
        
    Returns:
        Dictionary with forecast values
    """
    if not SKLEARN_AVAILABLE:
        return _simple_linear_forecast(series, periods)
    
    # Prepare features (time-based)
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast
    future_X = np.arange(len(series), len(series) + periods).reshape(-1, 1)
    future_y = model.predict(future_X)
    
    # Calculate prediction intervals
    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    
    return {
        "forecast_values": future_y.tolist(),
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "std_error": float(std_error)
    }


def run(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run predictive analytics to forecast next month revenue per region.
    
    Args:
        df: DataFrame containing sales data with date and region columns
        
    Returns:
        Dictionary containing:
            - forecasts: Dictionary of forecasts by region
            - overall_forecast: Overall revenue forecast
            - summary: Summary statistics
    """
    logger.info("Running predictive analytics...")
    
    # Validate required columns
    if DEFAULT_DATE_COLUMN not in df.columns:
        raise ValueError(f"Date column '{DEFAULT_DATE_COLUMN}' is required for predictive analytics")
    if DEFAULT_REVENUE_COLUMN not in df.columns:
        raise ValueError(f"Revenue column '{DEFAULT_REVENUE_COLUMN}' is required")
    
    # Prepare data
    data = df.copy()
    data[DEFAULT_DATE_COLUMN] = pd.to_datetime(data[DEFAULT_DATE_COLUMN], errors='coerce')
    data = data.sort_values(DEFAULT_DATE_COLUMN)
    
    forecasts = {}
    
    # Forecast by region
    if "region" in data.columns:
        regions = data["region"].unique()
        
        for region in regions:
            region_data = data[data["region"] == region].copy()
            
            # Aggregate to daily revenue
            region_data = region_data.set_index(DEFAULT_DATE_COLUMN)
            daily_revenue = region_data.resample('D')[DEFAULT_REVENUE_COLUMN].sum()
            daily_revenue = daily_revenue.fillna(0)
            
            if len(daily_revenue) < 7:  # Need minimum data points
                logger.warning(f"Insufficient data for region {region}")
                continue
            
            # Forecast next 30 days (approximately 1 month)
            forecast_periods = 30
            
            # Use sklearn if available, otherwise use simple linear regression
            if SKLEARN_AVAILABLE:
                forecast_result = _sklearn_forecast(daily_revenue, forecast_periods)
            else:
                forecast_result = _simple_linear_forecast(daily_revenue, forecast_periods)
            
            # Generate forecast dates
            last_date = daily_revenue.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_periods,
                freq='D'
            )
            
            # Calculate confidence intervals (95%)
            forecast_values = np.array(forecast_result["forecast_values"])
            std_error = forecast_result["std_error"]
            upper_bound = forecast_values + 1.96 * std_error
            lower_bound = forecast_values - 1.96 * std_error
            
            # Aggregate to monthly forecast (sum of daily forecasts)
            monthly_forecast = float(forecast_values.sum())
            monthly_upper = float(upper_bound.sum())
            monthly_lower = float(lower_bound.sum())
            
            forecasts[region] = {
                "monthly_forecast": monthly_forecast,
                "monthly_upper_bound": monthly_upper,
                "monthly_lower_bound": monthly_lower,
                "daily_forecast": {
                    "dates": [str(d) for d in forecast_dates],
                    "values": [float(v) for v in forecast_values],
                    "upper_bound": [float(v) for v in upper_bound],
                    "lower_bound": [float(v) for v in lower_bound]
                },
                "trend": {
                    "slope": forecast_result["slope"],
                    "direction": "increasing" if forecast_result["slope"] > 0 else "decreasing"
                }
            }
    
    # Overall forecast (aggregate all regions)
    if "region" in data.columns:
        overall_daily = data.set_index(DEFAULT_DATE_COLUMN).resample('D')[DEFAULT_REVENUE_COLUMN].sum()
        overall_daily = overall_daily.fillna(0)
        
        if len(overall_daily) >= 7:
            forecast_periods = 30
            if SKLEARN_AVAILABLE:
                forecast_result = _sklearn_forecast(overall_daily, forecast_periods)
            else:
                forecast_result = _simple_linear_forecast(overall_daily, forecast_periods)
            
            forecast_values = np.array(forecast_result["forecast_values"])
            std_error = forecast_result["std_error"]
            upper_bound = forecast_values + 1.96 * std_error
            lower_bound = forecast_values - 1.96 * std_error
            
            overall_forecast = {
                "monthly_forecast": float(forecast_values.sum()),
                "monthly_upper_bound": float(upper_bound.sum()),
                "monthly_lower_bound": float(lower_bound.sum()),
                "trend": {
                    "slope": forecast_result["slope"],
                    "direction": "increasing" if forecast_result["slope"] > 0 else "decreasing"
                }
            }
        else:
            overall_forecast = None
    else:
        overall_forecast = None
    
    results = {
        "forecasts": forecasts,
        "overall_forecast": overall_forecast,
        "summary": {
            "regions_forecasted": len(forecasts),
            "method": "sklearn_linear_regression" if SKLEARN_AVAILABLE else "numpy_linear_regression",
            "forecast_horizon_days": 30
        }
    }
    
    logger.info(f"Predictive analytics completed: {len(forecasts)} regions forecasted")
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
    
    # Run predictive analytics
    results = run(data)
    
    # Print results
    print("\n" + "="*70)
    print("PREDICTIVE ANALYTICS RESULTS")
    print("="*70)
    
    print(f"\nForecast Method: {results['summary']['method']}")
    print(f"Regions Forecasted: {results['summary']['regions_forecasted']}")
    
    if results['overall_forecast']:
        print(f"\nOverall Monthly Forecast:")
        print(f"  Revenue: ${results['overall_forecast']['monthly_forecast']:,.2f}")
        print(f"  Upper Bound (95%): ${results['overall_forecast']['monthly_upper_bound']:,.2f}")
        print(f"  Lower Bound (95%): ${results['overall_forecast']['monthly_lower_bound']:,.2f}")
        print(f"  Trend: {results['overall_forecast']['trend']['direction']}")
    
    print(f"\nRegional Monthly Forecasts:")
    for region, forecast in results['forecasts'].items():
        print(f"\n  {region}:")
        print(f"    Forecast: ${forecast['monthly_forecast']:,.2f}")
        print(f"    Range: ${forecast['monthly_lower_bound']:,.2f} - ${forecast['monthly_upper_bound']:,.2f}")
        print(f"    Trend: {forecast['trend']['direction']}")
    
    print("="*70 + "\n")
