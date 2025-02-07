import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_baseline(start_date, end_date, baseline_value, growth_rate=0, slope_changes=None, noise=0, preflight_days=0):
    """
    Generate a time series with revenue values including an optional preflight period
    
    Args:
        start_date (str or datetime): Start date of the time series
        end_date (str or datetime): End date of the time series  
        baseline_value (float): Initial value to use for revenue
        growth_rate (float, optional): Daily linear growth amount. Defaults to 0.
        slope_changes (dict, optional): Dictionary of dates and slope changes.
        noise (float, optional): Noise level between 0 and 1. Defaults to 0.
        preflight_days (int, optional): Number of days to add before start_date. Defaults to 0.
    
    Returns:
        pd.DataFrame: DataFrame containing dates and revenue values
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    # Add preflight period if specified
    if preflight_days > 0:
        start_date = start_date - timedelta(days=preflight_days)
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize revenue values with baseline
    revenue = np.ones(len(dates)) * baseline_value
    
    # Apply growth rate
    days = np.arange(len(dates))
    revenue += days * growth_rate
    
    # Apply slope changes if provided
    if slope_changes:
        for change_date, new_slope in slope_changes.items():
            if isinstance(change_date, str):
                change_date = pd.to_datetime(change_date)
            mask = dates >= change_date
            days_since_change = (dates[mask] - change_date).days
            revenue[mask] += days_since_change * (new_slope - growth_rate)
    
    # Add noise if specified
    if noise > 0:
        noise_factor = noise * baseline_value
        revenue += np.random.normal(0, noise_factor, size=len(revenue))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'revenue': revenue
    })
    
    return df 