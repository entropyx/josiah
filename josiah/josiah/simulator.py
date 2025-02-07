import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_spend_pattern(n_days, pattern_type='random', params=None):
    """
    Generate media spend patterns with different characteristics
    
    Args:
        n_days (int): Number of days to generate
        pattern_type (str): Type of pattern ('random', 'pulsed', 'seasonal', 'increasing')
        params (dict, optional): Parameters for the specific pattern type
            - random: {'mean': float, 'std': float}
            - pulsed: {'on_period': int, 'off_period': int, 'spend_level': float}
            - seasonal: {'base': float, 'amplitude': float}
            - increasing: {'start': float, 'end': float}
    
    Returns:
        np.array: Array of spend values
    """
    if params is None:
        params = {}
    
    if pattern_type == 'random':
        mean = params.get('mean', 1000)
        std = params.get('std', 200)
        spend = np.random.normal(mean, std, n_days)
        return np.maximum(spend, 0)
    
    elif pattern_type == 'pulsed':
        on_period = params.get('on_period', 7)
        off_period = params.get('off_period', 21)
        spend_level = params.get('spend_level', 1000)
        
        cycle = np.zeros(on_period + off_period)
        cycle[:on_period] = spend_level
        
        # Repeat the cycle
        n_cycles = (n_days // len(cycle)) + 1
        pattern = np.tile(cycle, n_cycles)
        return pattern[:n_days]
    
    elif pattern_type == 'seasonal':
        base = params.get('base', 1000)
        amplitude = params.get('amplitude', 500)
        
        t = np.arange(n_days)
        seasonal = base + amplitude * np.sin(2 * np.pi * t / 365)
        return np.maximum(seasonal, 0)
    
    elif pattern_type == 'increasing':
        start = params.get('start', 500)
        end = params.get('end', 1500)
        
        return np.linspace(start, end, n_days)
    
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

def add_time_varying_effects(df, channel, base_effectiveness, change_points=None):
    """
    Add time-varying effectiveness for a media channel
    
    Args:
        df (pd.DataFrame): DataFrame with date and channel spend
        channel (str): Name of the channel
        base_effectiveness (float): Base effectiveness coefficient
        change_points (dict): Dictionary of dates and new effectiveness values
    
    Returns:
        pd.DataFrame: DataFrame with updated channel revenue
    """
    result = df.copy()
    spend_col = f'{channel}_spend'
    revenue_col = f'{channel}_revenue'
    
    if change_points is None:
        # If no change points, just apply base effectiveness
        result[revenue_col] = base_effectiveness * result[spend_col]
        return result
    
    # Convert change points to datetime if needed
    change_points = {
        pd.to_datetime(date): value 
        for date, value in change_points.items()
    }
    
    # Sort dates
    dates = sorted(change_points.keys())
    
    # Apply different effectiveness for each period
    current_effect = base_effectiveness
    for i, change_date in enumerate(dates):
        if i == 0:
            mask = result['date'] < change_date
            result.loc[mask, revenue_col] = current_effect * result.loc[mask, spend_col]
            
        mask = result['date'] >= change_date
        if i < len(dates) - 1:
            mask &= result['date'] < dates[i + 1]
            
        current_effect = change_points[change_date]
        result.loc[mask, revenue_col] = current_effect * result.loc[mask, spend_col]
    
    return result

def create_promo_calendar(start_date, end_date, promo_types, frequency='random'):
    """
    Create a promotional calendar with specified characteristics
    
    Args:
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        promo_types (dict): Dictionary of promo types and their parameters
            Example: {'discount_small': {'duration': (3,7), 'frequency': 4}}
        frequency (str): How to schedule promos ('random', 'regular', 'seasonal')
    
    Returns:
        dict: Dictionary of dates and promo types
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    n_days = (end_date - start_date).days + 1
    
    calendar = {}
    
    for promo_type, params in promo_types.items():
        min_duration, max_duration = params.get('duration', (3, 7))
        n_promos = params.get('frequency', 4)
        
        if frequency == 'random':
            # Random dates throughout the year
            possible_dates = pd.date_range(start_date, end_date)
            promo_starts = np.random.choice(possible_dates, size=n_promos, replace=False)
            
        elif frequency == 'regular':
            # Evenly spaced throughout the year
            spacing = n_days // n_promos
            offsets = np.arange(0, n_days, spacing)
            promo_starts = [start_date + pd.Timedelta(days=int(offset)) for offset in offsets]
            
        elif frequency == 'seasonal':
            # Cluster around seasonal peaks (e.g., holidays)
            seasonal_peaks = [
                '2024-02-14',  # Valentine's
                '2024-07-04',  # Independence Day
                '2024-11-25',  # Black Friday
                '2024-12-15',  # Holiday Season
            ]
            peak_dates = pd.to_datetime(seasonal_peaks)
            
            # Add some random offset around peaks
            promo_starts = []
            for peak in peak_dates:
                offset = np.random.randint(-10, 10)
                promo_starts.append(peak + pd.Timedelta(days=offset))
        
        # Add promotions to calendar
        for start in promo_starts:
            duration = np.random.randint(min_duration, max_duration + 1)
            for i in range(duration):
                date = start + pd.Timedelta(days=i)
                if date <= end_date:
                    calendar[date] = promo_type
    
    return calendar

def add_media_effects(df, channels, adstock_rates=None, saturation_rates=None, 
                     spend_patterns=None, time_varying=None, noise=0):
    """
    Add media channel effects to the revenue data
    
    Args:
        df (pd.DataFrame): DataFrame with date and revenue columns
        channels (dict): Dictionary of channel names and their base effectiveness
        adstock_rates (dict, optional): Decay rates for each channel
        saturation_rates (dict, optional): Diminishing returns rates
        spend_patterns (dict, optional): Spend pattern specifications for each channel
        time_varying (dict, optional): Time-varying effectiveness parameters
        noise (float, optional): Random noise level. Defaults to 0.
    
    Returns:
        pd.DataFrame: DataFrame with added media channel columns and effects
    """
    result = df.copy()
    n_days = len(result)
    
    if adstock_rates is None:
        adstock_rates = {channel: 0.7 for channel in channels}
    
    if saturation_rates is None:
        saturation_rates = {channel: 0.5 for channel in channels}
        
    if spend_patterns is None:
        spend_patterns = {
            channel: {'type': 'random'} for channel in channels
        }
    
    for channel, effectiveness in channels.items():
        # Generate spend based on pattern
        pattern_params = spend_patterns[channel]
        base_spend = generate_spend_pattern(
            n_days, 
            pattern_params.get('type', 'random'),
            pattern_params.get('params')
        )
        
        # Apply adstock (decay)
        spend = base_spend.copy()
        decay = adstock_rates[channel]
        for i in range(1, len(spend)):
            spend[i] += spend[i-1] * decay
        
        # Add to DataFrame
        result[f'{channel}_spend'] = base_spend
        
        # Apply time-varying effects if specified
        if time_varying and channel in time_varying:
            result = add_time_varying_effects(
                result, 
                channel, 
                effectiveness,
                time_varying[channel]
            )
        else:
            # Apply saturation (diminishing returns)
            saturation = saturation_rates[channel]
            effect = effectiveness * np.power(spend, saturation)
            
            # Add noise if specified
            if noise > 0:
                effect *= (1 + np.random.normal(0, noise, len(effect)))
            
            result[f'{channel}_revenue'] = effect
    
    return result 

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