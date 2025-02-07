import pytest
from datetime import datetime
from josiah.simulator import (
    generate_baseline,
    add_seasonality,
    add_media_effects,
    add_promotions,
    add_context_variables,
    generate_complete_dataset
)

def test_generate_baseline():
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    baseline_value = 1000
    
    df = generate_baseline(start_date, end_date, baseline_value)
    
    assert len(df) == 366  # 2024 is a leap year
    assert 'date' in df.columns
    assert 'revenue' in df.columns
    assert df['revenue'].mean() >= baseline_value  # Should be >= due to growth

def test_add_seasonality():
    # First generate baseline data
    df = generate_baseline('2024-01-01', '2024-12-31', 1000)
    
    # Add seasonality
    result = add_seasonality(df)
    
    assert 'seasonality_revenue' in result.columns
    assert len(result) == len(df)

# Add more tests for other functions... 