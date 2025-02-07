# Josiah

A Python package for simulating and visualizing revenue data with various components like seasonality, promotions, and media effects.

## Overview

Josiah is a powerful tool for generating synthetic revenue data that mimics real-world patterns. It can help you:
- Create baseline revenue trends with configurable growth rates
- Add seasonal patterns and promotional effects
- Simulate media channel impacts
- Visualize revenue decomposition across different components

## Installation

Install using pip:

```bash
pip install josiah
```

## Quick Start

```python
from josiah import generate_baseline, plot_revenue_decomposition

Generate sample revenue data
df = generate_baseline(
start_date='2024-01-01',
end_date='2024-12-31',
baseline_value=1000, # Starting revenue
growth_rate=5, # Daily growth
noise=0.1, # Noise level
preflight_days=30 # Preflight period
)

plot_revenue_decomposition(df)
```


## Features

### Revenue Generation
- Base revenue with configurable starting points
- Linear and non-linear growth patterns
- Optional preflight period for before/after analysis
- Customizable noise levels for realistic variation

### Visualization
- Stacked area plots showing revenue components
- Clear separation of different revenue sources:
  - Baseline revenue
  - Seasonality effects
  - Media channel impacts
  - Promotional lifts
  - Context variables

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Links

- GitHub: [https://github.com/entropyx/josiah](https://github.com/entropyx/josiah)