from .simulator import (
    generate_baseline,
    add_seasonality,
    add_media_effects,
    add_promotions,
    add_context_variables,
    generate_complete_dataset
)
from .visualization import plot_revenue_decomposition

__version__ = '0.1.0' 