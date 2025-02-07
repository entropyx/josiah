import matplotlib.pyplot as plt
import numpy as np

def plot_revenue_decomposition(results_df):
    """
    Plot revenue decomposition with specific stacking order:
    1. Seasonality (bottom)
    2. Baseline
    3. Media channels
    4. Context variables
    5. Promotions (top)
    """
    # Create copy to avoid modifying original
    plot_data = results_df.copy()
    
    # Identify different types of revenue columns
    seasonality_cols = ['seasonality_revenue'] if 'seasonality_revenue' in plot_data.columns else []
    promo_cols = [col for col in plot_data.columns if col.endswith('_revenue') and 'promo' in col]
    media_cols = [col for col in plot_data.columns if col.endswith('_revenue') 
                 and col not in promo_cols 
                 and col not in seasonality_cols
                 and col != 'total_revenue'
                 and col != 'revenue'
                 and 'context' not in col]
    context_cols = [col for col in plot_data.columns if col.endswith('_revenue') and 'context' in col]
    
    # Order columns according to desired stacking
    ordered_cols = seasonality_cols + ['revenue'] + media_cols + context_cols + promo_cols
    
    # Create the stacked area plot
    plt.figure(figsize=(15, 8))
    
    # Convert data to numpy arrays for stacking
    x = plot_data['date']
    y_values = [plot_data[col].values for col in ordered_cols]
    
    # Stack the arrays
    y_stack = np.column_stack(y_values)
    y_cumsum = np.cumsum(y_stack, axis=1)
    
    # Plot areas from bottom to top
    plt.fill_between(x, 0, y_cumsum[:,0], 
                    label='Seasonality' if seasonality_cols else 'Baseline', 
                    alpha=0.7)
    
    for i in range(len(ordered_cols)-1):
        label = ordered_cols[i+1].replace('_revenue', '').title()
        plt.fill_between(x, y_cumsum[:,i], y_cumsum[:,i+1],
                        label=label,
                        alpha=0.7)
    
    plt.title('Revenue Decomposition Over Time')
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    
    # Improve legend placement and formatting
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    # Improve date formatting on x-axis
    plt.gcf().autofmt_xdate()
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add some padding to the layout
    plt.tight_layout()
    plt.show() 