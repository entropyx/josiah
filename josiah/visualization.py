import plotly.graph_objects as go
import numpy as np


def plot_revenue_decomposition(df, ground_truth, decomp=None):
    """Create an interactive Plotly stacked area chart of revenue components.

    Args:
        df: DataFrame from generator.
        ground_truth: Ground truth dict.
        decomp: Optional decomposition DataFrame. When provided, plot directly
                 from its columns instead of recomputing from ground truth.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    dates = df["date"]

    # If decomp DataFrame is available, use it directly
    if decomp is not None:
        fig.add_trace(go.Scatter(
            x=dates, y=decomp["intercept"], mode="lines", name="Intercept",
            stackgroup="one", fillcolor="rgba(200,200,200,0.5)",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=decomp["trend"], mode="lines", name="Trend",
            stackgroup="one",
        ))
        if decomp["seasonality"].abs().sum() > 0:
            fig.add_trace(go.Scatter(
                x=dates, y=decomp["seasonality"], mode="lines", name="Seasonality",
                stackgroup="one",
            ))
        for col in decomp.columns:
            if col.endswith("_contribution"):
                label = col.replace("_contribution", "")
                # Check if it's a promo
                if label in ground_truth.get("promos", {}):
                    name = f"{label} (promo)"
                else:
                    name = label.title()
                fig.add_trace(go.Scatter(
                    x=dates, y=decomp[col], mode="lines", name=name,
                    stackgroup="one",
                ))

    # Fallback: reconstruct components from ground truth
    elif ground_truth.get("engine") == "pymc":
        from .components.channels import channel_effect
        from .components.controls import generate_controls
        from .components.trend import linear_trend, cube_root_trend
        from .components.seasonality import fourier_seasonality

        n = len(df)
        intercept = ground_truth["intercept"]

        # Intercept
        intercept_arr = np.full(n, intercept)
        fig.add_trace(go.Scatter(
            x=dates, y=intercept_arr, mode="lines", name="Intercept",
            stackgroup="one", fillcolor="rgba(200,200,200,0.5)",
        ))

        # Trend
        if ground_truth["trend_type"] == "linear":
            trend = linear_trend(n, ground_truth["trend_params"].get("slope", 0))
        elif ground_truth["trend_type"] == "cube_root":
            trend = cube_root_trend(n, ground_truth["trend_params"].get("max_val", 100),
                                     ground_truth["trend_params"].get("offset", 1.0))
        else:
            trend = np.zeros(n)
        fig.add_trace(go.Scatter(
            x=dates, y=trend, mode="lines", name="Trend",
            stackgroup="one",
        ))

        # Seasonality
        if ground_truth["seasonality_n_terms"] > 0:
            seas = fourier_seasonality(dates, ground_truth["seasonality_n_terms"],
                                        ground_truth["seasonality_coefficients"])
            fig.add_trace(go.Scatter(
                x=dates, y=seas, mode="lines", name="Seasonality",
                stackgroup="one",
            ))

        # Channels
        for ch_name, ch_params in ground_truth.get("channels", {}).items():
            if f"{ch_name}_spend" in df.columns:
                spend = df[f"{ch_name}_spend"].values
                contrib, _ = channel_effect(spend, ch_params["alpha"], ch_params["l_max"],
                                            ch_params["lam"], ch_params["beta"])
                fig.add_trace(go.Scatter(
                    x=dates, y=contrib, mode="lines", name=ch_name.title(),
                    stackgroup="one",
                ))

        # Controls
        for ctrl_name, ctrl_params in ground_truth.get("controls", {}).items():
            if ctrl_name in df.columns:
                contrib = ctrl_params["coefficient"] * df[ctrl_name].values
                fig.add_trace(go.Scatter(
                    x=dates, y=contrib, mode="lines", name=ctrl_name,
                    stackgroup="one",
                ))

        # Promos
        for promo_name, promo_params in ground_truth.get("promos", {}).items():
            if promo_name in df.columns:
                contrib = promo_params["coefficient"] * df[promo_name].values
                fig.add_trace(go.Scatter(
                    x=dates, y=contrib, mode="lines",
                    name=f"{promo_name} (promo)",
                    stackgroup="one",
                ))

    # Actual y
    fig.add_trace(go.Scatter(
        x=dates, y=df["y"], mode="lines", name="Actual y",
        line=dict(color="black", width=2, dash="dot"),
    ))

    fig.update_layout(
        title="Revenue Decomposition",
        xaxis_title="Date",
        yaxis_title="Revenue",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_channel_spend(df, ground_truth):
    """Create a Plotly chart showing spend per channel.

    Args:
        df: DataFrame from generator.
        ground_truth: Ground truth dict.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    spend_cols = [c for c in df.columns if c.endswith("_spend")]
    for col in spend_cols:
        name = col.replace("_spend", "").title()
        fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=name))

    fig.update_layout(
        title="Channel Spend Over Time",
        xaxis_title="Date",
        yaxis_title="Spend",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig
