

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_price_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    """Create an interactive Plotly chart with moving averages."""
    chart_df = data.copy()
    chart_df["MA 20"] = chart_df["Close"].rolling(window=20).mean()
    chart_df["MA 50"] = chart_df["Close"].rolling(window=50).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#3b82f6", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MA 20"],
            mode="lines",
            name="20-day MA",
            line=dict(color="#10b981", width=1.5, dash="dot"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["MA 50"],
            mode="lines",
            name="50-day MA",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
        )
    )

    fig.update_layout(
        title=f"{ticker.upper()} Historical Price Trend",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig
