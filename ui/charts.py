"""
ui/charts.py — Plotly chart builders for UrbanPulse AI dashboard.

Every function returns a ``plotly.graph_objects.Figure`` so it can be
handed directly to ``st.plotly_chart()``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import (
    CHART_HEIGHT,
    CONGESTED_ROAD_COLOR,
    MAP_HEIGHT,
    NODE_COLOR,
    NORMAL_ROAD_COLOR,
)
from network.road_network import CityRoadNetwork


# ---------------------------------------------------------------------------
# Shared layout defaults for all charts
# ---------------------------------------------------------------------------

_CHART_LAYOUT = dict(
    plot_bgcolor="#0b0e14",
    paper_bgcolor="#111827",
    font=dict(color="#e2e8f0", family="Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"),
    title_font=dict(color="#93c5fd", size=15, family="Inter, Segoe UI, sans-serif"),
    legend=dict(
        bgcolor="#1e2740",
        bordercolor="#2d3a5a",
        borderwidth=1,
        font=dict(color="#cbd5e1"),
    ),
    margin=dict(l=10, r=10, t=50, b=10),
)

_AXIS_STYLE = dict(
    gridcolor="#1f2937",
    zerolinecolor="#1f2937",
    tickfont=dict(color="#94a3b8", size=11),
    title_font=dict(color="#90caf9", size=12),
)


# ---------------------------------------------------------------------------
# Traffic flow timeline
# ---------------------------------------------------------------------------

def traffic_flow_timeline(df: pd.DataFrame) -> go.Figure:
    """Line chart of total vehicle count over time.

    Parameters
    ----------
    df:
        Traffic DataFrame with ``timestamp`` and ``vehicle_count`` columns.
    """
    if df.empty:
        return _empty_figure("No data available")

    hourly = (
        df.groupby("hour")["vehicle_count"]
        .sum()
        .reset_index()
        .rename(columns={"vehicle_count": "total_vehicles"})
    )

    fig = px.line(
        hourly,
        x="hour",
        y="total_vehicles",
        markers=True,
        title="🚗 Traffic Flow Timeline",
        labels={"hour": "Hour of Day", "total_vehicles": "Total Vehicles"},
        color_discrete_sequence=["#2196F3"],
    )
    fig.update_layout(
        height=CHART_HEIGHT,
        **_CHART_LAYOUT,
        xaxis=dict(tickmode="linear", dtick=1, **_AXIS_STYLE),
        yaxis=dict(**_AXIS_STYLE),
    )
    return fig


# ---------------------------------------------------------------------------
# Predicted vs actual
# ---------------------------------------------------------------------------

def predicted_vs_actual(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
) -> go.Figure:
    """Scatter plot comparing actual and predicted traffic flow."""
    fig = go.Figure()

    # Diagonal reference line
    vmin = min(y_actual.min(), y_predicted.min())
    vmax = max(y_actual.max(), y_predicted.max())
    fig.add_trace(
        go.Scatter(
            x=[vmin, vmax],
            y=[vmin, vmax],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="#ffffff", dash="dash", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_actual,
            y=y_predicted,
            mode="markers",
            name="Prediction",
            marker=dict(
                color="#2196F3",
                size=4,
                opacity=0.6,
            ),
        )
    )
    fig.update_layout(
        title="📊 Predicted vs Actual Traffic Flow",
        xaxis_title="Actual Vehicles",
        yaxis_title="Predicted Vehicles",
        height=CHART_HEIGHT,
        **_CHART_LAYOUT,
        xaxis=dict(**_AXIS_STYLE),
        yaxis=dict(**_AXIS_STYLE),
    )
    return fig


# ---------------------------------------------------------------------------
# Congestion heatmap
# ---------------------------------------------------------------------------

def congestion_heatmap(pivot: pd.DataFrame) -> go.Figure:
    """Heatmap: roads (y-axis) × hours (x-axis) → congestion ratio."""
    if pivot.empty:
        return _empty_figure("No congestion data")

    # Limit to top 20 most congested roads to keep the chart readable
    top_roads = pivot.mean(axis=1).nlargest(20).index
    pivot_top = pivot.loc[pivot.index.isin(top_roads)]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_top.values,
            x=[str(h) for h in pivot_top.columns],
            y=pivot_top.index.tolist(),
            colorscale="RdYlGn_r",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Congestion"),
        )
    )
    fig.update_layout(
        title="🌡️ Congestion Heatmap (Road × Hour)",
        xaxis_title="Hour of Day",
        yaxis_title="Road ID",
        height=max(CHART_HEIGHT, len(pivot_top) * 20 + 100),
        **_CHART_LAYOUT,
        xaxis=dict(**_AXIS_STYLE),
        yaxis=dict(**_AXIS_STYLE),
    )
    return fig


# ---------------------------------------------------------------------------
# Feature importance bar chart
# ---------------------------------------------------------------------------

def feature_importance_chart(importances: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of RF feature importances."""
    items = sorted(importances.items(), key=lambda x: x[1])
    features, values = zip(*items) if items else ([], [])

    fig = go.Figure(
        go.Bar(
            x=list(values),
            y=list(features),
            orientation="h",
            marker_color="#2196F3",
        )
    )
    fig.update_layout(
        title="🔍 Feature Importance",
        xaxis_title="Importance",
        height=CHART_HEIGHT,
        **_CHART_LAYOUT,
        xaxis=dict(**_AXIS_STYLE),
        yaxis=dict(**_AXIS_STYLE),
    )
    return fig


# ---------------------------------------------------------------------------
# Network graph visualisation
# ---------------------------------------------------------------------------

def road_network_graph(network: CityRoadNetwork) -> go.Figure:
    """Draw the city road graph with edges coloured by congestion."""
    G = network.graph
    pos = {n: data["pos"] for n, data in G.nodes(data=True)}

    # ---- edges ----
    edge_traces: list[go.Scatter] = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos.get(u, (0, 0))
        x1, y1 = pos.get(v, (0, 0))
        seg = data["segment"]
        color = CONGESTED_ROAD_COLOR if seg.is_congested else NORMAL_ROAD_COLOR
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=1.5, color=color),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # ---- nodes ----
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = [G.nodes[n].get("name", str(n)) for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(n) for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=8, color=NODE_COLOR, line=dict(width=1, color="#ffffff")),
        showlegend=False,
    )

    # ---- legend proxies ----
    legend_normal = go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=NORMAL_ROAD_COLOR, width=3),
        name="Normal Road",
    )
    legend_congested = go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=CONGESTED_ROAD_COLOR, width=3),
        name="Congested Road",
    )

    fig = go.Figure(data=[*edge_traces, node_trace, legend_normal, legend_congested])
    fig.update_layout(
        title="🗺️ City Road Network",
        height=MAP_HEIGHT,
        **_CHART_LAYOUT,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Route visualisation
# ---------------------------------------------------------------------------

def route_visualisation(
    network: CityRoadNetwork,
    path: list[int],
    algorithm: str = "",
) -> go.Figure:
    """Highlight a specific route in the road network graph."""
    G = network.graph
    pos = {n: data["pos"] for n, data in G.nodes(data=True)}
    path_edges = set(zip(path[:-1], path[1:]))

    edge_traces: list[go.Scatter] = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos.get(u, (0, 0))
        x1, y1 = pos.get(v, (0, 0))
        seg = data["segment"]
        if (u, v) in path_edges:
            color = "#FF9800"    # orange = selected route
            width = 4.0
        elif seg.is_congested:
            color = CONGESTED_ROAD_COLOR
            width = 1.5
        else:
            color = NORMAL_ROAD_COLOR
            width = 1.0
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="none",
                showlegend=False,
            )
        )

    node_colors = [
        "#FF5722" if n in (path[0], path[-1]) else
        "#FF9800" if n in path else
        NODE_COLOR
        for n in G.nodes()
    ]
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers",
        hovertext=[G.nodes[n].get("name", str(n)) for n in G.nodes()],
        hoverinfo="text",
        marker=dict(size=10, color=node_colors, line=dict(width=1, color="#ffffff")),
        showlegend=False,
    )

    title = f"🧭 Optimal Route — {algorithm}" if algorithm else "🧭 Optimal Route"
    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        title=title,
        height=MAP_HEIGHT,
        **_CHART_LAYOUT,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Weather impact bar chart
# ---------------------------------------------------------------------------

def weather_impact_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing mean vehicles per weather condition."""
    if df.empty:
        return _empty_figure("No weather data")

    weather_stats = (
        df.groupby("weather_condition")["vehicle_count"].mean().reset_index()
    )
    fig = px.bar(
        weather_stats,
        x="weather_condition",
        y="vehicle_count",
        title="🌦️ Weather Impact on Traffic",
        color="vehicle_count",
        color_continuous_scale="Blues",
        labels={"vehicle_count": "Avg Vehicles", "weather_condition": "Weather"},
    )
    fig.update_layout(
        height=CHART_HEIGHT,
        **_CHART_LAYOUT,
        coloraxis_showscale=False,
        xaxis=dict(**_AXIS_STYLE),
        yaxis=dict(**_AXIS_STYLE),
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#64748b"),
    )
    fig.update_layout(
        height=CHART_HEIGHT,
        **_CHART_LAYOUT,
    )
    return fig
