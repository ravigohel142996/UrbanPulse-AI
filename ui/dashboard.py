"""
ui/dashboard.py — Main dashboard layout for UrbanPulse AI.

Orchestrates all UI sections using the chart builders and controls.
Keeps all ``st.*`` calls here; the analytics / ML layers have no
Streamlit dependency.
"""

from __future__ import annotations

import streamlit as st

from analytics.traffic_metrics import (
    build_congestion_heatmap_data,
    compute_kpis,
    hourly_traffic_volume,
    peak_hour_analysis,
    top_congested_roads,
    weather_impact_summary,
)
from config import PAGE_ICON, PAGE_TITLE
from models.congestion_detector import CongestionDetector
from models.traffic_predictor import TrafficPredictor
from network.road_network import CityRoadNetwork
from network.route_optimizer import RouteOptimizer
from ui.charts import (
    congestion_heatmap,
    feature_importance_chart,
    predicted_vs_actual,
    road_network_graph,
    route_visualisation,
    traffic_flow_timeline,
    weather_impact_chart,
)
from utils.helpers import format_number


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

def configure_page() -> None:
    """Set Streamlit page-level configuration."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_header() -> None:
    """Render the top-level page header."""
    st.markdown(
        f"""
        <div class="up-header">
            <span class="up-header-icon">{PAGE_ICON}</span>
            <div>
                <h1 class="up-header-title">UrbanPulse AI</h1>
                <p class="up-header-subtitle">
                    Smart City Traffic Intelligence System &nbsp;·&nbsp;
                    Real-time simulation &nbsp;·&nbsp; ML prediction &nbsp;·&nbsp; Route optimisation
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def render_kpi_overview(kpis, prev_kpis=None) -> None:
    """Render the top KPI tiles."""
    st.subheader("📈 Traffic Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "🚗 Total Vehicles",
            format_number(kpis.total_vehicles),
        )
    with c2:
        st.metric(
            "🚦 Congested Roads",
            f"{kpis.congested_roads} / {kpis.total_roads}",
        )
    with c3:
        st.metric(
            "⚡ Avg Speed",
            f"{kpis.average_speed_kmh} km/h",
        )
    with c4:
        st.metric(
            "📊 Flow Index",
            f"{kpis.traffic_flow_index:.1f} / 100",
        )
    with c5:
        st.metric(
            "⚠️ Anomaly Rate",
            f"{kpis.anomaly_rate_pct:.1f}%",
        )
    st.divider()


def render_prediction_analytics(
    df,
    predictor: TrafficPredictor,
) -> None:
    """Render the Traffic Prediction Analytics section."""
    st.subheader("🤖 Traffic Prediction Analytics")

    if predictor.evaluation:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{predictor.evaluation.mae:.1f}")
        col2.metric("RMSE", f"{predictor.evaluation.rmse:.1f}")
        col3.metric("R² Score", f"{predictor.evaluation.r2:.4f}")
        col4.metric("CV-R²", f"{predictor.evaluation.cv_r2_mean:.4f}")

    tab1, tab2, tab3 = st.tabs(
        ["📉 Predicted vs Actual", "🌡️ Congestion Heatmap", "📊 Feature Importance"]
    )

    with tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(
                traffic_flow_timeline(df),
                use_container_width=True,
            )
        with col_r:
            if (
                predictor.test_actuals is not None
                and predictor.test_predictions is not None
            ):
                st.plotly_chart(
                    predicted_vs_actual(
                        predictor.test_actuals,
                        predictor.test_predictions,
                    ),
                    use_container_width=True,
                )

    with tab2:
        pivot = build_congestion_heatmap_data(df)
        st.plotly_chart(
            congestion_heatmap(pivot),
            use_container_width=True,
        )

    with tab3:
        if predictor.evaluation and predictor.evaluation.feature_importances:
            col_fi, col_wi = st.columns(2)
            with col_fi:
                st.plotly_chart(
                    feature_importance_chart(predictor.evaluation.feature_importances),
                    use_container_width=True,
                )
            with col_wi:
                st.plotly_chart(
                    weather_impact_chart(df),
                    use_container_width=True,
                )

    st.divider()


def render_network_visualisation(network: CityRoadNetwork) -> None:
    """Render the road network graph."""
    st.subheader("🗺️ Road Network Visualisation")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.plotly_chart(road_network_graph(network), use_container_width=True)

    with col_right:
        stats = network.congestion_summary()
        st.markdown("**Network Statistics**")
        st.json(stats, expanded=True)

        edge_df = network.edge_list_df()
        congested_df = edge_df[edge_df["is_congested"]]
        if not congested_df.empty:
            st.markdown(f"**🔴 Congested Roads ({len(congested_df)})**")
            st.dataframe(
                congested_df[["road_id", "road_type", "congestion_ratio"]],
                use_container_width=True,
                hide_index=True,
            )

    st.divider()


def render_route_optimisation(network: CityRoadNetwork) -> None:
    """Render the route optimisation section."""
    st.subheader("🧭 Route Optimisation")
    optimizer = RouteOptimizer(network)
    nodes = sorted(network.graph.nodes())

    col1, col2, col3 = st.columns(3)
    with col1:
        source = st.selectbox("🟢 Origin", nodes, index=0, key="route_src")
    with col2:
        target = st.selectbox(
            "🔴 Destination",
            nodes,
            index=min(len(nodes) - 1, 5),
            key="route_dst",
        )
    with col3:
        algorithm = st.radio(
            "Algorithm",
            options=["Dijkstra", "A*", "Compare Both"],
            horizontal=True,
            key="route_algo",
        )

    if st.button("🔍 Find Optimal Route", key="find_route"):
        if source == target:
            st.warning("Origin and destination must be different.")
            return

        if algorithm == "Compare Both":
            results = optimizer.compare_routes(source, target)
            cols = st.columns(2)
            for idx, (algo_name, result) in enumerate(results.items()):
                with cols[idx]:
                    if result:
                        st.markdown(f"**{result.algorithm}**")
                        st.markdown(f"- Path: `{' → '.join(str(n) for n in result.path)}`")
                        st.markdown(f"- Travel time: **{result.total_travel_time} min**")
                        st.markdown(f"- Distance: **{result.total_distance_km} km**")
                        st.markdown(f"- Congested roads en route: {result.num_congested_roads}")
                        st.plotly_chart(
                            route_visualisation(network, result.path, result.algorithm),
                            use_container_width=True,
                        )
                    else:
                        st.error(f"No path found with {algo_name.capitalize()}.")
        else:
            if algorithm == "Dijkstra":
                result = optimizer.dijkstra(source, target)
            else:
                result = optimizer.astar(source, target)

            if result:
                st.success(
                    f"✅ Route found: {len(result.path)} nodes · "
                    f"{result.total_travel_time} min · {result.total_distance_km} km"
                )
                col_info, col_map = st.columns([1, 2])
                with col_info:
                    st.markdown(f"**Path**: `{' → '.join(str(n) for n in result.path)}`")
                    st.markdown(f"**Congested roads**: {result.num_congested_roads}")
                with col_map:
                    st.plotly_chart(
                        route_visualisation(network, result.path, result.algorithm),
                        use_container_width=True,
                    )
            else:
                st.error("No path found between the selected nodes.")

    st.divider()


def render_anomaly_section(df) -> None:
    """Render the anomaly detection results."""
    if "is_anomaly" not in df.columns:
        return

    st.subheader("⚠️ Anomaly Detection")
    anomalies = df[df["is_anomaly"]]

    col1, col2 = st.columns(2)
    col1.metric("Total Anomalies", len(anomalies))
    col2.metric(
        "Anomaly Rate",
        f"{len(anomalies) / max(len(df), 1) * 100:.2f}%",
    )

    if not anomalies.empty:
        top = (
            anomalies.groupby("road_id")["anomaly_score"]
            .mean()
            .nlargest(10)
            .reset_index()
        )
        st.markdown("**Top 10 Roads by Anomaly Score**")
        st.dataframe(top, use_container_width=True, hide_index=True)

    st.divider()


def render_raw_data(df) -> None:
    """Expandable raw data viewer."""
    with st.expander("📄 Raw Simulation Data (sample)"):
        sample = df.sample(min(200, len(df)), random_state=42)
        st.dataframe(sample, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* =====================================================
           UrbanPulse AI — Custom Theme
           Dark background · high-contrast text · accent blue
        ===================================================== */

        /* ---------- Global typography ---------- */
        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            color: #e8eaf0;
        }
        .stApp { background-color: #0b0e14; }

        /* ---------- Headings ---------- */
        h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
        h1 { font-size: 2rem !important; font-weight: 700 !important; }
        h2 { font-size: 1.5rem !important; font-weight: 600 !important; }
        h3 { font-size: 1.15rem !important; font-weight: 600 !important; }

        /* ---------- Subheaders / section titles ---------- */
        [data-testid="stMarkdownContainer"] h3 { color: #90caf9 !important; }
        .stSubheader, [data-testid="stSubheader"] { color: #90caf9 !important; }

        /* ---------- Body text & paragraphs ---------- */
        p, li, span, label { color: #c9d1d9 !important; }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #1f2937;
        }
        [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: #93c5fd !important; }
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #94a3b8 !important;
            font-size: 0.78rem;
        }

        /* ---------- KPI / Metric cards ---------- */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #1e2740 0%, #1a2035 100%);
            border-radius: 12px;
            padding: 18px 20px;
            border: 1px solid #2d3a5a;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(33, 150, 243, 0.2);
        }
        [data-testid="metric-container"] label,
        [data-testid="metric-container"] [data-testid="stMetricLabel"] {
            color: #90caf9 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 1.9rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }
        [data-testid="metric-container"] [data-testid="stMetricDelta"] {
            color: #4ade80 !important;
            font-size: 0.8rem !important;
        }

        /* ---------- Buttons ---------- */
        div.stButton > button {
            background: linear-gradient(135deg, #1976D2, #2196F3);
            color: #ffffff !important;
            border: none;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
            padding: 10px 20px;
            box-shadow: 0 3px 8px rgba(33, 150, 243, 0.35);
            transition: background 0.2s ease, box-shadow 0.2s ease, transform 0.1s ease;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #1565C0, #1976D2);
            box-shadow: 0 5px 16px rgba(33, 150, 243, 0.5);
            transform: translateY(-1px);
        }
        div.stButton > button:active { transform: translateY(0); }

        /* ---------- Tabs ---------- */
        [data-testid="stTabs"] [role="tab"] {
            color: #94a3b8 !important;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 8px 16px;
            border-radius: 6px 6px 0 0;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: #60a5fa !important;
            border-bottom: 2px solid #2196F3 !important;
        }
        [data-testid="stTabs"] [role="tab"]:hover { color: #bfdbfe !important; }

        /* ---------- Selectbox / Dropdowns ---------- */
        [data-testid="stSelectbox"] label { color: #94a3b8 !important; font-size: 0.85rem; }
        [data-testid="stSelectbox"] [data-baseweb="select"] {
            background-color: #1a2035 !important;
            border-radius: 8px;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] * { color: #e2e8f0 !important; }

        /* ---------- Sliders ---------- */
        [data-testid="stSlider"] label { color: #94a3b8 !important; font-size: 0.85rem; }
        [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
            background-color: #2196F3 !important;
        }

        /* ---------- Radio ---------- */
        [data-testid="stRadio"] label { color: #cbd5e1 !important; }
        [data-testid="stRadio"] [data-testid="stWidgetLabel"] { color: #94a3b8 !important; }

        /* ---------- Number input ---------- */
        [data-testid="stNumberInput"] label { color: #94a3b8 !important; font-size: 0.85rem; }
        [data-testid="stNumberInput"] input {
            background-color: #1a2035 !important;
            color: #e2e8f0 !important;
            border-color: #2d3a5a !important;
        }

        /* ---------- Expander ---------- */
        [data-testid="stExpander"] {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 10px;
        }
        [data-testid="stExpander"] summary { color: #93c5fd !important; font-weight: 600; }

        /* ---------- Dataframes / Tables ---------- */
        [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
        [data-testid="stDataFrame"] th {
            background-color: #1e2740 !important;
            color: #90caf9 !important;
            font-weight: 700;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        [data-testid="stDataFrame"] td { color: #e2e8f0 !important; font-size: 0.88rem; }
        [data-testid="stDataFrame"] tr:nth-child(even) td {
            background-color: #131924 !important;
        }

        /* ---------- Alerts (info / warning / success / error) ---------- */
        [data-testid="stAlert"] {
            border-radius: 10px;
            font-weight: 500;
        }
        [data-testid="stAlert"][kind="info"] { border-left: 4px solid #2196F3; }
        [data-testid="stAlert"][kind="warning"] { border-left: 4px solid #FF9800; }
        [data-testid="stAlert"][kind="success"] { border-left: 4px solid #4CAF50; }
        [data-testid="stAlert"][kind="error"] { border-left: 4px solid #F44336; }

        /* ---------- Divider ---------- */
        hr { border-color: #1f2937 !important; margin: 10px 0 !important; }

        /* ---------- Spinner text ---------- */
        [data-testid="stSpinner"] p { color: #90caf9 !important; }

        /* ---------- Custom components ---------- */

        /* Page header */
        .up-header {
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 16px 0 8px;
        }
        .up-header-icon { font-size: 3rem; line-height: 1; }
        .up-header-title {
            margin: 0 !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            color: #60a5fa;  /* fallback for browsers without background-clip:text support */
            background: linear-gradient(90deg, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .up-header-subtitle {
            margin: 4px 0 0 !important;
            color: #64748b !important;
            font-size: 0.9rem !important;
        }

        /* Welcome banner */
        .up-welcome-banner {
            background: linear-gradient(135deg, #1e3a5f 0%, #1a2e4a 100%);
            border: 1px solid #2563eb;
            border-left: 4px solid #3b82f6;
            border-radius: 10px;
            padding: 16px 22px;
            color: #bfdbfe !important;
            font-size: 1.05rem;
            font-weight: 500;
        }
        .up-welcome-banner strong { color: #93c5fd !important; }

        /* Info cards on welcome screen */
        .up-info-card {
            background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 22px 24px;
            height: 100%;
            /* min-height keeps both cards equal when one has more content */
            min-height: 280px;
        }
        .up-info-card h3 { color: #93c5fd !important; margin-bottom: 12px; }
        .up-info-card p { color: #94a3b8 !important; margin-bottom: 14px; }
        .up-info-card strong { color: #e2e8f0 !important; }

        /* Welcome table inside info card */
        .up-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.87rem;
            margin-top: 6px;
        }
        .up-table th {
            color: #60a5fa !important;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.06em;
            border-bottom: 1px solid #1f2937;
            padding: 6px 8px;
            text-align: left;
        }
        .up-table td {
            color: #cbd5e1 !important;
            padding: 6px 8px;
            border-bottom: 1px solid #0f172a;
        }
        .up-table tr:hover td { background-color: #1e2740; }

        /* Feature list */
        .up-feature-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .up-feature-list li {
            color: #cbd5e1 !important;
            padding: 7px 0;
            border-bottom: 1px solid #1f2937;
            font-size: 0.92rem;
        }
        .up-feature-list li:last-child { border-bottom: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )
