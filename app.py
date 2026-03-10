"""
app.py — Entry point for UrbanPulse AI.

Run with:
    streamlit run app.py

Architecture:
    1. Sidebar renders simulation parameters.
    2. On "Run Simulation":
       a. Build / rebuild road network.
       b. Generate synthetic traffic data.
       c. Train RandomForest predictor + IsolationForest detector.
       d. Annotate data with predictions and anomaly flags.
    3. Render all dashboard sections.
"""

from __future__ import annotations

import streamlit as st

from config import RANDOM_SEED
from data.traffic_generator import TrafficDataGenerator
from models.congestion_detector import CongestionDetector
from models.traffic_predictor import TrafficPredictor
from network.road_network import CityRoadNetwork
from ui.controls import SimulationParams, render_sidebar
from ui.dashboard import (
    configure_page,
    render_anomaly_section,
    render_header,
    render_kpi_overview,
    render_network_visualisation,
    render_prediction_analytics,
    render_raw_data,
    render_route_optimisation,
)
from analytics.traffic_metrics import compute_kpis


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _get_or_init(key: str, factory):
    """Return st.session_state[key], initialising it with factory() if absent."""
    if key not in st.session_state:
        st.session_state[key] = factory()
    return st.session_state[key]


# ---------------------------------------------------------------------------
# Simulation pipeline
# ---------------------------------------------------------------------------

def run_pipeline(params: SimulationParams) -> None:
    """Build network, generate data, fit models, store results in session."""
    with st.spinner("🔧 Building city road network…"):
        network = CityRoadNetwork(
            num_nodes=params.num_nodes,
            num_edges=params.num_edges,
            seed=RANDOM_SEED,
        )
        st.session_state["network"] = network

    with st.spinner("🚗 Simulating traffic…"):
        generator = TrafficDataGenerator(network, seed=RANDOM_SEED)
        df = generator.generate(
            num_vehicles=params.num_vehicles,
            simulation_hours=params.simulation_hours,
        )
        st.session_state["df_raw"] = df

    with st.spinner("🤖 Training traffic predictor…"):
        predictor = TrafficPredictor()
        predictor.fit(df)
        df_predicted = predictor.predict_with_actuals(df)
        st.session_state["predictor"] = predictor
        st.session_state["df_predicted"] = df_predicted

    with st.spinner("🔍 Detecting anomalies…"):
        detector = CongestionDetector()
        detector.fit(df)
        df_annotated = detector.detect(df)
        st.session_state["df_annotated"] = df_annotated

    with st.spinner("🗺️ Updating network with traffic state…"):
        latest = df.sort_values("timestamp").groupby("road_id").last()
        traffic_map = latest["vehicle_count"].to_dict()
        network.update_traffic(traffic_map)

    st.session_state["simulation_complete"] = True
    st.success("✅ Simulation complete!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_page()
    params = render_sidebar()
    render_header()

    if params.run_simulation:
        run_pipeline(params)

    if not st.session_state.get("simulation_complete"):
        # Welcome screen
        st.markdown(
            """
            <div class="up-welcome-banner">
                👈&nbsp; Configure simulation parameters in the sidebar, then click
                <strong>▶ Run Simulation</strong> to start.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        col_about, col_features = st.columns(2)
        with col_about:
            st.markdown(
                """
                <div class="up-info-card">
                    <h3>🚦 About UrbanPulse AI</h3>
                    <p>A <strong>Smart City Traffic Intelligence Platform</strong>
                    built with cutting-edge ML and graph algorithms.</p>
                    <table class="up-table">
                        <thead><tr><th>Component</th><th>Technology</th></tr></thead>
                        <tbody>
                            <tr><td>City Road Network</td><td>NetworkX directed graph</td></tr>
                            <tr><td>Traffic Simulation</td><td>Stochastic event-driven generator</td></tr>
                            <tr><td>Flow Prediction</td><td>RandomForest Regressor</td></tr>
                            <tr><td>Anomaly Detection</td><td>IsolationForest</td></tr>
                            <tr><td>Route Optimisation</td><td>Dijkstra + A* algorithms</td></tr>
                            <tr><td>Visualisation</td><td>Plotly interactive charts</td></tr>
                            <tr><td>Dashboard</td><td>Streamlit</td></tr>
                        </tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_features:
            st.markdown(
                """
                <div class="up-info-card">
                    <h3>✨ Features</h3>
                    <ul class="up-feature-list">
                        <li>🕐 Rush-hour peak traffic patterns</li>
                        <li>🌦️ Weather-impact modelling (clear / rain / fog / snow)</li>
                        <li>🎉 Special-event traffic spikes</li>
                        <li>🛣️ Congestion-aware route optimisation</li>
                        <li>📊 Real-time KPI tiles &amp; anomaly rate</li>
                        <li>🤖 ML-powered traffic predictions</li>
                        <li>🗺️ Interactive road network graph</li>
                        <li>🔍 Automated anomaly detection</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # Retrieve stored objects
    network: CityRoadNetwork = st.session_state["network"]
    df_annotated = st.session_state["df_annotated"]
    predictor: TrafficPredictor = st.session_state["predictor"]

    # Filter by weather if requested
    df_display = df_annotated.copy()
    if params.weather_filter != "All":
        df_display = df_display[
            df_display["weather_condition"] == params.weather_filter
        ]
        if df_display.empty:
            st.warning(
                f"No records for weather condition '{params.weather_filter}'. "
                "Showing all data."
            )
            df_display = df_annotated.copy()

    # Compute KPIs
    kpis = compute_kpis(df_display, network.graph.number_of_edges())

    # Render sections
    render_kpi_overview(kpis)
    render_prediction_analytics(df_display, predictor)
    render_network_visualisation(network)
    render_route_optimisation(network)
    render_anomaly_section(df_display)
    render_raw_data(df_display)


if __name__ == "__main__":
    main()
