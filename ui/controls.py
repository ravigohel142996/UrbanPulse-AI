"""
ui/controls.py — Streamlit sidebar controls for UrbanPulse AI.

Renders all simulation parameters and returns them as a typed dataclass
so the rest of the UI code never calls ``st.sidebar`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from config import (
    DEFAULT_NUM_VEHICLES,
    DEFAULT_SIMULATION_HOURS,
    NETWORK_NUM_EDGES,
    NETWORK_NUM_NODES,
    WEATHER_CONDITIONS,
)


@dataclass
class SimulationParams:
    """Parameters collected from the sidebar."""

    num_vehicles: int
    simulation_hours: int
    weather_filter: str           # "All" or a specific condition
    traffic_intensity: float      # 0.5 – 2.0 multiplier
    num_nodes: int
    num_edges: int
    run_simulation: bool


def render_sidebar() -> SimulationParams:
    """Render the sidebar and return the collected parameters."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/traffic-light.png",
            width=80,
        )
        st.title("UrbanPulse AI")
        st.caption("Smart City Traffic Intelligence")
        st.divider()

        # ---- Simulation controls ----
        st.subheader("⚙️ Simulation Parameters")

        num_vehicles = st.slider(
            "Number of Vehicles",
            min_value=100,
            max_value=2_000,
            value=DEFAULT_NUM_VEHICLES,
            step=50,
            help="Base vehicle count active during the simulation.",
        )

        simulation_hours = st.slider(
            "Simulation Duration (hours)",
            min_value=1,
            max_value=48,
            value=DEFAULT_SIMULATION_HOURS,
            step=1,
            help="Total time window simulated.",
        )

        traffic_intensity = st.slider(
            "Traffic Intensity Multiplier",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            format="%.1f×",
            help="Scales the overall vehicle volume up or down.",
        )

        st.divider()

        # ---- Network controls ----
        st.subheader("🗺️ Network Configuration")

        num_nodes = st.number_input(
            "Intersections (nodes)",
            min_value=10,
            max_value=100,
            value=NETWORK_NUM_NODES,
            step=5,
        )

        num_edges = st.number_input(
            "Road Segments (edges)",
            min_value=15,
            max_value=300,
            value=NETWORK_NUM_EDGES,
            step=5,
        )

        st.divider()

        # ---- Filter ----
        st.subheader("🌦️ Weather Filter")
        weather_filter = st.selectbox(
            "Show data for weather",
            options=["All"] + WEATHER_CONDITIONS,
            index=0,
        )

        st.divider()

        # ---- Run button ----
        run_simulation = st.button(
            "▶ Run Simulation",
            use_container_width=True,
            type="primary",
        )

        # ---- Info footer ----
        st.divider()
        st.caption(
            "UrbanPulse AI v1.0 · "
            "Powered by Scikit-learn, NetworkX & Plotly"
        )

    return SimulationParams(
        num_vehicles=int(num_vehicles * traffic_intensity),
        simulation_hours=int(simulation_hours),
        weather_filter=weather_filter,
        traffic_intensity=float(traffic_intensity),
        num_nodes=int(num_nodes),
        num_edges=int(num_edges),
        run_simulation=run_simulation,
    )
