# UrbanPulse AI — Smart City Traffic Intelligence System

> A production-quality AI platform that simulates city traffic, predicts congestion, detects anomalies, and optimises routes — all rendered in an interactive Streamlit dashboard.

---

## Features

| Feature | Technology |
|---|---|
| City Road Network | NetworkX directed graph |
| Traffic Simulation | Stochastic event-driven generator |
| Flow Prediction | RandomForest Regressor (Scikit-learn) |
| Anomaly Detection | IsolationForest (Scikit-learn) |
| Route Optimisation | Dijkstra + A* algorithms |
| Visualisation | Plotly interactive charts |
| Dashboard | Streamlit |

---

## Project Structure

```
urbanpulse-ai/
├── app.py                      # Entry point — streamlit run app.py
├── config.py                   # Global constants & tuneable parameters
├── requirements.txt
│
├── data/
│   └── traffic_generator.py    # Stochastic traffic simulation
│
├── models/
│   ├── traffic_predictor.py    # RandomForest traffic flow predictor
│   └── congestion_detector.py  # IsolationForest anomaly detector
│
├── network/
│   ├── road_network.py         # NetworkX city graph
│   └── route_optimizer.py      # Dijkstra & A* route optimisation
│
├── analytics/
│   └── traffic_metrics.py      # KPI calculations & aggregations
│
├── ui/
│   ├── dashboard.py            # Dashboard layout & section renderers
│   ├── charts.py               # Plotly chart builders
│   └── controls.py             # Streamlit sidebar controls
│
└── utils/
    └── helpers.py              # Shared utility functions
```

---

## Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Launch the dashboard

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser, configure the simulation in the sidebar, and click **▶ Run Simulation**.

---

## Dashboard Sections

### 📈 Traffic Overview
KPI tiles: Total Vehicles · Congested Roads · Average Speed · Flow Index · Anomaly Rate

### 🤖 Traffic Prediction Analytics
- Predicted vs Actual scatter plot
- Congestion heatmap (road × hour)
- Feature importance bar chart
- Weather impact chart

### 🗺️ Road Network Visualisation
Interactive graph with normal roads in **blue** and congested roads in **red**.

### 🧭 Route Optimisation
Choose origin/destination, pick Dijkstra or A* (or compare both), and see the optimal route highlighted in orange.

### ⚠️ Anomaly Detection
IsolationForest flags traffic spikes and abnormal road usage patterns.

---

## Configuration

All tuneable parameters live in `config.py` — no magic numbers anywhere else:

- Network size (nodes / edges)
- Traffic capacity ranges
- Peak-hour multipliers
- Weather conditions & weights
- ML hyperparameters (RF n_estimators, IsolationForest contamination, …)
- Dashboard styling

---

## Deployment on Streamlit Cloud

1. Fork the repository.
2. Connect it to [Streamlit Cloud](https://streamlit.io/cloud).
3. Set **Main file path** → `app.py`.
4. Click **Deploy**.

The app runs directly from `requirements.txt` — no extra configuration needed.
