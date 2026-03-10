"""
network/road_network.py — City road-network graph built with NetworkX.

Nodes  = intersections
Edges  = road segments with rich attributes
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from config import (
    NETWORK_NUM_EDGES,
    NETWORK_NUM_NODES,
    RANDOM_SEED,
    ROAD_LENGTH_MAX,
    ROAD_LENGTH_MIN,
    TRAFFIC_CAPACITY_MAX,
    TRAFFIC_CAPACITY_MIN,
    TRAVEL_TIME_MAX,
    TRAVEL_TIME_MIN,
)
from utils.helpers import congestion_ratio


# ---------------------------------------------------------------------------
# Data class for a road segment
# ---------------------------------------------------------------------------

@dataclass
class RoadSegment:
    """Attributes carried by every edge in the city graph."""

    road_id: str
    road_length: float        # km
    traffic_capacity: int     # vehicles / hour
    travel_time: float        # minutes (free-flow)
    current_traffic: int = 0  # vehicles currently on the road
    road_type: str = "arterial"

    @property
    def congestion_ratio(self) -> float:
        """Fraction of capacity currently used, clamped to [0, 1]."""
        return congestion_ratio(self.current_traffic, self.traffic_capacity)

    @property
    def is_congested(self) -> bool:
        """True when the road exceeds 75 % capacity."""
        return self.congestion_ratio >= 0.75

    def effective_travel_time(self, penalty: float = 5.0) -> float:
        """Travel time after applying a congestion penalty."""
        return self.travel_time * (1 + penalty * self.congestion_ratio ** 2)


# ---------------------------------------------------------------------------
# Road types
# ---------------------------------------------------------------------------

ROAD_TYPES: list[str] = ["highway", "arterial", "collector", "local"]
ROAD_TYPE_CAPACITY_FACTOR: dict[str, float] = {
    "highway": 2.0,
    "arterial": 1.0,
    "collector": 0.6,
    "local": 0.3,
}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class CityRoadNetwork:
    """Manages the NetworkX directed graph representing the city."""

    def __init__(
        self,
        num_nodes: int = NETWORK_NUM_NODES,
        num_edges: int = NETWORK_NUM_EDGES,
        seed: int = RANDOM_SEED,
    ) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.graph: nx.DiGraph = self._build_graph()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> nx.DiGraph:
        """Generate a directed city graph with random road attributes."""
        # Seed the base undirected random graph then convert to directed
        base = nx.gnm_random_graph(
            self.num_nodes,
            self.num_edges,
            seed=self._rng.randint(0, 10_000),
        )
        G: nx.DiGraph = base.to_directed()

        # Assign node positions (lat/lon-like coords for visualisation)
        pos = nx.spring_layout(G, seed=self._rng.randint(0, 10_000))
        for node, (x, y) in pos.items():
            G.nodes[node]["pos"] = (float(x), float(y))
            G.nodes[node]["name"] = f"Intersection {node}"

        # Assign edge attributes
        for u, v in G.edges():
            road_type = self._rng.choice(ROAD_TYPES)
            cap_factor = ROAD_TYPE_CAPACITY_FACTOR[road_type]
            capacity = int(
                self._rng.uniform(TRAFFIC_CAPACITY_MIN, TRAFFIC_CAPACITY_MAX)
                * cap_factor
            )
            length = round(self._rng.uniform(ROAD_LENGTH_MIN, ROAD_LENGTH_MAX), 2)
            travel_time = round(
                self._rng.uniform(TRAVEL_TIME_MIN, TRAVEL_TIME_MAX), 1
            )
            segment = RoadSegment(
                road_id=f"R{u}-{v}",
                road_length=length,
                traffic_capacity=max(50, capacity),
                travel_time=travel_time,
                road_type=road_type,
            )
            G[u][v]["segment"] = segment
            G[u][v]["weight"] = travel_time   # default weight = free-flow time

        return G

    # ------------------------------------------------------------------
    # Traffic updates
    # ------------------------------------------------------------------

    def update_traffic(self, traffic_data: dict[str, int]) -> None:
        """Push current vehicle counts from a simulation snapshot.

        Parameters
        ----------
        traffic_data:
            Mapping of road_id → vehicle_count.
        """
        for u, v, data in self.graph.edges(data=True):
            seg: RoadSegment = data["segment"]
            if seg.road_id in traffic_data:
                seg.current_traffic = traffic_data[seg.road_id]
            # Update edge weight to reflect congestion
            data["weight"] = seg.effective_travel_time()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_congested_edges(self) -> list[tuple[int, int, RoadSegment]]:
        """Return list of (u, v, segment) tuples for congested roads."""
        return [
            (u, v, data["segment"])
            for u, v, data in self.graph.edges(data=True)
            if data["segment"].is_congested
        ]

    def congestion_summary(self) -> dict[str, Any]:
        """Return aggregate network-level congestion stats."""
        ratios = [
            data["segment"].congestion_ratio
            for _, _, data in self.graph.edges(data=True)
        ]
        if not ratios:
            return {}
        arr = np.array(ratios)
        return {
            "total_roads": len(ratios),
            "congested_roads": int((arr >= 0.75).sum()),
            "avg_congestion_ratio": float(arr.mean()),
            "max_congestion_ratio": float(arr.max()),
        }

    def edge_list_df(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """Return a DataFrame with one row per edge and key attributes."""
        import pandas as pd

        rows = []
        for u, v, data in self.graph.edges(data=True):
            seg: RoadSegment = data["segment"]
            rows.append(
                {
                    "road_id": seg.road_id,
                    "from_node": u,
                    "to_node": v,
                    "road_type": seg.road_type,
                    "road_length_km": seg.road_length,
                    "capacity": seg.traffic_capacity,
                    "current_traffic": seg.current_traffic,
                    "congestion_ratio": round(seg.congestion_ratio, 3),
                    "is_congested": seg.is_congested,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        stats = self.congestion_summary()
        return (
            f"CityRoadNetwork("
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, "
            f"congested={stats.get('congested_roads', 0)})"
        )
