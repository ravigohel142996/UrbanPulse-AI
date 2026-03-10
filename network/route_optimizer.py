"""
network/route_optimizer.py — Dijkstra and A* route optimisation.

Finds fastest / least-congested routes through the city graph.
"""

from __future__ import annotations

from typing import NamedTuple

import networkx as nx
import numpy as np

from config import CONGESTION_PENALTY, CONGESTION_THRESHOLD
from network.road_network import CityRoadNetwork, RoadSegment


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class RouteResult(NamedTuple):
    """Immutable result returned by the route optimizer."""

    algorithm: str
    source: int
    target: int
    path: list[int]
    total_travel_time: float       # minutes
    total_distance_km: float
    num_congested_roads: int
    congestion_avoidance: bool


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class RouteOptimizer:
    """Wraps NetworkX shortest-path algorithms with congestion awareness."""

    def __init__(self, network: CityRoadNetwork) -> None:
        self.network = network

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dijkstra(
        self,
        source: int,
        target: int,
        avoid_congestion: bool = True,
    ) -> RouteResult | None:
        """Find the shortest (least-cost) path using Dijkstra's algorithm.

        Parameters
        ----------
        source, target:
            Node IDs in the city graph.
        avoid_congestion:
            When True the edge weights include a congestion penalty so the
            algorithm naturally routes around heavy traffic.
        """
        G = self._weighted_graph(avoid_congestion)
        try:
            path = nx.dijkstra_path(G, source, target, weight="weight")
            return self._build_result("Dijkstra", source, target, path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def astar(
        self,
        source: int,
        target: int,
        avoid_congestion: bool = True,
    ) -> RouteResult | None:
        """Find the shortest path using A* with a Euclidean heuristic.

        Parameters
        ----------
        source, target:
            Node IDs in the city graph.
        avoid_congestion:
            Same as in :meth:`dijkstra`.
        """
        G = self._weighted_graph(avoid_congestion)

        def heuristic(u: int, v: int) -> float:
            pos_u = self.network.graph.nodes[u].get("pos", (0, 0))
            pos_v = self.network.graph.nodes[v].get("pos", (0, 0))
            # Euclidean distance in the layout space (proxy for travel time)
            return float(np.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1]))

        try:
            path = nx.astar_path(G, source, target, heuristic=heuristic, weight="weight")
            return self._build_result("A*", source, target, path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def compare_routes(
        self,
        source: int,
        target: int,
    ) -> dict[str, RouteResult | None]:
        """Return both Dijkstra and A* results for side-by-side comparison."""
        return {
            "dijkstra": self.dijkstra(source, target),
            "astar": self.astar(source, target),
        }

    def all_pairs_shortest_paths(self) -> dict[tuple[int, int], float]:
        """Compute all-pairs shortest travel times (minutes).

        Returns a dict keyed by (source, target) node pairs.
        """
        lengths: dict[tuple[int, int], float] = {}
        G = self._weighted_graph(avoid_congestion=True)
        for source, targets in nx.all_pairs_dijkstra_path_length(G, weight="weight"):
            for target, dist in targets.items():
                if source != target:
                    lengths[(source, target)] = dist
        return lengths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _weighted_graph(self, avoid_congestion: bool) -> nx.DiGraph:
        """Return a copy of the graph with updated edge weights."""
        G = self.network.graph.copy()
        if avoid_congestion:
            for u, v, data in G.edges(data=True):
                seg: RoadSegment = data["segment"]
                data["weight"] = seg.effective_travel_time(CONGESTION_PENALTY)
        else:
            for u, v, data in G.edges(data=True):
                seg = data["segment"]
                data["weight"] = seg.travel_time
        return G

    def _build_result(
        self,
        algorithm: str,
        source: int,
        target: int,
        path: list[int],
    ) -> RouteResult:
        """Aggregate path statistics into a RouteResult."""
        total_time = 0.0
        total_dist = 0.0
        congested_count = 0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.network.graph.has_edge(u, v):
                seg: RoadSegment = self.network.graph[u][v]["segment"]
                total_time += seg.effective_travel_time(CONGESTION_PENALTY)
                total_dist += seg.road_length
                if seg.is_congested:
                    congested_count += 1

        return RouteResult(
            algorithm=algorithm,
            source=source,
            target=target,
            path=path,
            total_travel_time=round(total_time, 2),
            total_distance_km=round(total_dist, 2),
            num_congested_roads=congested_count,
            congestion_avoidance=True,
        )
