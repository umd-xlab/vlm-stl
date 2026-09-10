from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import random
from typing import Callable, Iterable

import numpy as np


Point = tuple[float, float]
CostFunction = Callable[[Point], float]
CollisionFunction = Callable[[Point, Point], bool]


@dataclass
class RRTXConfig:
    """Configuration values for the planner.

    Inputs:
        bounds: Planning bounds as (xmin, xmax, ymin, ymax).
        step_size: Maximum distance added by one tree extension.
        neighbor_radius: Radius used to find nearby nodes for rewiring.
        goal_radius: Distance at which a node can connect to the goal.
        max_iterations: Number of random samples to try.
        goal_sample_rate: Probability of sampling the goal directly.
        collision_resolution: Spacing used when checking an edge for collision.
        rewire: Whether to improve nearby parent links after adding nodes.
        seed: Optional random seed for repeatable plans.
    """

    bounds: tuple[float, float, float, float]
    step_size: float = 0.5
    neighbor_radius: float = 1.5
    goal_radius: float = 0.75
    max_iterations: int = 1500
    goal_sample_rate: float = 0.1
    collision_resolution: float = 0.1
    rewire: bool = True
    seed: int | None = None


@dataclass(eq=False)
class RRTXNode:
    """One vertex in the RRT-X tree/graph.

    Inputs:
        point: 2D world coordinate for this node.
        parent: Current best predecessor toward the goal-root.
        children: Nodes farther from the goal-root that currently use this node as parent.
        neighbors: Nearby nodes that can be considered for rewiring.
        cost_to_come: Current committed path cost from the goal-root to this node.
        lmc: One-step lookahead cost used by RRT-X consistency repair.
        edge_cost: Cost of the edge from parent, closer to the goal-root, to this node.
        blocked: Whether this node was disconnected by a map/collision change.
    """

    point: Point
    parent: RRTXNode | None = None
    children: set[RRTXNode] = field(default_factory=set)
    neighbors: set[RRTXNode] = field(default_factory=set)
    cost_to_come: float = math.inf
    lmc: float = math.inf
    edge_cost: float = math.inf
    blocked: bool = False

    def __hash__(self) -> int:
        """Allow nodes to be stored in sets by object identity."""
        return id(self)


class GridCostmap:
    """Adapter that turns a 2D numpy costmap into planner callbacks."""

    def __init__(self, costmap: np.ndarray, origin: Point = (0.0, 0.0), resolution: float = 1.0, lethal_cost: float = 255.0):
        """Store costmap metadata.

        Inputs:
            costmap: 2D array indexed as costmap[row, col].
            origin: World coordinate of grid cell (0, 0).
            resolution: Meters per grid cell.
            lethal_cost: Cost value treated as an obstacle.
        """
        self.costmap = np.asarray(costmap, dtype=float)
        self.origin = origin
        self.resolution = resolution
        self.lethal_cost = lethal_cost

    @property
    def height(self) -> int:
        """Return number of grid rows."""
        return int(self.costmap.shape[0])

    @property
    def width(self) -> int:
        """Return number of grid columns."""
        return int(self.costmap.shape[1])

    def world_to_grid(self, point: Point) -> tuple[int, int] | None:
        """Convert a world point into a grid index.

        Inputs:
            point: 2D world coordinate.
        Returns:
            (row, col) if the point is inside the map, otherwise None.
        """
        x = int(math.floor((point[0] - self.origin[0]) / self.resolution))
        y = int(math.floor((point[1] - self.origin[1]) / self.resolution))
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return None
        return y, x

    def cost_at(self, point: Point) -> float:
        """Return the map cost at a world point.

        Inputs:
            point: 2D world coordinate.
        Returns:
            Cell cost, or lethal_cost if outside the map.
        """
        cell = self.world_to_grid(point)
        if cell is None:
            return self.lethal_cost
        return float(self.costmap[cell])

    def is_collision_free(self, start: Point, end: Point, resolution: float = 0.1) -> bool:
        """Check whether a straight edge crosses a lethal cell.

        Inputs:
            start: Edge start point.
            end: Edge end point.
            resolution: Distance between sampled points on the edge.
        Returns:
            True if every sampled point is below lethal_cost.
        """
        distance = euclidean(start, end)
        steps = max(1, int(math.ceil(distance / resolution)))
        for i in range(steps + 1):
            t = i / steps
            point = (start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1]))
            if self.cost_at(point) >= self.lethal_cost:
                return False
        return True


class RRTXPlanner:
    """RRT-X-inspired incremental planner for 2D waypoint generation."""

    def __init__(
        self,
        config: RRTXConfig,
        cost_function: CostFunction | None = None,
        collision_function: CollisionFunction | None = None,
    ):
        """Create a planner instance.

        Inputs:
            config: Planner tuning parameters.
            cost_function: Optional point cost callback, cost_function(point) -> float.
            collision_function: Optional edge validity callback, collision_function(start, end) -> bool.
        """
        self.config = config
        self.cost_function = cost_function or (lambda _: 0.0)
        self.collision_function = collision_function or (lambda _, __: True)
        self.nodes: list[RRTXNode] = []
        self.start: RRTXNode | None = None
        self.goal: RRTXNode | None = None
        self.queue: list[tuple[tuple[float, float], int, RRTXNode]] = []
        self.queue_counter = 0
        self.rng = random.Random(config.seed)

    @classmethod
    def from_grid_costmap(
        cls,
        config: RRTXConfig,
        costmap: np.ndarray,
        origin: Point = (0.0, 0.0),
        resolution: float = 1.0,
        lethal_cost: float = 255.0,
    ) -> RRTXPlanner:
        """Build a planner using a numpy grid as both cost and collision source.

        Inputs:
            config: Planner tuning parameters.
            costmap: 2D grid of traversal costs.
            origin: World coordinate of grid cell (0, 0).
            resolution: Meters per grid cell.
            lethal_cost: Cost value treated as an obstacle.
        Returns:
            Configured RRTXPlanner.
        """
        grid = GridCostmap(costmap, origin=origin, resolution=resolution, lethal_cost=lethal_cost)
        return cls(
            config=config,
            cost_function=grid.cost_at,
            collision_function=lambda start, end: grid.is_collision_free(start, end, config.collision_resolution),
        )

    def plan(self, start: Point, goal: Point) -> list[Point]:
        """Grow a goal-rooted tree toward the robot start and return start-to-goal waypoints.

        Inputs:
            start: Robot start point in world/map coordinates.
            goal: Goal point in world/map coordinates.
        Returns:
            Ordered waypoint list. Empty if no path is found.
        """
        self.reset(start, goal)
        for _ in range(self.config.max_iterations):
            sample = self.sample(start)
            nearest = self.nearest(sample)
            candidate_point = self.steer(nearest.point, sample)
            if not self.in_bounds(candidate_point) or not self.collision_function(nearest.point, candidate_point):
                continue
            candidate = RRTXNode(candidate_point)
            near_nodes = self.near(candidate.point)
            parent, _, _ = self.choose_parent(candidate, near_nodes or [nearest])
            if parent is None:
                continue
            self.nodes.append(candidate)
            self.connect_neighbors(candidate, near_nodes)
            self.make_parent(candidate, parent)
            self.verify_orphan(candidate)
            self.reduce_inconsistency()
            if self.config.rewire:
                self.rewire_neighbors(candidate, near_nodes)
                self.reduce_inconsistency()
            if self.start is not None and euclidean(candidate.point, self.start.point) <= self.config.goal_radius:
                self.try_connect_start(candidate)
                self.reduce_inconsistency()
        return self.get_path()

    def reset(self, start: Point, goal: Point) -> None:
        """Clear previous planning state and root the tree at the goal.

        Inputs:
            start: Robot start point.
            goal: Goal point.
        """
        if self.config.seed is not None:
            self.rng.seed(self.config.seed)
        self.queue.clear()
        self.queue_counter = 0
        self.start = RRTXNode(start)
        self.goal = RRTXNode(goal, cost_to_come=0.0, lmc=0.0, edge_cost=0.0)
        self.nodes = [self.goal, self.start]

    def update_environment(
        self,
        cost_function: CostFunction | None = None,
        collision_function: CollisionFunction | None = None,
        changed_region: tuple[float, float, float, float] | None = None,
    ) -> list[Point]:
        """Repair the existing tree after a costmap or obstacle update.

        Inputs:
            cost_function: Replacement point-cost callback, if costs changed.
            collision_function: Replacement edge-collision callback, if obstacles changed.
            changed_region: Optional affected bounds as (xmin, xmax, ymin, ymax). None repairs all nodes.
        Returns:
            Updated waypoint list. Empty if the goal is disconnected.
        """
        if cost_function is not None:
            self.cost_function = cost_function
        if collision_function is not None:
            self.collision_function = collision_function
        affected = self.affected_nodes(changed_region)
        for node in affected:
            self.verify_orphan(node)
            self.update_lmc(node)
            self.enqueue_if_inconsistent(node)
        self.reduce_inconsistency()
        return self.get_path()

    def sample(self, target: Point) -> Point:
        """Draw a random sample, sometimes biasing directly toward the target.

        Inputs:
            target: Target point used for biased sampling, usually the robot start.
        Returns:
            Sampled point inside planner bounds or the target point.
        """
        if self.rng.random() < self.config.goal_sample_rate:
            return target
        xmin, xmax, ymin, ymax = self.config.bounds
        return self.rng.uniform(xmin, xmax), self.rng.uniform(ymin, ymax)

    def nearest(self, point: Point) -> RRTXNode:
        """Find the existing node closest to a point.

        Inputs:
            point: Query point.
        Returns:
            Nearest node by Euclidean distance.
        """
        return min(self.nodes, key=lambda node: euclidean(node.point, point))

    def near(self, point: Point) -> list[RRTXNode]:
        """Find existing nodes within the rewiring radius.

        Inputs:
            point: Query point.
        Returns:
            Nodes within config.neighbor_radius.
        """
        return [node for node in self.nodes if euclidean(node.point, point) <= self.config.neighbor_radius and node.point != point]

    def steer(self, start: Point, target: Point) -> Point:
        """Move from start toward target by at most step_size.

        Inputs:
            start: Current tree point.
            target: Desired sample point.
        Returns:
            New candidate point.
        """
        distance = euclidean(start, target)
        if distance <= self.config.step_size:
            return target
        scale = self.config.step_size / distance
        return start[0] + scale * (target[0] - start[0]), start[1] + scale * (target[1] - start[1])

    def in_bounds(self, point: Point) -> bool:
        """Check whether a point lies inside planner bounds.

        Inputs:
            point: 2D point to test.
        Returns:
            True if point is inside (xmin, xmax, ymin, ymax).
        """
        xmin, xmax, ymin, ymax = self.config.bounds
        return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax

    def connect_neighbors(self, node: RRTXNode, near_nodes: Iterable[RRTXNode]) -> None:
        """Add undirected neighbor links for later rewiring/repair.

        Inputs:
            node: New node being inserted.
            near_nodes: Existing nearby nodes.
        """
        for other in near_nodes:
            if other is node:
                continue
            node.neighbors.add(other)
            other.neighbors.add(node)

    def choose_parent(self, node: RRTXNode, candidates: Iterable[RRTXNode]) -> tuple[RRTXNode | None, float, float]:
        """Select the lowest-cost collision-free parent without mutating tree links.

        Inputs:
            node: Node needing a parent.
            candidates: Potential parent nodes.
        Returns:
            Best parent, resulting lmc, and selected edge cost.
        """
        best_parent = None
        best_lmc = math.inf
        best_edge_cost = math.inf
        for candidate in candidates:
            if candidate is node or candidate.cost_to_come == math.inf:
                continue
            if not self.collision_function(candidate.point, node.point):
                continue
            edge_cost = self.transition_cost(candidate.point, node.point)
            candidate_cost = candidate.cost_to_come + edge_cost
            if candidate_cost < best_lmc:
                best_parent = candidate
                best_lmc = candidate_cost
                best_edge_cost = edge_cost
        return best_parent, best_lmc, best_edge_cost

    def make_parent(self, child: RRTXNode, parent: RRTXNode | None) -> None:
        """Attach child to parent and update tree costs.

        Inputs:
            child: Node being attached or reattached.
            parent: New parent node. None detaches child.
        """
        if child.parent is not None:
            child.parent.children.discard(child)
        child.parent = parent
        if parent is not None:
            parent.children.add(child)
            child.edge_cost = self.transition_cost(parent.point, child.point)
            child.cost_to_come = parent.cost_to_come + child.edge_cost
            child.lmc = child.cost_to_come

    def rewire_neighbors(self, node: RRTXNode, near_nodes: Iterable[RRTXNode]) -> None:
        """Try to improve nearby nodes by routing them through a new node.

        Inputs:
            node: Newly inserted or improved node.
            near_nodes: Nearby candidate nodes to rewire.
        """
        for other in near_nodes:
            if other is node or other is self.goal or other.blocked:
                continue
            if not self.collision_function(node.point, other.point):
                continue
            edge_cost = self.transition_cost(node.point, other.point)
            candidate_lmc = node.cost_to_come + edge_cost
            if candidate_lmc < other.lmc:
                self.make_parent(other, node)
                self.enqueue_if_inconsistent(other)

    def verify_orphan(self, node: RRTXNode) -> None:
        """Disconnect a node and descendants if its parent edge is now invalid.

        Inputs:
            node: Node whose parent edge should be checked.
        """
        if node.parent is None:
            return
        if self.collision_function(node.parent.point, node.point):
            node.blocked = False
            return
        old_parent = node.parent
        old_parent.children.discard(node)
        node.parent = None
        node.cost_to_come = math.inf
        node.lmc = math.inf
        node.edge_cost = math.inf
        node.blocked = True
        for child in list(node.children):
            self.verify_orphan(child)

    def update_lmc(self, node: RRTXNode) -> None:
        """Recompute a node's one-step lookahead cost from valid neighbors.

        Inputs:
            node: Node whose lmc value should be repaired.
        """
        if node is self.goal:
            node.lmc = 0.0
            return
        best_parent = None
        best_lmc = math.inf
        best_edge_cost = math.inf
        for neighbor in node.neighbors:
            if neighbor.cost_to_come == math.inf:
                continue
            if not self.collision_function(neighbor.point, node.point):
                continue
            edge_cost = self.transition_cost(neighbor.point, node.point)
            candidate_lmc = neighbor.cost_to_come + edge_cost
            if candidate_lmc < best_lmc:
                best_parent = neighbor
                best_lmc = candidate_lmc
                best_edge_cost = edge_cost
        if best_parent is not None and best_lmc < node.cost_to_come:
            self.make_parent(node, best_parent)
        else:
            node.lmc = best_lmc
            node.edge_cost = best_edge_cost

    def enqueue_if_inconsistent(self, node: RRTXNode) -> None:
        """Put a node in the repair queue if cost_to_come and lmc disagree.

        Inputs:
            node: Node to check for inconsistency.
        """
        if node.cost_to_come == node.lmc:
            return
        self.queue_counter += 1
        heapq.heappush(self.queue, (self.key(node), self.queue_counter, node))

    def reduce_inconsistency(self) -> None:
        """Process the repair queue until local inconsistencies are resolved."""
        while self.queue:
            _, _, node = heapq.heappop(self.queue)
            if node.cost_to_come == node.lmc:
                continue
            if node.lmc < node.cost_to_come:
                node.cost_to_come = node.lmc
            else:
                node.cost_to_come = math.inf
                self.update_lmc(node)
                self.enqueue_if_inconsistent(node)
            for neighbor in node.neighbors:
                self.update_lmc(neighbor)
                self.enqueue_if_inconsistent(neighbor)

    def key(self, node: RRTXNode) -> tuple[float, float]:
        """Compute priority key for the inconsistency queue.

        Inputs:
            node: Node to prioritize.
        Returns:
            Tuple ordered by best known cost, then committed cost.
        """
        value = min(node.cost_to_come, node.lmc)
        return value, node.cost_to_come

    def try_connect_start(self, node: RRTXNode) -> None:
        """Attempt to connect a nearby node directly to the robot start.

        Inputs:
            node: Candidate node within goal_radius of the start.
        """
        if self.start is None or not self.collision_function(node.point, self.start.point):
            return
        self.start.neighbors.add(node)
        node.neighbors.add(self.start)
        edge_cost = self.transition_cost(node.point, self.start.point)
        if node.cost_to_come + edge_cost < self.start.lmc:
            self.make_parent(self.start, node)
            self.enqueue_if_inconsistent(self.start)

    def affected_nodes(self, region: tuple[float, float, float, float] | None) -> list[RRTXNode]:
        """Select nodes affected by an environment update.

        Inputs:
            region: Optional bounds as (xmin, xmax, ymin, ymax). None selects all nodes.
        Returns:
            Nodes inside the region padded by neighbor_radius, so endpoints of edges crossing the changed area are repaired.
        """
        if region is None:
            return list(self.nodes)
        padding = self.config.neighbor_radius
        xmin, xmax, ymin, ymax = region
        xmin -= padding
        xmax += padding
        ymin -= padding
        ymax += padding
        return [node for node in self.nodes if xmin <= node.point[0] <= xmax and ymin <= node.point[1] <= ymax]

    def transition_cost(self, start: Point, end: Point) -> float:
        """Compute edge traversal cost.

        Inputs:
            start: Edge start point.
            end: Edge end point.
        Returns:
            Euclidean edge length scaled by midpoint map/semantic cost.
        """
        distance = euclidean(start, end)
        midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
        return distance * (1.0 + max(0.0, self.cost_function(midpoint)))

    def get_path(self) -> list[Point]:
        """Read the current best path by following start parent links toward the goal-root.

        Returns:
            Ordered waypoint list from start to goal, or empty if disconnected.
        """
        if self.start is None or self.goal is None or self.start.parent is None or self.start.cost_to_come == math.inf:
            return []
        path = []
        node: RRTXNode | None = self.start
        seen = set()
        while node is not None and node not in seen:
            seen.add(node)
            path.append(node.point)
            if node is self.goal:
                return path
            node = node.parent
        return []

    def get_waypoints(self) -> list[Point]:
        """Return the current path using navigation terminology."""
        return self.get_path()


def euclidean(a: Point, b: Point) -> float:
    """Compute Euclidean distance between two 2D points.

    Inputs:
        a: First point.
        b: Second point.
    Returns:
        Straight-line distance between a and b.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def plan_rrtx(
    start: Point,
    goal: Point,
    bounds: tuple[float, float, float, float],
    costmap: np.ndarray | None = None,
    origin: Point = (0.0, 0.0),
    resolution: float = 1.0,
    config: RRTXConfig | None = None,
) -> list[Point]:
    """Convenience wrapper for one-shot planning.

    Inputs:
        start: Robot start point.
        goal: Goal point.
        bounds: Planning bounds as (xmin, xmax, ymin, ymax).
        costmap: Optional 2D grid used for costs and obstacles.
        origin: World coordinate of grid cell (0, 0).
        resolution: Meters per grid cell.
        config: Optional planner config. If None, a default config is created from bounds.
    Returns:
        Ordered waypoint list. Empty if no path is found.
    """
    planner_config = config or RRTXConfig(bounds=bounds)
    if costmap is None:
        planner = RRTXPlanner(planner_config)
    else:
        planner = RRTXPlanner.from_grid_costmap(planner_config, costmap, origin=origin, resolution=resolution)
    return planner.plan(start, goal)


__all__ = [
    "GridCostmap",
    "RRTXConfig",
    "RRTXNode",
    "RRTXPlanner",
    "plan_rrtx",
]
