from __future__ import annotations

"""Planner integration layer used by the ROS node.

This module gives the rest of the project one stable planner API even though the
current low-level implementation lives in ``rrtx.py``.  The ROS glue code should
call ``RRTPlanner.plan(...)`` and receive a ``Route`` object without needing to
know how RRT-X is loaded, how costmaps are adapted, or whether a fallback was
used.
"""

from dataclasses import dataclass
from enum import Enum
import time
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from robot_state import Observation, Pose2D, RobotState, Route, route_from_waypoints
except ImportError:
    from .robot_state import Observation, Pose2D, RobotState, Route, route_from_waypoints


Point = tuple[float, float]


@dataclass
class PlannerSettings:
    """Configuration values for the integration planner.

    Inputs:
        bounds: Optional planning bounds as (xmin, xmax, ymin, ymax). If None,
            bounds are inferred from the costmap or start/goal locations.
        origin: World coordinate of costmap cell (0, 0).
        resolution: Meters per costmap cell.
        lethal_cost: Costmap value treated as an obstacle by RRT-X.
        max_iterations: Maximum RRT-X sampling iterations.
        step_size: Maximum distance added by one RRT-X tree extension.
        neighbor_radius: Radius used by RRT-X when searching nearby nodes.
        goal_radius: Distance threshold for connecting to the goal/start.
        collision_resolution: Distance between samples when checking an edge.
        nominal_speed: Speed used to timestamp route samples.
        fallback_samples: Number of waypoints in the straight-line fallback.
        seed: Optional random seed for repeatable plans.
    """

    bounds: tuple[float, float, float, float] | None = None
    origin: Point = (0.0, 0.0)
    resolution: float = 1.0
    lethal_cost: float = 255.0
    max_iterations: int = 1500
    step_size: float = 0.5
    neighbor_radius: float = 1.5
    goal_radius: float = 0.75
    collision_resolution: float = 0.1
    nominal_speed: float = 0.5
    fallback_samples: int = 20
    seed: int | None = None


class PlannerStatus(str, Enum):
    """Outcome categories for a planner invocation."""

    SUCCESS = "success"
    NO_PATH = "no_path"
    IMPORT_ERROR = "import_error"
    PLANNER_ERROR = "planner_error"
    INVALID_INPUT = "invalid_input"
    FALLBACK = "fallback"


@dataclass
class PlannerResult:
    """Typed planner outcome and reproducibility metadata."""

    route: Route | None
    status: PlannerStatus
    message: str
    planning_time_s: float
    seed: int | None
    goal_tolerance_m: float
    goal_error_m: float | None
    goal_reached: bool
    frame_id: str
    position_units: str
    time_units: str
    map_version: str | None
    origin: Point
    resolution_m: float
    lethal_cost: float
    fallback_used: bool = False


class RRTPlanner:
    """High-level planner that converts RobotState into a Route.

    This class is intentionally an adapter.  It does not store robot state and it
    does not own perception.  It reads the current state, calls the lower-level
    RRT-X implementation when possible, and returns a Route object for the rest
    of the system to execute or monitor.
    """

    def __init__(self, settings: PlannerSettings | None = None):
        """Create the planner adapter.

        Inputs:
            settings: Optional planner configuration. Defaults are used when no
                settings are provided.
        """
        self.settings = settings or PlannerSettings()
        self._rrtx_module = None

    def plan(
        self,
        robot_state: RobotState,
        observation: Observation | None = None,
        cost_adjustments: dict[str, Any] | np.ndarray | None = None,
    ) -> Route:
        """Plan a route from the robot's current pose to its current goal.

        Inputs:
            robot_state: Shared state object containing current pose, goal, and
                optionally the latest observation.
            observation: Optional perception snapshot. If None, the planner uses
                robot_state.current_observation.
            cost_adjustments: Optional extra costmap information. A numpy array
                is treated as a full costmap. A dictionary can contain "costmap",
                "scale", or "offset" entries to modify the base costmap.
        Returns:
            Route containing waypoints, trajectory poses, cost metadata, and
            per-sample time/speed/cost entries in route.metadata["samples"].
        Raises:
            ValueError: If robot_state does not contain a current pose or goal.
        """
        result = self.plan_result(robot_state, observation, cost_adjustments, allow_fallback=True)
        if result.route is None:
            raise ValueError(result.message)
        return result.route

    def plan_result(
        self,
        robot_state: RobotState,
        observation: Observation | None = None,
        cost_adjustments: dict[str, Any] | np.ndarray | None = None,
        *,
        allow_fallback: bool = False,
        frame_id: str = "map",
        position_units: str = "meters",
        time_units: str = "seconds",
        map_version: str | None = None,
        goal_tolerance_m: float = 0.75,
    ) -> PlannerResult:
        """Plan and return a typed result without hiding planner failures."""
        start_time = time.perf_counter()
        try:
            start = robot_state.planner_start()
            goal = robot_state.planner_goal()
        except ValueError as error:
            return PlannerResult(
                None, PlannerStatus.INVALID_INPUT, str(error), time.perf_counter() - start_time,
                self.settings.seed, goal_tolerance_m, None, False, frame_id, position_units, time_units,
                map_version, self.settings.origin, self.settings.resolution, self.settings.lethal_cost,
            )

        observation = observation or robot_state.current_observation
        costmap = self._build_costmap(robot_state, observation, cost_adjustments)
        bounds = self.settings.bounds or self._infer_bounds(start, goal, costmap)
        status, waypoints, message = self._plan_with_rrtx_result(start, goal, bounds, costmap)
        fallback_used = False
        planner_name = "rrtx"
        if not waypoints and allow_fallback and status in {PlannerStatus.NO_PATH, PlannerStatus.IMPORT_ERROR, PlannerStatus.PLANNER_ERROR}:
            waypoints = self._straight_line(start, goal, self.settings.fallback_samples)
            status = PlannerStatus.FALLBACK
            message = "RRT-X did not produce a route; returned explicit straight-line fallback"
            planner_name = "straight_line_fallback"
            fallback_used = True

        waypoints = self._remove_consecutive_duplicates(waypoints)

        route = None
        goal_error = None
        goal_reached = False
        if waypoints:
            route = route_from_waypoints(waypoints, costs={"distance": self._path_distance(waypoints)})
            route.trajectory = [Pose2D(x, y) for x, y in route.waypoints]
            goal_error = math.hypot(waypoints[-1][0] - goal[0], waypoints[-1][1] - goal[1])
            goal_reached = goal_error <= goal_tolerance_m
            route.metadata.update(
                {
                    "planner": planner_name,
                    "samples": self._trajectory_samples(route.waypoints),
                    "bounds": bounds,
                    "used_costmap": costmap is not None,
                    "frame_id": frame_id,
                    "position_units": position_units,
                    "time_units": time_units,
                    "map_version": map_version,
                    "origin_m": list(self.settings.origin),
                    "resolution_m": self.settings.resolution,
                    "lethal_cost": self.settings.lethal_cost,
                    "seed": self.settings.seed,
                }
            )
        elif status == PlannerStatus.SUCCESS:
            status = PlannerStatus.NO_PATH
            message = "RRT-X returned no usable waypoints"

        return PlannerResult(
            route, status, message, time.perf_counter() - start_time,
            self.settings.seed, goal_tolerance_m, goal_error, goal_reached, frame_id,
            position_units, time_units, map_version, self.settings.origin,
            self.settings.resolution, self.settings.lethal_cost, fallback_used,
        )

    def _plan_with_rrtx(
        self,
        start: Point,
        goal: Point,
        bounds: tuple[float, float, float, float],
        costmap: np.ndarray | None,
    ) -> list[Point]:
        """Call the existing RRT-X implementation if it can be loaded.

        Inputs:
            start: Robot start point as (x, y).
            goal: Goal point as (x, y).
            bounds: Planning bounds as (xmin, xmax, ymin, ymax).
            costmap: Optional 2D traversal-cost grid.
        Returns:
            Ordered waypoint list from start to goal, or an empty list if RRT-X
            cannot load, fails, or does not find a path.
        """
        _, waypoints, _ = self._plan_with_rrtx_result(start, goal, bounds, costmap)
        return waypoints

    def _plan_with_rrtx_result(
        self,
        start: Point,
        goal: Point,
        bounds: tuple[float, float, float, float],
        costmap: np.ndarray | None,
    ) -> tuple[PlannerStatus, list[Point], str]:
        module = self._load_rrtx()
        if module is None:
            return PlannerStatus.IMPORT_ERROR, [], "Unable to import rrtx.py"
        config = module.RRTXConfig(
            bounds=bounds,
            step_size=self.settings.step_size,
            neighbor_radius=self.settings.neighbor_radius,
            goal_radius=self.settings.goal_radius,
            max_iterations=self.settings.max_iterations,
            collision_resolution=self.settings.collision_resolution,
            seed=self.settings.seed,
        )
        try:
            waypoints = module.plan_rrtx(
                start,
                goal,
                bounds,
                costmap=costmap,
                origin=self.settings.origin,
                resolution=self.settings.resolution,
                config=config,
            )
        except Exception as error:
            return PlannerStatus.PLANNER_ERROR, [], f"RRT-X raised {type(error).__name__}: {error}"
        if not waypoints:
            return PlannerStatus.NO_PATH, [], "RRT-X returned no path"
        return PlannerStatus.SUCCESS, waypoints, "RRT-X returned a route"

    def _load_rrtx(self):
        """Load ``rrtx.py``

        Returns:
            Imported module object with RRTXConfig and plan_rrtx, or None if the
            file cannot be imported.
        """
        if self._rrtx_module is not None:
            return self._rrtx_module
        path = Path(__file__).with_name("rrtx.py")
        spec = importlib.util.spec_from_file_location("rrtx", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None
        self._rrtx_module = module
        return module

    def _build_costmap(
        self,
        robot_state: RobotState,
        observation: Observation | None,
        cost_adjustments: dict[str, Any] | np.ndarray | None,
    ) -> np.ndarray | None:
        """Choose and optionally adjust the costmap used by the planner.

        Inputs:
            robot_state: Shared state object that may contain a current
                observation and planner costmap.
            observation: Optional perception snapshot to prefer over state-held
                perception data.
            cost_adjustments: Optional numpy costmap or dictionary containing
                extra costmap, scale, or offset values.
        Returns:
            Floating-point costmap clipped to [0, lethal_cost], or None when no
            costmap is available.
        """
        base_costmap = None
        if observation is not None:
            base_costmap = observation.top_down_costmap if observation.top_down_costmap is not None else observation.image_costmap
        if base_costmap is None:
            base_costmap = robot_state.planner_costmap()
        if base_costmap is None and isinstance(cost_adjustments, np.ndarray):
            base_costmap = cost_adjustments
        if base_costmap is None:
            return None
        costmap = np.asarray(base_costmap, dtype=float).copy()
        if isinstance(cost_adjustments, dict):
            extra = cost_adjustments.get("costmap")
            if extra is not None:
                costmap = costmap + np.asarray(extra, dtype=float)
            scale = cost_adjustments.get("scale")
            if scale is not None:
                costmap = costmap * float(scale)
            offset = cost_adjustments.get("offset")
            if offset is not None:
                costmap = costmap + float(offset)
        return np.clip(costmap, 0.0, self.settings.lethal_cost)

    def _infer_bounds(self, start: Point, goal: Point, costmap: np.ndarray | None) -> tuple[float, float, float, float]:
        """Infer planner bounds when none are explicitly configured.

        Inputs:
            start: Robot start point as (x, y).
            goal: Goal point as (x, y).
            costmap: Optional 2D grid whose shape can define map bounds.
        Returns:
            Bounds as (xmin, xmax, ymin, ymax).
        """
        if costmap is not None:
            height, width = costmap.shape[:2]
            xmin = self.settings.origin[0]
            ymin = self.settings.origin[1]
            xmax = xmin + width * self.settings.resolution
            ymax = ymin + height * self.settings.resolution
            return xmin, xmax, ymin, ymax
        margin = max(2.0, self.settings.step_size * 4.0)
        return (
            min(start[0], goal[0]) - margin,
            max(start[0], goal[0]) + margin,
            min(start[1], goal[1]) - margin,
            max(start[1], goal[1]) + margin,
        )

    def _straight_line(self, start: Point, goal: Point, samples: int) -> list[Point]:
        """Create a deterministic direct path used when RRT-X fails.

        Inputs:
            start: Robot start point as (x, y).
            goal: Goal point as (x, y).
            samples: Number of evenly spaced waypoints to return.
        Returns:
            List of waypoints including start and goal.
        """
        count = max(2, samples)
        return [
            (start[0] + (goal[0] - start[0]) * i / (count - 1), start[1] + (goal[1] - start[1]) * i / (count - 1))
            for i in range(count)
        ]

    def _trajectory_samples(self, waypoints: list[Point]) -> list[dict[str, float]]:
        """Convert route waypoints into time-expanded samples.

        Inputs:
            waypoints: Ordered 2D route points.
        Returns:
            List of dictionaries containing index, x, y, time, speed, and cost.
        """
        samples = []
        elapsed = 0.0
        previous = None
        for index, point in enumerate(waypoints):
            if previous is not None:
                segment_time = math.hypot(point[0] - previous[0], point[1] - previous[1]) / max(self.settings.nominal_speed, 1e-6)
                elapsed += segment_time
            samples.append(
                {
                    "index": float(index),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "time": float(elapsed),
                    "speed": float(self.settings.nominal_speed),
                    "cost": 0.0,
                }
            )
            previous = point
        return samples

    @staticmethod
    def _remove_consecutive_duplicates(waypoints: list[Point]) -> list[Point]:
        """Remove adjacent identical points before route timing and serialization."""
        if not waypoints:
            return []
        normalized = [waypoints[0]]
        for point in waypoints[1:]:
            if point != normalized[-1]:
                normalized.append(point)
        return normalized

    def _path_distance(self, waypoints: list[Point]) -> float:
        """Compute total length of a waypoint path.

        Inputs:
            waypoints: Ordered 2D route points.
        Returns:
            Sum of Euclidean distances between consecutive waypoints.
        """
        if len(waypoints) < 2:
            return 0.0
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(waypoints[:-1], waypoints[1:]))


def plan(
    robot_state: RobotState,
    observation: Observation | None = None,
    cost_adjustments: dict[str, Any] | np.ndarray | None = None,
    settings: PlannerSettings | None = None,
) -> Route:
    """One-shot convenience wrapper around RRTPlanner.

    Inputs:
        robot_state: Shared state object containing current pose and goal.
        observation: Optional perception snapshot.
        cost_adjustments: Optional costmap or costmap modifiers.
        settings: Optional planner configuration.
    Returns:
        Planned Route from the robot's current pose to its current goal.
    """
    return RRTPlanner(settings).plan(robot_state, observation, cost_adjustments)


__all__ = ["PlannerResult", "PlannerSettings", "PlannerStatus", "RRTPlanner", "plan"]
