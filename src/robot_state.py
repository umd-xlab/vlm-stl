from __future__ import annotations

"""Shared state and data contracts for navigation integration.

This is a PASSIVE DATA OBJECTS only file! Do not call ROS, RRT-X, or RTA monitors directly. The ROS planning node should update these objects instead.
"""

from dataclasses import dataclass, field
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


Point2D = tuple[float, float]
"""Planar point represented as (x, y)."""

Waypoint = tuple[float, float]
"""Route waypoint represented as (x, y)."""


@dataclass
class Pose2D:
    """2D robot or goal pose.

    Inputs:
        x: Position along the world/map x-axis.
        y: Position along the world/map y-axis.
        theta: Heading angle in radians.
    """

    x: float
    y: float
    theta: float = 0.0

    def distance_to(self, other: Pose2D | Point2D) -> float:
        """Compute planar distance to another pose or point.

        Inputs:
            other: Pose2D or (x, y) point to measure against.
        Returns:
            Euclidean distance in the same units as x/y.
        """
        if isinstance(other, Pose2D):
            return math.hypot(self.x - other.x, self.y - other.y)
        return math.hypot(self.x - other[0], self.y - other[1])

    def as_point(self) -> Point2D:
        """Return only the translational component.

        Returns:
            (x, y) point for planners that ignore heading.
        """
        return self.x, self.y

    def as_array(self) -> np.ndarray:
        """Return pose as a numpy vector.

        Returns:
            Array containing [x, y, theta].
        """
        return np.array([self.x, self.y, self.theta], dtype=float)


@dataclass
class Velocity2D:
    """2D velocity command or odometry estimate.

    Inputs:
        linear: Forward linear velocity.
        angular: Yaw angular velocity.
    """

    linear: float = 0.0
    angular: float = 0.0

    @property
    def speed(self) -> float:
        """Return nonnegative translational speed.

        Returns:
            Absolute value of linear velocity.
        """
        return abs(self.linear)


@dataclass
class Observation:
    """Perception snapshot from camera/depth/segmentation processing.

    Inputs:
        timestamp: Observation time (seconds)
        point_cloud: dense/sparse 3D point cloud. (Optional)
        image_costmap: image-frame costmap. (Optional)
        top_down_costmap: robot/world-frame costmap for planning. (optional)
        top_down_semantics: projected semantic labels or one-hot map. (optional)
        class_names: Semantic class labels used by perception.
        class_distances: Minimum distances to semantic classes.
        metadata: Extra perception data that does not fit fixed fields.
    """

    timestamp: float
    point_cloud: np.ndarray | None = None
    image_costmap: np.ndarray | None = None
    top_down_costmap: np.ndarray | None = None
    top_down_semantics: np.ndarray | None = None
    class_names: list[str] = field(default_factory=list)
    class_distances: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def min_distance_to_class(self, class_name: str, default: float = math.inf) -> float:
        """Look up the closest known distance to a semantic class.

        Inputs:
            class_name: Semantic class name, such as "water" or "building".
            default: Value returned if class_name is unavailable.
        Returns:
            Minimum distance to that class.
        """
        return float(self.class_distances.get(class_name, default))

    def surface_score(self, safe_classes: set[str] | None = None, unsafe_classes: set[str] | None = None) -> float:
        """Convert semantic distances into an STL-friendly surface score.

        Inputs:
            safe_classes: Classes that increase the score when nearby.
            unsafe_classes: Classes that decrease the score when nearby.
        Returns:
            Positive values favor safe surfaces; negative values indicate unsafe surfaces.
        """
        safe_classes = safe_classes or {"road", "sidewalk", "pavement", "path"}
        unsafe_classes = unsafe_classes or {"water", "grass", "building", "stop sign", "obstacle"}
        if not self.class_distances:
            return 0.0
        score = 0.0
        for class_name, distance in self.class_distances.items():
            normalized = class_name.lower().strip()
            contribution = 1.0 / max(float(distance), 1e-6)
            if normalized in safe_classes:
                score += contribution
            elif normalized in unsafe_classes:
                score -= contribution
        return score


@dataclass
class Route:
    """Candidate or accepted robot route.

    Inputs:
        waypoints: Ordered 2D waypoints, usually from the planner.
        trajectory: Optional time/state-expanded route as poses.
        costs: Named cost components used to compare routes.
        generated_at: Wall-clock time when this route was produced.
        metadata: Extra route/planner data that does not fit fixed fields.
    """

    waypoints: list[Waypoint] = field(default_factory=list)
    trajectory: list[Pose2D] = field(default_factory=list)
    costs: dict[str, float] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Check whether route has no usable geometry.

        Returns:
            True if both waypoints and trajectory are empty.
        """
        return not self.waypoints and not self.trajectory

    def first_waypoint(self) -> Waypoint | None:
        """Return the next waypoint for a controller.

        Returns:
            First waypoint, or None if no waypoint exists.
        """
        return self.waypoints[0] if self.waypoints else None

    def distance(self) -> float:
        """Compute total route length.

        Returns:
            Sum of straight-line distances between consecutive route points.
        """
        points: list[Point2D]
        if self.trajectory:
            points = [pose.as_point() for pose in self.trajectory]
        else:
            points = list(self.waypoints)
        if len(points) < 2:
            return 0.0
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points[:-1], points[1:]))

    def estimated_duration(self, nominal_speed: float) -> float:
        """Estimate traversal time for the route.

        Inputs:
            nominal_speed: Assumed positive translational speed.
        Returns:
            Route length divided by nominal_speed, or infinity if speed is invalid.
        """
        if nominal_speed <= 0.0:
            return math.inf
        return self.distance() / nominal_speed


@dataclass
class RTASignals:
    """Signal values consumed by the current CSV-based RTA monitors.

    Inputs:
        time: Signal timestamp in seconds.
        spd: Robot speed signal.
        dst: Distance-to-goal or distance-to-hazard signal.
        srf: Surface safety score signal.
        sdst: Stop-sign/obstacle distance signal.
    """

    time: float
    spd: float = math.nan
    dst: float = math.nan
    srf: float = math.nan
    sdst: float = math.nan

    def as_dict(self) -> dict[str, float]:
        """Convert signals to named values.

        Returns:
            Dictionary with keys matching the RTA feed CSV columns.
        """
        return {
            "time": self.time,
            "spd": self.spd,
            "dst": self.dst,
            "srf": self.srf,
            "sdst": self.sdst,
        }

    def as_csv_row(self) -> list[float]:
        """Convert signals to ordered CSV row values.

        Returns:
            Values ordered as time, spd, dst, srf, sdst.
        """
        values = self.as_dict()
        return [values["time"], values["spd"], values["dst"], values["srf"], values["sdst"]]


@dataclass
class RobotState:
    """Central state object shared by perception, planning, RTA, and control.

    Inputs:
        current_pose: Latest robot pose.
        previous_pose: Previous robot pose.
        current_velocity: Latest robot velocity.
        previous_velocity: Previous robot velocity.
        goal_pose: Active goal pose.
        current_observation: Latest perception observation.
        previous_observation: Previous perception observation.
        current_route: Latest planned/accepted route.
        previous_route: Previous route.
        pose_history: Recent pose history.
        observation_history: Recent observation history.
        route_history: Recent route history.
        rta_history: Recent RTA signal history.
        pose_time_history: Timestamps associated with pose_history.
        max_history: Maximum entries kept per history buffer.
        start_time: Wall-clock time used for elapsed-time signals.
    """

    current_pose: Pose2D | None = None
    previous_pose: Pose2D | None = None
    current_velocity: Velocity2D = field(default_factory=Velocity2D)
    previous_velocity: Velocity2D | None = None
    goal_pose: Pose2D | None = None
    current_observation: Observation | None = None
    previous_observation: Observation | None = None
    current_route: Route | None = None
    previous_route: Route | None = None
    pose_history: list[Pose2D] = field(default_factory=list)
    observation_history: list[Observation] = field(default_factory=list)
    route_history: list[Route] = field(default_factory=list)
    rta_history: list[RTASignals] = field(default_factory=list)
    pose_time_history: list[float] = field(default_factory=list)
    max_history: int = 100
    start_time: float = field(default_factory=time.time)

    def update_pose(self, pose: Pose2D, timestamp: float | None = None) -> None:
        """Update current/previous pose and estimate speed from pose delta.

        This is usually called from the odometry callback in planning_node.py.
        It stores both the latest pose and enough history to estimate linear
        speed when an explicit velocity estimate is not available.

        Inputs:
            pose: New robot pose.
            timestamp: Optional timestamp in seconds. If None, elapsed_time() is used.
        """
        timestamp = self.elapsed_time() if timestamp is None else timestamp
        self.previous_pose = self.current_pose
        previous_timestamp = self.pose_time_history[-1] if self.pose_time_history else None
        self.current_pose = pose
        self.pose_history.append(pose)
        self.pose_time_history.append(timestamp)
        self._trim_history(self.pose_history)
        self._trim_history(self.pose_time_history)
        if previous_timestamp is not None and self.previous_pose is not None:
            dt = timestamp - previous_timestamp
            if dt > 0.0:
                self.current_velocity.linear = self.previous_pose.distance_to(pose) / dt

    def update_velocity(self, velocity: Velocity2D) -> None:
        """Update current/previous velocity.

        Inputs:
            velocity: New velocity estimate or command.
        """
        self.previous_velocity = self.current_velocity
        self.current_velocity = velocity

    def update_goal(self, goal_pose: Pose2D) -> None:
        """Set the active navigation goal.

        Inputs:
            goal_pose: Desired goal pose.
        """
        self.goal_pose = goal_pose

    def update_observation(self, observation: Observation) -> None:
        """Update current/previous perception observation.

        This is the handoff point from the perception team into the integration
        layer.  The observation may contain raw image-frame products, top-down
        planning products, semantic distances, or any combination of those.

        Inputs:
            observation: New perception snapshot.
        """
        self.previous_observation = self.current_observation
        self.current_observation = observation
        self.observation_history.append(observation)
        self._trim_history(self.observation_history)

    def update_route(self, route: Route) -> None:
        """Update current/previous planned route.

        This is the handoff point from the planner back into shared state.  RTA
        pre-checks, live monitors, and controllers can all inspect the same
        current_route after this method is called.

        Inputs:
            route: New planner output or accepted route.
        """
        self.previous_route = self.current_route
        self.current_route = route
        self.route_history.append(route)
        self._trim_history(self.route_history)

    def get_current_route_trace(self) -> dict[str, Any]:
        """Return the validated, JSON-compatible current route handoff."""
        if self.current_route is None:
            raise ValueError("current_route is required before requesting a route trace")

        route = self.current_route
        metadata = route.metadata
        if len(route.waypoints) != len(route.trajectory):
            raise ValueError("route waypoints and trajectory lengths must match")

        waypoints = []
        for index, (waypoint, pose) in enumerate(zip(route.waypoints, route.trajectory)):
            if len(waypoint) != 2 or not all(math.isfinite(float(value)) for value in waypoint):
                raise ValueError(f"route waypoint {index} must be a finite 2D point")
            if not all(math.isfinite(float(value)) for value in (pose.x, pose.y, pose.theta)):
                raise ValueError(f"route trajectory pose {index} must be finite")
            if not math.isclose(float(waypoint[0]), float(pose.x)) or not math.isclose(float(waypoint[1]), float(pose.y)):
                raise ValueError(f"route trajectory pose {index} does not match its waypoint")
            waypoints.append({"x": float(waypoint[0]), "y": float(waypoint[1])})

        raw_samples = metadata.get("samples")
        if raw_samples is None:
            raise ValueError("route metadata must contain timed samples")
        if len(raw_samples) != len(waypoints):
            raise ValueError("route timed samples and waypoints lengths must match")

        samples = []
        previous_time = None
        for index, (sample, waypoint) in enumerate(zip(raw_samples, waypoints)):
            try:
                x = float(sample["x"])
                y = float(sample["y"])
                time_s = float(sample["time"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"route sample {index} is missing x, y, or time") from error
            if not all(math.isfinite(value) for value in (x, y, time_s)):
                raise ValueError(f"route sample {index} must contain finite values")
            if previous_time is not None and time_s <= previous_time:
                raise ValueError("route sample timestamps must be strictly increasing")
            if not math.isclose(x, waypoint["x"]) or not math.isclose(y, waypoint["y"]):
                raise ValueError(f"route sample {index} does not match its waypoint")
            if samples and x == samples[-1]["x"] and y == samples[-1]["y"]:
                raise ValueError("route contains adjacent duplicate timed samples without a dwell event")
            samples.append({"waypoint_index": index, "x": x, "y": y, "time_s": time_s})
            previous_time = time_s

        origin = metadata.get("origin_m")
        if origin is not None:
            if len(origin) != 2 or not all(math.isfinite(float(value)) for value in origin):
                raise ValueError("route origin_m must be a finite 2D point")
            origin = [float(value) for value in origin]

        trace = {
            "frame_id": metadata.get("frame_id"),
            "position_units": metadata.get("position_units"),
            "time_units": metadata.get("time_units"),
            "map_version": metadata.get("map_version"),
            "origin_m": origin,
            "resolution_m": metadata.get("resolution_m"),
            "lethal_cost": metadata.get("lethal_cost"),
            "waypoints": waypoints,
            "trajectory_samples": samples,
        }
        for key in ("resolution_m", "lethal_cost"):
            if trace[key] is not None:
                value = float(trace[key])
                if not math.isfinite(value):
                    raise ValueError(f"route {key} must be finite")
                trace[key] = value
        return trace

    def elapsed_time(self) -> float:
        """Compute elapsed wall-clock time since RobotState creation.

        Returns:
            Seconds since start_time.
        """
        return time.time() - self.start_time

    def distance_to_goal(self) -> float:
        """Compute current distance to active goal.

        Returns:
            Euclidean distance to goal, or NaN if pose/goal is unavailable.
        """
        if self.current_pose is None or self.goal_pose is None:
            return math.nan
        return self.current_pose.distance_to(self.goal_pose)

    def distance_to_nearest_obstacle(self) -> float:
        """Estimate nearest obstacle-like semantic distance.

        Returns:
            Minimum distance among obstacle/building/stop sign/water, or NaN if unavailable.
        """
        if self.current_observation is None:
            return math.nan
        obstacle_names = ["obstacle", "building", "stop sign", "water"]
        distances = [self.current_observation.min_distance_to_class(name) for name in obstacle_names]
        finite_distances = [distance for distance in distances if math.isfinite(distance)]
        if not finite_distances:
            return math.nan
        return min(finite_distances)

    def surface_score(self) -> float:
        """Compute current semantic surface safety score.

        Returns:
            Surface score from current observation, or NaN if unavailable.
        """
        if self.current_observation is None:
            return math.nan
        return self.current_observation.surface_score()

    def to_rta_signals(self, timestamp: float | None = None) -> RTASignals:
        """Convert current robot state into RTA monitor signals.

        Inputs:
            timestamp: Optional RTA timestamp. If None, elapsed_time() is used.
        Returns:
            RTASignals object with spd, dst, srf, and sdst populated.
        """
        signals = RTASignals(
            time=self.elapsed_time() if timestamp is None else timestamp,
            spd=self.current_velocity.speed,
            dst=self.distance_to_goal(),
            srf=self.surface_score(),
            sdst=self.distance_to_nearest_obstacle(),
        )
        self.rta_history.append(signals)
        self._trim_history(self.rta_history)
        return signals

    def append_rta_feed(self, feed_path: str | Path, timestamp: float | None = None) -> RTASignals:
        """Append current RTA signals to a CSV feed file.

        This method is the integration layer's current contract with the RTA
        monitors.  It writes the columns time, spd, dst, srf, and sdst without
        requiring RobotState to know how the monitor processes those values.

        Inputs:
            feed_path: Path to RTA feed CSV.
            timestamp: Optional RTA timestamp. If None, elapsed_time() is used.
        Returns:
            Signals that were written to the feed.
        """
        feed_path = Path(feed_path)
        signals = self.to_rta_signals(timestamp=timestamp)
        file_exists = feed_path.exists()
        with feed_path.open("a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists or feed_path.stat().st_size == 0:
                writer.writerow(["time", "spd", "dst", "srf", "sdst"])
            writer.writerow(signals.as_csv_row())
        return signals

    def planner_start(self) -> Point2D:
        """Return current pose in planner point format.

        Returns:
            (x, y) start point for the planner.
        Raises:
            ValueError: If current_pose is not set.
        """
        if self.current_pose is None:
            raise ValueError("current_pose is required before planning")
        return self.current_pose.as_point()

    def planner_goal(self) -> Point2D:
        """Return goal pose in planner point format.

        Returns:
            (x, y) goal point for the planner.
        Raises:
            ValueError: If goal_pose is not set.
        """
        if self.goal_pose is None:
            raise ValueError("goal_pose is required before planning")
        return self.goal_pose.as_point()

    def planner_costmap(self) -> np.ndarray | None:
        """Return the best available costmap for planning.

        Returns:
            top_down_costmap if present, otherwise image_costmap, otherwise None.
        """
        if self.current_observation is None:
            return None
        if self.current_observation.top_down_costmap is not None:
            return self.current_observation.top_down_costmap
        return self.current_observation.image_costmap

    def snapshot(self) -> dict[str, Any]:
        """Create a shallow debug/status snapshot of important state fields.

        Returns:
            Dictionary containing current pose, velocity, goal, observation, route, and latest RTA signal.
        """
        return {
            "current_pose": self.current_pose,
            "previous_pose": self.previous_pose,
            "current_velocity": self.current_velocity,
            "goal_pose": self.goal_pose,
            "current_observation": self.current_observation,
            "current_route": self.current_route,
            "latest_rta": self.rta_history[-1] if self.rta_history else None,
        }

    def _trim_history(self, history: list[Any]) -> None:
        """Limit a history list to max_history entries.

        Inputs:
            history: Mutable history list to trim in place.
        """
        if self.max_history <= 0:
            history.clear()
            return
        del history[:-self.max_history]


def route_from_waypoints(waypoints: list[Waypoint] | np.ndarray, costs: dict[str, float] | None = None) -> Route:
    """Create a Route object from planner waypoints.

    Inputs:
        waypoints: Iterable/array of points shaped like [(x, y), ...].
        costs: Optional named planner cost components.
    Returns:
        Route with normalized float waypoints.
    """
    waypoint_list = [(float(point[0]), float(point[1])) for point in waypoints]
    return Route(waypoints=waypoint_list, costs=costs or {})


def observation_from_perception(
    timestamp: float,
    perception_module: Any,
    class_distances: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Observation:
    """Create an Observation from the existing PerceptionModule API.

    Inputs:
        timestamp: Observation timestamp in seconds.
        perception_module: Object with perception outputs like point_cloud and image_costmap.
        class_distances: Optional semantic distance dictionary.
        metadata: Optional extra observation metadata.
    Returns:
        Observation populated from available perception fields/methods.
    """
    point_cloud = getattr(perception_module, "point_cloud", None)
    image_costmap = getattr(perception_module, "image_costmap", None)
    environment_state = getattr(perception_module, "environment_state", None)
    prompts = getattr(perception_module, "prompts", [])
    top_down_costmap = None
    top_down_semantics = None
    if hasattr(perception_module, "get_top_down_costmap"):
        try:
            top_down_costmap = perception_module.get_top_down_costmap()
        except Exception:
            top_down_costmap = None
    if hasattr(perception_module, "get_top_down_environment_state"):
        try:
            top_down_semantics = perception_module.get_top_down_environment_state()
        except Exception:
            top_down_semantics = None
    return Observation(
        timestamp=timestamp,
        point_cloud=point_cloud,
        image_costmap=image_costmap,
        top_down_costmap=top_down_costmap,
        top_down_semantics=top_down_semantics if top_down_semantics is not None else environment_state,
        class_names=list(prompts),
        class_distances=class_distances or {},
        metadata=metadata or {},
    )


__all__ = [
    "Observation",
    "Pose2D",
    "RTASignals",
    "RobotState",
    "Route",
    "Velocity2D",
    "observation_from_perception",
    "route_from_waypoints",
]
