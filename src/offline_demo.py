"""Planner-only offline sanity check on a deterministic synthetic map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from robot_state import Observation, Pose2D, RobotState
    from rrt_planner import PlannerResult, PlannerSettings, PlannerStatus, RRTPlanner
except ImportError:
    from .robot_state import Observation, Pose2D, RobotState
    from .rrt_planner import PlannerResult, PlannerSettings, PlannerStatus, RRTPlanner

# rrtx.py is intentionally loaded dynamically by the planner.
import importlib.util

_rrtx_spec = importlib.util.spec_from_file_location("rrtx", Path(__file__).with_name("rrtx.py"))
if _rrtx_spec is None or _rrtx_spec.loader is None:
    raise ImportError("Unable to load rrt-x.py")
_rrtx_module = importlib.util.module_from_spec(_rrtx_spec)
sys.modules[_rrtx_spec.name] = _rrtx_module
_rrtx_spec.loader.exec_module(_rrtx_module)
GridCostmap = _rrtx_module.GridCostmap


FRAME_ID = "map"
POSITION_UNITS = "meters"
TIME_UNITS = "seconds"
MAP_VERSION = "synthetic-detour-v1"
ORIGIN = (0.0, 0.0)
RESOLUTION_M = 1.0
LETHAL_COST = 255.0
SEED = 17
START = (3.0, 5.0)
GOAL = (26.0, 25.0)
GOAL_TOLERANCE_M = 0.75


def build_synthetic_costmap() -> np.ndarray:
    """Return a 30 m by 30 m map with a wall requiring an end detour."""
    costmap = np.zeros((30, 30), dtype=np.float64)
    costmap[15, 4:27] = LETHAL_COST # this would resemble water, for example
    costmap[8:12, 18:22] = 30.0
    return costmap


def build_demo_state() -> tuple[RobotState, np.ndarray]:
    """Build the future perception-to-planner handoff using a static map."""
    costmap = build_synthetic_costmap()
    observation = Observation(
        timestamp=0.0,
        top_down_costmap=costmap,
        class_names=["synthetic_traversable", "synthetic_obstacle"],
        metadata={"frame_id": FRAME_ID, "map_version": MAP_VERSION},
    )
    state = RobotState(
        current_pose=Pose2D(*START),
        goal_pose=Pose2D(*GOAL),
        current_observation=observation,
    )
    return state, costmap


def _validate_route(result: PlannerResult, costmap: np.ndarray) -> None:
    if result.route is None:
        return
    grid = GridCostmap(costmap, origin=ORIGIN, resolution=RESOLUTION_M, lethal_cost=LETHAL_COST)
    for point in result.route.waypoints:
        cell = grid.world_to_grid(point)
        if cell is None:
            raise ValueError(f"Route point is outside map bounds: {point}")
    for start, end in zip(result.route.waypoints[:-1], result.route.waypoints[1:]):
        if not grid.is_collision_free(start, end, resolution=0.1):
            raise ValueError(f"Route segment is not collision-free: {start} -> {end}")


def save_top_down_plot(costmap: np.ndarray, result: PlannerResult, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 7))
    image = axis.imshow(
        costmap,
        origin="lower",
        extent=(ORIGIN[0], ORIGIN[0] + costmap.shape[1] * RESOLUTION_M,
                ORIGIN[1], ORIGIN[1] + costmap.shape[0] * RESOLUTION_M),
        cmap="viridis",
        vmin=0.0,
        vmax=LETHAL_COST,
    )
    figure.colorbar(image, ax=axis, label="cost (0 traversable, 255 lethal)")
    lethal_rows, lethal_columns = np.where(costmap >= LETHAL_COST)
    axis.scatter(lethal_columns + 0.5, lethal_rows + 0.5, marker="s", color="black", label="lethal obstacle")
    axis.scatter(*START, marker="o", color="lime", edgecolors="black", label="start")
    axis.scatter(*GOAL, marker="*", s=150, color="red", edgecolors="black", label="goal")
    if result.route is not None:
        points = np.asarray(result.route.waypoints)
        axis.plot(points[:, 0], points[:, 1], color="white", linewidth=2.5, label="planned route")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(f"Offline planner sanity check: {result.status.value}")
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)


def run_demo(output_dir: str | Path = "src/output") -> tuple[RobotState, PlannerResult, Path, Path]:
    robot_state, costmap = build_demo_state()
    settings = PlannerSettings(
        bounds=(0.0, 30.0, 0.0, 30.0),
        origin=ORIGIN,
        resolution=RESOLUTION_M,
        lethal_cost=LETHAL_COST,
        step_size=1.0,
        neighbor_radius=2.5,
        goal_radius=0.75,
        collision_resolution=0.1,
        max_iterations=6000,
        seed=SEED,
    )
    result = RRTPlanner(settings).plan_result(
        robot_state,
        allow_fallback=False,
        frame_id=FRAME_ID,
        position_units=POSITION_UNITS,
        time_units=TIME_UNITS,
        map_version=MAP_VERSION,
        goal_tolerance_m=GOAL_TOLERANCE_M,
    )
    if result.route is not None:
        robot_state.update_route(result.route)
        _validate_route(result, costmap)
        route_trace = robot_state.get_current_route_trace()
    else:
        route_trace = None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trace_path = output_path / "offline_demo_trace.json"
    plot_path = output_path / "offline_demo_top_down.png"
    if route_trace is None:
        raise RuntimeError("successful offline demo requires a current route trace")
    trace_path.write_text(json.dumps(route_trace, indent=2) + "\n")
    save_top_down_plot(costmap, result, plot_path)

    route_consistent = result.route is not None and robot_state.current_route is result.route
    waypoints = route_trace["waypoints"]
    samples = route_trace["trajectory_samples"]
    print(f"RTA handoff: robot_state.get_current_route_trace() -> waypoints={len(waypoints)}, samples={len(samples)}")
    print(f"planner status: {result.status.value} ({result.message})")
    print(f"RRT-X succeeded: {result.status == PlannerStatus.SUCCESS}; fallback used: {result.fallback_used}")
    print(f"start: {START}; goal: {GOAL}")
    print(f"goal reached: {result.goal_reached}; final goal error: {result.goal_error_m} m")
    print(f"path length: {None if result.route is None else result.route.distance()} m")
    print(f"planning time: {result.planning_time_s:.6f} s; random seed: {result.seed}")
    print(f"map origin: {result.origin}; resolution: {result.resolution_m} m/cell; lethal cost: {result.lethal_cost}")
    print(f"frame ID: {result.frame_id}; position units: {result.position_units}; time units: {result.time_units}")
    print(f"map version: {result.map_version}; RobotState.route identity and data consistent: {route_consistent}")
    print(f"trace artifact: {trace_path}; plot artifact: {plot_path}")
    print("RTA evaluation not run: planner-only synthetic-map sanity check.")
    return robot_state, result, trace_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="src/output")
    args = parser.parse_args()
    run_demo(args.output_dir)


if __name__ == "__main__":
    main()