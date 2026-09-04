import numpy as np

from src.offline_demo import (
    FRAME_ID,
    GOAL,
    LETHAL_COST,
    MAP_VERSION,
    ORIGIN,
    RESOLUTION_M,
    SEED,
    START,
    build_demo_state,
    build_synthetic_costmap,
    GridCostmap,
)
from src.rrt_planner import PlannerSettings, PlannerStatus, RRTPlanner


def planner_settings(seed=SEED):
    return PlannerSettings(
        bounds=(0.0, 30.0, 0.0, 30.0),
        origin=ORIGIN,
        resolution=RESOLUTION_M,
        lethal_cost=LETHAL_COST,
        step_size=1.0,
        neighbor_radius=2.5,
        goal_radius=0.75,
        collision_resolution=0.1,
        max_iterations=6000,
        seed=seed,
    )


def test_detour_route_is_valid_and_metadata_is_explicit():
    state, costmap = build_demo_state()
    result = RRTPlanner(planner_settings()).plan_result(
        state,
        allow_fallback=False,
        frame_id=FRAME_ID,
        position_units="meters",
        time_units="seconds",
        map_version=MAP_VERSION,
    )

    # Check that the planner succeeded and the route is valid
    assert result.status is PlannerStatus.SUCCESS

    # Check that the resulting route is not None
    assert result.route is not None

    # Check that the first waypoint is equal to the provided start point
    assert result.route.waypoints[0] == START

    # Check that the last waypoint is equal to the provided goal point but also within the tolerance
    assert np.allclose(result.route.waypoints[-1], GOAL, atol=result.goal_tolerance_m)

    # Check that all of our metadata gets saved correctly
    assert result.route.metadata["seed"] == SEED
    assert result.route.metadata["frame_id"] == FRAME_ID
    assert result.route.metadata["position_units"] == "meters"
    assert result.route.metadata["time_units"] == "seconds"
    assert result.route.metadata["map_version"] == MAP_VERSION

    # Check that the route distance is greater than the straight-line distance between start and goal
    assert result.route.distance() > np.linalg.norm(np.subtract(GOAL, START))


    # Check that the route does not contain any collisions with the costmap and that all waypoints are within the bounds of the costmap
    grid = GridCostmap(costmap, origin=ORIGIN, resolution=RESOLUTION_M, lethal_cost=LETHAL_COST)
    for point in result.route.waypoints:
        assert grid.world_to_grid(point) is not None
    assert all(
        grid.is_collision_free(start, end, resolution=0.1)
        for start, end in zip(result.route.waypoints[:-1], result.route.waypoints[1:])
    )


def test_fixed_seed_reproduces_route_geometry():
    """Tests that using the same random seed produces the same route geometry."""
    first_state, _ = build_demo_state()
    second_state, _ = build_demo_state()
    first = RRTPlanner(planner_settings()).plan_result(first_state, allow_fallback=False)
    second = RRTPlanner(planner_settings()).plan_result(second_state, allow_fallback=False)


    assert first.status is second.status is PlannerStatus.SUCCESS
    assert first.route is not None and second.route is not None
    assert first.route.waypoints == second.route.waypoints


def test_blocked_map_returns_typed_no_path_without_fallback():
    """Generating a fully blocked map with all lethal cost obstacles should
    return a NO_PATH result without falling back to a straight line."""
    state, _ = build_demo_state()
    blocked = np.full((30, 30), LETHAL_COST, dtype=np.float64)
    state.current_observation.top_down_costmap = blocked
    result = RRTPlanner(planner_settings()).plan_result(state, allow_fallback=False)

    assert result.status is PlannerStatus.NO_PATH
    assert result.route is None
    assert not result.fallback_used
    assert "no path" in result.message.lower()


def test_time_parameterized_samples_are_strictly_increasing():
    """The time parameterization of the samples in the route should be strictly increasing."""
    state, _ = build_demo_state()
    result = RRTPlanner(planner_settings()).plan_result(state, allow_fallback=False)

    assert result.route is not None
    samples = result.route.metadata["samples"]
    times = [sample["time"] for sample in samples]
    assert all(current > previous for previous, current in zip(times, times[1:]))
    assert all(
        (current["x"], current["y"]) != (previous["x"], previous["y"])
        for previous, current in zip(samples, samples[1:])
    )
    assert all({"x", "y", "time", "speed", "cost"} <= sample.keys() for sample in samples)


def test_path_length_matches_route_geometry():
    """Test that the path length matches the route geometry."""
    state, _ = build_demo_state()
    result = RRTPlanner(planner_settings()).plan_result(state, allow_fallback=False)

    assert result.route is not None
    geometry_length = sum(
        np.linalg.norm(np.subtract(end, start))
        for start, end in zip(result.route.waypoints[:-1], result.route.waypoints[1:])
    )
    assert result.route.costs["distance"] == geometry_length
    assert result.route.distance() == geometry_length
