import json

from src.offline_demo import run_demo
from src.rrt_planner import PlannerStatus


def test_demo_keeps_route_out_of_state_and_writes_trace_and_plot(tmp_path):
    state, result, trace_path, plot_path = run_demo(tmp_path)

    # The planner output belongs to the coordinator, separate from robot state.
    assert result.status is PlannerStatus.SUCCESS

    # Check if the resulting route exists
    assert result.route is not None

    assert not hasattr(state, "current_route")
    assert "current_route" not in state.snapshot()

    assert state.current_pose is not None
    assert state.current_pose.as_point() == (2.5, 5.0)
    assert state.goal_pose is not None
    assert state.goal_pose.as_point() == (25.0, 25.0)
    assert state.current_observation is not None
    assert state.current_observation.top_down_costmap is not None
    assert state.current_observation.top_down_costmap.shape == (30, 30)

    # Check if the route contains waypoints
    assert result.route.waypoints

    # Check if the length of the route of waypoints equal the length of the route of trajectory
    assert len(result.route.waypoints) == len(result.route.trajectory)

    # Check that the trace path exists and is not empty, and that the plot path exists and is not empty
    assert trace_path.exists() and trace_path.stat().st_size > 0
    assert plot_path.exists() and plot_path.stat().st_size > 0

    trace = json.loads(trace_path.read_text())
    assert trace == result.route.to_trace()
    assert trace["frame_id"] == "map"
    assert trace["position_units"] == "meters"
    assert trace["time_units"] == "seconds"
    assert trace["origin_m"] == [0.0, 0.0]
    assert trace["resolution_m"] == 1.0
    assert trace["lethal_cost"] == 255.0
    assert trace["waypoints"]
    samples = trace["trajectory_samples"]
    assert all(
        samples[i + 1]["time_s"] > samples[i]["time_s"]
        for i in range(len(samples) - 1)
    )
    assert all(
        (samples[i]["x"], samples[i]["y"])
        != (samples[i + 1]["x"], samples[i + 1]["y"])
        for i in range(len(samples) - 1)
    )
    assert [(point["x"], point["y"]) for point in trace["waypoints"]] == [
        (sample["x"], sample["y"]) for sample in samples
    ]
