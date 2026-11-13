"""Exercise the planner-to-RTA contract without storing a route in robot state."""

import math

import numpy as np
import pytest

from src.robot_state import Observation, Pose2D, RobotState
from src.rrt_planner import PlannerSettings, PlannerStatus, RRTPlanner
from rta.config import RTAConfig
from rta.rta import RTA
from rta.signal_generator import SignalGenerator


@pytest.fixture
def planned_route():
    state = RobotState(
        current_pose=Pose2D(1.0, 1.0),
        goal_pose=Pose2D(8.0, 8.0),
        current_observation=Observation(
            timestamp=0.0, top_down_costmap=np.zeros((10, 10)),
        ),
    )
    result = RRTPlanner(PlannerSettings(seed=17)).plan_result(state)
    assert result.status is PlannerStatus.SUCCESS
    assert result.route is not None
    return state, result.route


def test_planner_trace_reaches_rta_without_route_in_state(planned_route, tmp_path):
    state, route = planned_route
    rules = tmp_path / "rules.csv"
    rules.write_text("rule,input,output\nout = (dst > 5),,dst\n")
    trace = route.to_trace()
    signals = SignalGenerator(
        trace, rules, obj_hardx=8.0, obj_hardy=8.0,
    ).generate_signals()
    rta = RTA(RTAConfig(
        config_description="Direct route handoff",
        offline_mode=True, enable_logging=False, stl_rule_file=str(rules),
    ), signals)
    rta.start_offline()

    assert not {"current_route", "previous_route", "route_history"} & vars(state).keys()
    assert "current_route" not in state.snapshot()
    assert len(rta.feed.states) == len(route.waypoints)
    for sample, signal in zip(trace["trajectory_samples"], rta.feed.states):
        assert signal.time == sample["time_s"]
        assert signal.waypoint_index == sample["waypoint_index"]
        assert signal.variables["dst"] == pytest.approx(
            math.hypot(sample["x"] - 8.0, sample["y"] - 8.0)
        )
    failures = rta.get_failures()
    assert failures
    assert all(failure["waypoint_index"] is not None for failure in failures)


@pytest.mark.parametrize("invalid_sample", ["time", "geometry", "nonfinite"])
def test_trace_rejects_invalid_timed_samples(planned_route, invalid_sample):
    _, route = planned_route
    samples = route.metadata["samples"]
    if invalid_sample == "time":
        samples[1]["time"] = samples[0]["time"]
    elif invalid_sample == "geometry":
        samples[1]["x"] += 1.0
    else:
        samples[1]["time"] = float("nan")
    with pytest.raises(ValueError):
        route.to_trace()
