# What are important tests I would want to do?
# import numpy as np

# from src.offline_demo import(
    



# )




# test data imports correctly (and in the right format)

import threading
from pathlib import Path
from rta.state_data import State, StateData, StateFeed
from rta.dripper import Dripper
from rta.rta import RTA
from rta.config import RTAConfig

def test_dripper():
    robot_states = StateData()

    robot_states.append(State(time=0.0, variables={"dst": 10.0}))
    robot_states.append(State(time=1.0, variables={"dst": 20.0}))
    robot_states.append(State(time=2.0, variables={"dst": 30.0}))

    feed = StateFeed()
    stop_event = threading.Event()

    dripper = Dripper(feed, robot_states, stop_event)
    dripper.start()

    assert len(feed.states) == 3
    assert feed.states[0].variables["dst"] == 10.0
    assert feed.states[1].variables["dst"] == 20.0
    assert feed.states[2].variables["dst"] == 30.0
    assert feed.completed is True
    assert stop_event.is_set() is False

def test_offline_rta():
    config_path = Path(__file__).parent / "config.json"
    config = RTAConfig.from_json(config_path)

    robot_states = StateData()

    robot_states.append(State(time=0.0, variables={"dst":1}))
    robot_states.append(State(time=1.0, variables={"dst":0.5}))
    robot_states.append(State(time=2.0, variables={"dst":1}))

    rta = RTA(config, robot_states)
    rta.start_offline()

    assert len(rta.feed.states) == 3
    assert rta.feed.completed is True
    assert len(rta.booter.monitors) > 0

    results = rta.get_robustness_results()
    assert len(results) > 0

    failures = rta.get_failures()
    assert isinstance(failures, list)