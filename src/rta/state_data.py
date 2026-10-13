import threading
from dataclasses import dataclass, field

@dataclass
class State:
    time : float
    variables : dict[str, float]
    waypoint_index: int = None

@dataclass
class StateData:
    states: list[State] = field(default_factory=list)

    def append(self, state: State):
        self.states.append(state)

@dataclass
class StateFeed:
    states: list[State] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition)
    completed: bool = False

    def append(self, state: State):
        with self.condition:
            self.states.append(state)
            self.condition.notify_all()

    def finish(self):
        with self.condition:
            self.completed = True
            self.condition.notify_all()