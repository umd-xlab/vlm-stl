import threading
import csv
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

    def to_csv(self, file_path:str):
        """
        Exports StateData to a CSV file
        """
        signals = set()
        for state in self.states:
            for name in state.variables:
                signals.add(name)
        fieldnames = ["time", "waypoint_index"] + list(signals)

        with open(file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames = fieldnames)
            writer.writeheader()
            for state in self.states:
                row = {"time":state.time, "waypoint_index": state.waypoint_index, **state.variables}
                writer.writerow(row)

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