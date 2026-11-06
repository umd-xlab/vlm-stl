import math
import csv

from rta.state_data import State, StateData

class Preprocessor:
    def __init__(self, robot_trace, rule_path):
        """
        robot_trace provided by the RRT planer as robot_state.get_current_route_trace()
        rule_path rules used for preprocessing
        """
        self.robot_trace = robot_trace
        self.rule_path = rule_path

    def process_dist_rule(self, objx, objy):
        """
        Checks whether any STL rule requires the 'dst' variable.

        If required, calculates distance for every waypoint
        and returns the processed states in memory
        """
        state_data = StateData()
        trajectory = self.robot_trace["trajectory_samples"]

        with open(self.rule_path, mode='r') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            for r in csvFile:
                if r[2] == "dst":
                    for traj in trajectory:
                        d = self.calculate_distace(float(traj['x']),float(traj['y']),objx,objy)
                        state_data.append(State(time=float(traj['time_s']), variables={"dst":d}, waypoint_index=int(traj["waypoint_index"])))
                    break
        return state_data


    def calculate_distace(self, rx, ry, objx, objy) -> float:
        return math.sqrt(math.pow((objx-rx), 2) + math.pow((objy - ry), 2))


