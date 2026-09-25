import math
import csv

class Preprocessor:
    def __init__(self, robot_trace, rule_path, output_path):
        """
        robot_trace provided by the RRT planer as robot_state.get_current_route_trace()
        rule_path rules used for preprocessing
        output_path where to place the processed robot states
        """
        self.robot_trace = robot_trace
        self.rule_path = rule_path
        self.output_path = output_path

    def process_dist_rule(self, objx, objy):
        trajectory = self.robot_trace["trajectory_samples"]
        with open(self.rule_path, mode='r') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            for r in csvFile:
                if r[2] == "dst":
                    with open (self.output_path, mode='w') as f:
                        writer = csv.writer(f)
                        writer.writerow(["time", "dst"])
                        for traj in trajectory:
                            d = self.calculate_distace(float(traj['x']),float(traj['y']),objx,objy)
                            writer.writerow([traj['time_s'], d])

    def calculate_distace(self, rx, ry, objx, objy) -> float:
        return math.sqrt(math.pow((objx-rx), 2) + math.pow((objy - ry), 2))


