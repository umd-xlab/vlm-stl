import math
import csv
import numpy as np
from scipy.spatial import KDTree

from rta.state_data import State, StateData

class SignalGenerator:
    """Generates Signals for RTA"""
    def __init__(self, robot_trace, rule_path, perception = None, grid_costmap = None, obj_hardx = None, obj_hardy = None):
        """
        Saves Modules/Parameters to use
        
        :param self: Signal Generator instance
        :param robot_trace: validated planner route trace from Route.to_trace()
        :param rule_path: path to STL rules CSV file
        :param perception: Perception Module instance
        :param grid_costmap: grid_costmap instance for coordinate conversion
        :param obj_hardx: x coordinate of hardcoded object
        :param obj_hardy: y coordinate of hardcoded object
        """
        self.robot_trace = robot_trace
        self.rule_path = rule_path
        self.perception = perception
        self.grid_costmap = grid_costmap
        self.environment_map = None
        self.objects_to_avoid = None #Based on environment_map to avoid colliding into trees (when treated as surface)
        self.obj_hardx = obj_hardx
        self.obj_hardy = obj_hardy

    def get_required_signals(self) -> set:
        """
        Returns a set of signals required

        Currently Supports:
            dst: distance to hardcoded object
            srf: surface classification
            dnr: distance to nearest tree/building

        Returns
        -------
        set[str]
            signal names
        """

        required_signals = set()

        with open(self.rule_path, mode='r') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            for r in csvFile:
                if r[2] in ["dst", "srf", "dnr"]:
                    required_signals.add(r[2])

        return required_signals

    def generate_signals(self) -> StateData:
        """
        Returns a StateData object of generated signals at every timestep

        Currently Supports:
            dst: distance to hardcoded object
            srf: surface classification
            dnr: distance to nearest tree/building

        Returns
        -------
        StateData = list[State]
            @dataclass
            class State:
                time : float
                variables : dict[str, float]
                waypoint_index: int = None
        """
        state_data = StateData()
        trajectory = self.robot_trace["trajectory_samples"]
        required_signals = self.get_required_signals()

        if "dst" in required_signals:
            if self.obj_hardx is None or self.obj_hardy is None:
                raise ValueError("Hard coded object position is missing for dst rule")

        if "srf" in required_signals:
            if self.perception is None or self.grid_costmap is None:
                raise ValueError("Perception or costmap is missing for srf rule")
            self.environment_map = self.perception.get_top_down_environment_state()

        if "dnr" in required_signals:
            if self.perception is None or self.grid_costmap is None:
                raise ValueError("Perception or costmap is missing for dnr rule")
            if self.environment_map is None:
                self.environment_map = self.perception.get_top_down_environment_state()
            self.objects_to_avoid = self.create_object_mapping()

        for traj in trajectory:
            variables = {}

            curx = float(traj['x'])
            cury = float(traj['y'])

            if "dst" in required_signals:
                variables["dst"] = self.calculate_distace(curx, cury, self.obj_hardx, self.obj_hardy)

            if "srf" in required_signals:
                variables["srf"] = self.calculate_current_surface(curx, cury, self.environment_map)

            if "dnr" in required_signals:
                variables["dnr"] = self.calculate_nearest_obstacle_distance(curx, cury, self.objects_to_avoid)

            state_data.append(State(time=float(traj["time_s"]), variables=variables, waypoint_index=int(traj["waypoint_index"])))

        return state_data

    def calculate_distace(self, curx: float, cury: float, objx: float, objy:float) -> float:
        """Returns the distance to hardcoded object

        Parameters
        ----------
        curx : float
            current x coordinate of robot
        cury : float 
            current y coordinate of robot
        objx : float
            x coordinate of hardcoded object
        objy : float
            y coordinate of hardcoded object
        
        Returns
        -------
        float
            distance
        """
        return math.sqrt(math.pow((objx-curx), 2) + math.pow((objy - cury), 2))

    def calculate_current_surface(self, curx: float, cury: float, environment_map) -> float:
        """Returns the signal based on current state

        Parameters
        ----------
        curx : float
            current x coordinate of robot
        cury : float 
            current y coordinate of robot
        environment_map : perception_module
            class presdictions at each cell
        
        Returns
        -------
        float
            surface signal (currently 1.0 for sidewalk, -1.0 for everything else)
        """
        cell = self.grid_costmap.world_to_grid((curx, cury))
        if cell is None: #Will avoid pathing outside of "known world"
            return -1.0
        row, col = cell

        if row < 0 or row > environment_map.shape[1] or col < 0 or col > environment_map.shape[2]:
            return -1.0

        class_idx = int(np.argmax(environment_map[:, row, col])) #Gets index of maximum value, aka the predicted class

        if class_idx == 0: #Uknown channel
            return -1.0

        class_name = self.perception.prompts[class_idx - 1]
        if class_name == "Sidewalk":
            return 1.0

        return -1.0

    def calculate_nearest_obstacle_distance(self, curx: float, cury: float, objects_to_avoid: KDTree) -> float:
        """Calculates distance to nearest object to avoid

        Parameters
        ----------
        curx : float
            current x coordinate of robot
        cury : float 
            current y coordinate of robot
        objects_to_avoid : KDTree
            all object avoidance points
        
        Returns
        -------
        float
            distance to nearest object to avoid
        """

        if self.grid_costmap.world_to_grid((curx, cury)) is None:
            return 0.0

        if objects_to_avoid is None:
            return float('inf')

        distance, _ = objects_to_avoid.query((curx, cury)) #Remmeber this will contain the center of each cell
        return distance



    def create_object_mapping(self) -> KDTree | None:
        """
        Create KDTree of obstacle_points for rapid lookup

        Returns
        -------
        KDTree
            x,y coordinates of all obstacles
        """

        # print(f"Environment map shape: {self.environment_map.shape}")
        # print(f"Planner costmap shape: {self.grid_costmap.costmap.shape}")
        # print(f"Perception classes: {self.perception.prompts}")

        #Currently b/c of demo its ["synthetic_traversable", "synthetic_obstacle"]
        #And my environment map shape is (3,1,1) when it should be (3,30,30) I think


        class_map = np.argmax(self.environment_map, axis=0)

        obstacle_classes = [0] #0 is the unkown channel

        for idx, class_name in enumerate(self.perception.prompts):
            if class_name in ["Tree", "Building"]:
                obstacle_classes.append(idx+1)

        obstacle_mask = np.isin(class_map, obstacle_classes) #True if obstacle
        obstacle_rows, obstacle_columns = np.where(obstacle_mask)

        # print("HERE")
        # print(obstacle_mask)
        # print("")

        obstacle_points = []

        for row, col in zip(obstacle_rows, obstacle_columns):
            point = self.grid_costmap.grid_to_world((int(row), int(col)))

            if point is not None:
                obstacle_points.append(point)

        if len(obstacle_points) == 0:
            return None

        return KDTree(obstacle_points) 