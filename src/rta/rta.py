import multiprocessing
import time
from dripper import Dripper
from booter import Booter

class RTA:
    def __init__(self, feed_path, rule_path, robot_states_path = None):
        """
        Master class used to conduct runtime assurance (rta)

        feed_path - (string) file path to where the robot state data will come in
        rule_path - (string) file path to all stl logic rules to be implemented
        robot_states_path - (string) OPTIONAL, file path, to be used for offline evaluation. Contains full trace of robot states
        stop_event - internal variable, used for parallel processing, passed through to all classes

        """
        self.feed_path = feed_path
        self.rule_path = rule_path
        self.robot_states_path = robot_states_path
        self.stop_event = None

    def start_online(self):
        """
        Online Evaluation Mode

        Creates a stop event, to tell all threads when to shutdown
        Creates and starts the booter
        At the end, sets the stop event to terminate all threads

        """
        try:
            self.stop_event = multiprocessing.Event()
            b = Booter(self.feed_path, self.rule_path, self.stop_event)
            b.start()
        finally:
            self.stop_event.set()

    def start_offline(self):
        """
        Offline Evaluation Mode

        "Fakes online" by using Dripper class to rapidly feed full trace data in robot_states_path
        Creates and starts the booter.
        Links the processes together so they terminate when the Dripper class is done sending robot_states_path data
        At the end, sets the stop event to terminate all threads

        """
        if self.robot_states = None:
            raise Exception("cannot start offline processing without robot states data")
        try:
            self.stop_event = multiprocessing.Event()
            d = Dripper(self.feed_path, self.robot_states_path, self.stop_event)
            b = Booter(self.feed_path, self.rule_path, self.stop_event)
            p1 = multiprocessing.Process(target = d.start)
            p2 = multiprocessing.Process(target = b.start)
            p1.start()
            p2.start()
            self.stop_event.wait()
            p1.join()
            p2.join()
        finally:
            self.stop_event.set()

if __name__ == "__main__":
    """
    Sample setup to be run with 'python rta.py' from terminal
    
    """
    r = RTA('feed.csv', 'rules.csv', robot_states_path='robot_states.csv')
    r.start_offline()
    print("finished")