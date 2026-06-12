import multiprocessing
import time
from dripper import Dripper
from booter import Booter

#This is the master of them all where the dripper returns true when its finished and that triggers the booter into a shutdown sequence

class RTA:
    def __init__(self, feed_path, rule_path, robot_states_path):
        self.feed_path = feed_path
        self.rule_path = rule_path
        self.robot_states_path = robot_states_path
        self.stop_event = None

    def start_online(self):
        try:
            self.stop_event = multiprocessing.Event()
            b = Booter(self.feed_path, self.rule_path, self.stop_event)
            b.start()
        finally:
            self.stop_event.set()

    def start_offline(self):
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
    r = RTA('feed.csv', 'rules.csv', robot_states_path='robot_states.csv')
    r.start_offline()
    print("finished")