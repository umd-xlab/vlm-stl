import csv
import time

class Dripper:
    def __init__(self, feed_path, robot_states_path, stop_event):
        self.feed_path = feed_path
        self.robot_states_path = robot_states_path
        self.stop_event = stop_event

    def start(self):
        if self.robot_states_path == None:
            self.stop_event.set()
            return True
        with open(self.robot_states_path, mode='r') as file:
            csvFile = csv.reader(file)
            with open (self.feed_path, mode='w'
                #'a'
                ) as f:
                writer = csv.writer(f)
                for line in csvFile:
                    writer.writerow(line)
                    f.flush()
                    time.sleep(1)
        self.stop_event.set()

        return True