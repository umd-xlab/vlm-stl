import csv
import time

class Dripper:
    def __init__(self, feed_path, robot_states_path, stop_event):
        """
        Called from rta.py, used to help "fake online" mode in offline evaluation

        feed_path - (string) file path to where the dripper will send the robot state data
        robot_states_path - (string), contains full trace of robot states, will be rapidly fed into the file located at feed_path
        stop_event - internal variable, used for parallel processing, passed through to all classes

        """
        self.feed_path = feed_path
        self.robot_states_path = robot_states_path
        self.stop_event = stop_event

    def start(self):
        """

        If robot_states_path managed to be none, nothing to process therefore sends termination signal
        Opens robot_states_path (which should be csv) and rapidly sends it over line by line into the feed_path
        Speed for debugging can be adjusted on line 37 with time.sleep()
        Sends the stop_event for threads to all linked classes once complete

        """
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
                    time.sleep(1) #Can be very low value (units are in seconds) to increase speed of evaluation
        self.stop_event.set()

        return True