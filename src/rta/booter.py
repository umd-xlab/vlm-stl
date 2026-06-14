import multiprocessing
import csv
from monitor import Monitor
import multiprocessing

class Booter:
    def __init__(self, feed_path, rule_path, stop_event):
        """
        Called from rta.py responsible for booting up the monitors for each stl rule provided in rule_path
        
        feed_path - (string) file path to where the robot state data will come in
        rule_path - (string) file path to all stl logic rules to be instantiated
        stop_event - internal variable, used for parallel processing, passed through to all classes
        failure_log - (string) file path where to send all stl rule violations, passed through to each monitor instance
        data_dump_log - (string) file path to where to send all outputed robustness values for stl rules, passed through to each monitor instance

        """
        self.feed_path = feed_path
        self.rule_path = rule_path
        self.stop_event = stop_event
        self.failure_log = "./logs/failure_log.csv"
        self.data_dump_log = "./logs/data_dump_log.csv"

        self.instantiate_log_files()

    def instantiate_log_files(self):
        """
        Instantiated Log Files with Propper Headers

        'Severity' refers to "how bad" the rule was violated
        'Robustness' refers specifically to output robustness

        """
        with open (self.failure_log, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(["Rule", "Time_of_Violation", "Severity"])
        with open(self.data_dump_log, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(["Rule", "Time", "Robustness"])

    def start(self):
        """
        The booter function for all individual rule monitors

        Reads each rule from the rule_path file
        Creates an instance of the monitor object and activates it
        Waits for stop_event to be "set" at which point terminates all processes

        """
        processes = []
        
        with open(self.rule_path, mode='r') as file:
            csvFile = csv.reader(file)
            next(csvFile)
            for r in csvFile:
                m = Monitor(self.feed_path, r, self.stop_event, self.failure_log, self.data_dump_log)
                process = multiprocessing.Process(target=m.activate)
                processes.append(process)
                process.start()

        self.stop_event.wait()
        for p in processes:
            p.join()

    
if __name__ == '__main__':
    """
    Sample setup to be run with 'python monitor.py' from terminal

    """
    stop_event = multiprocessing.Event()
    b = Booter('feed.csv', 'rules.csv', stop_event)
    b.start()
    time.sleep(15)
    stop_event.set()
