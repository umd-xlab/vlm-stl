import threading
import csv
from rta.monitor import Monitor

class Booter:
    def __init__(self, feed, rule_path, stop_event, config):
        """
        Called from rta.py responsible for starting one monitor per STL rule
        
        feed - shared in-memory feed for robot states
        rule_path - (string) file path containing STL rules
        stop_event - internal variable, used for threading, signal to stop running
        config - RTAConfig containing evaluation settings

        """
        self.feed = feed
        self.rule_path = rule_path
        self.stop_event = stop_event
        self.config = config
        self.monitors = []

    def start(self):
        """
        Reads STL rules, starts their monitors in separate threads
        and waits for completion. Propogates montior exceptions to rRTA class
        """
        threads = []
        errors = []
        error_lock = threading.Lock()

        def run_monitor(monitor):
            try:
                monitor.activate()
            except BaseException as e:
                with error_lock:
                    errors.append(e)

                self.stop_event.set()
                self.feed.finish()

        try:
            with open(self.rule_path, mode='r') as file:
                csvFile = csv.reader(file)
                next(csvFile)

                for rule in csvFile:
                    if self.stop_event.is_set():
                        break
                    monitor = Monitor(self.feed, rule, self.stop_event)
                    self.monitors.append(monitor)
                    thread = threading.Thread(target=run_monitor, args=(monitor,))
                    threads.append(thread)
                    thread.start()
        finally:
            if self.stop_event.is_set():
                self.feed.finish()
            for thread in threads:
                thread.join()

        if errors:
            raise RuntimeError("One or more STL monitors failed") from errors[0]