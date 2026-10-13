import threading

from rta.dripper import Dripper
from rta.booter import Booter
from rta.state_data import StateFeed
from rta.logger import RTALogger

class RTA:
    def __init__(self, config, robot_states=None):
        """
        Master class used to conduct runtime assurance (rta)

        config - RTAConfig containing evaluation settings
        robot_states - optional StateData containing the complete preprocessed trajectory
        stop_event - internal variable, used for threading, signal to stop running

        """
        self.config = config
        self.robot_states = robot_states
        self.stop_event = None
        self.feed = None
        self.booter = None

    def start_online(self):
        """
        Online Evaluation Mode

        Creates a stop event, to tell all threads when to shutdown
        Creates and starts the booter
        At the end, sets the stop event to terminate all threads

        """
        raise NotImplementedError("Online Mode has not yet been created")

    def start_offline(self):
        """
        Offline Evaluation Mode

        Creates a shared state feed.
        Starts the Booter and the Dripper in separate threads
        and waits for both to finish

        Exceptions in worker threads are propogated back to calling thread
        """
        self.stop_event = threading.Event()
        self.feed = StateFeed()
        d = Dripper(self.feed, self.robot_states, self.stop_event)
        self.booter = Booter(self.feed, self.config.stl_rule_file, self.stop_event, self.config)

        errors = []
        error_lock = threading.Lock()

        def run_worker(target):
            try:
                target()
            except BaseException as e:
                with error_lock:
                    errors.append(e)

                self.stop_event.set()
                self.feed.finish()

        p1 = threading.Thread(target=lambda: run_worker(d.start))
        p2 = threading.Thread(target=lambda: run_worker(self.booter.start))

        try:
            p2.start()
            p1.start()
            p1.join()
            p2.join()

            if errors:
                raise RuntimeError("Offline RTA evaluation failed") from errors[0]

            if self.config.enable_logging:
                if not self.config.logging_output_folder:
                    raise ValueError("logging_output folder must be configured when logging is enabled, else set to false")
                logger = RTALogger(self.config.logging_output_folder)
                logger.export(self.booter.monitors)
        finally:
            self.stop_event.set()

    def get_failures(self):
        """
        Returns violations collected from all monitors.
        """

        if self.booter is None:
            return []

        failures = []
        for monitor in self.booter.monitors:
            failures.extend(monitor.failures)

        return sorted(failures, key=lambda result: result["time"])

    def get_robustness_results(self):
        """
        Returns all robustness results from all monitors
        """
        if self.booter is None:
            return []

        results = []

        for monitor in self.booter.monitors:
            results.extend(monitor.robustness_results)

        return sorted(results, key=lambda result: result["time"])