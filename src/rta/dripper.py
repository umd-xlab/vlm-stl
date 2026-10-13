import time

from rta.state_data import StateData, StateFeed

class Dripper:
    def __init__(self, feed: StateFeed, robot_states: StateData, stop_event):
        """
        Called from rta.py, used to help "fake online" mode in offline evaluation for propper robustness identification

        feed - shared in-memory feed for robot states
        robot_states - complete trajectory with prepreocessed rule variables
        stop_event - internal variable, used for threading, signal to stop running

        """
        self.feed = feed
        self.robot_states = robot_states
        self.stop_event = stop_event

    def start(self):
        """
        Published each preprocessed robot state into the shared feed (StateFeed)
        one observation at a time

        Signals feed compeltion after all states have been published
        """
        try: 
            if self.robot_states is None:
                return True

            for state in self.robot_states.states:
                if self.stop_event.is_set():
                    break

                self.feed.append(state)

                #Optional delay for debugging
                time.sleep(0.1)
            return True
        finally:
            self.feed.finish()