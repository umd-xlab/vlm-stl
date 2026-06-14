import time
import pandas as pd
import rtamt
import csv
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from pathvalidate import sanitize_filename #https://www.tradingcode.net/python/sanitise-clean-filename/


class Monitor:
    def __init__(self, feed_path, rule_line, stop_event, failure_log, data_dump_log):
        self.feed_path = feed_path
        self.rule = None
        self.input = None
        self.output = None
        self.spec = None
        self.stop_event = stop_event
        self.failure_log = failure_log
        self.data_dump_log = data_dump_log
        self.fig, self.graph = plt.subplots()
        self.rule_data = ([],[]) # tuple containing x and y values of generated rule robustness
        self.state_data = {} #dictionary containing state datapoints for each variable

        self.add_rule(rule_line)
        self.instantiate_rule()
        self.instantiate_state_data()


    def add_rule(self, line: list):
        self.rule = line[0]
        self.input = line[1]
        if self.input == '':
            self.input = None
        self.output = line[2]
        self.spec = rtamt.StlDenseTimeSpecification(semantics=rtamt.Semantics.OUTPUT_ROBUSTNESS)

    def instantiate_rule(self):
        self.spec.declare_var('out', 'float')
        if self.input != None:
            self.spec.declare_var(str(self.input), 'float')
            self.spec.set_var_io_type(str(self.input), 'input')
        if self.output != None:
            self.spec.declare_var(str(self.output), 'float')
            self.spec.set_var_io_type(str(self.output), 'output')

        self.spec.spec = self.rule.strip("'")

        try:
            self.spec.parse()
            self.spec.pastify()
        except rtamt.RTAMTException as err:
            print('RTAMT Exception: {}'.format(err))
            sys.exit()
    
    def instantiate_state_data(self):
        for var in [self.input, self.output]:
            if var is not None:
                self.state_data[var] = ([],[])


    def update(self, line: pd.DataFrame):
        vars_to_update = []

        for var in [self.input, self.output]:
            if var != None:
                if not pd.isna(line.iloc[0][var]):
                    vars_to_update.append(var)
        
        updates = []

        for var in vars_to_update:
            time = float(line.iloc[0]['time'])
            signal = float(line.iloc[0][var])
            self.state_data[var][0].append(time)
            self.state_data[var][1].append(signal)
            updates.append([var, [[time, signal]]])
            print(f"Updating State {var} with data of time: {time} and signal: {signal}")
        print(f"Total state dictionary: {self.state_data}")

        # print(f"Rule: {self.rule}. Met Criterion for Following Update: {updates}")
        rob = self.spec.update(*updates)

        # print(f"Rule: {self.rule}. Generated Robustness online: {rob}")

        for i in range(0, len(rob)):
            self.route_data(rob[i])
            print(f"updating graph at time: {rob[i][0]} with value: {rob[i][1]}")
            self.rule_data[0].append(rob[i][0])
            self.rule_data[1].append(rob[i][1])
            self.update_graph()

    def route_data(self, output):
        time = output[0]
        robustness = output[1]
        if robustness < 0:
            with open (self.failure_log, mode='a') as f:
                writer = csv.writer(f)
                writer.writerow([self.rule, time, robustness])
        with open(self.data_dump_log, mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([self.rule, time, robustness])

    def update_graph(self):
        self.graph.clear()
        self.graph.step(*self.rule_data, where='post', label = 'Rule Output Robustness')
        for var in self.state_data:
            self.graph.step(*self.state_data[var], where='post', label = f'{var}', alpha=0.5)

        print(f"Current Graphing Data: {self.rule_data}")
        start_shading = None
        for (x,y) in zip(self.rule_data[0], self.rule_data[1]):
            if y == float('inf') and start_shading is None:
                start_shading = x
            elif y != float('inf') and start_shading is not None:
                self.graph.axvspan(start_shading, x, color='gray', alpha=0.25)
                start_shading = None

        self.graph.grid(True, which='both')
        self.graph.set_title(f"Rule: {self.rule}", wrap=True)
        self.graph.axhline(y=0, color='k')
        self.graph.axvline(x=0, color='k')
        self.graph.legend()
        plt.tight_layout()
        clean_name = sanitize_filename(self.rule, replacement_text="_")
        self.fig.savefig(f"plots/{clean_name}.png")

    def overseer(self):
        current_lines = 0

        def on_modified(event):
            cur_iter = count_lines()
            nonlocal current_lines
            if cur_iter > current_lines:
                current_lines = cur_iter
                df = pd.read_csv(event.src_path)
                # print(f"Rule: {self.rule}. Detected the update: {df[-1:]}")
                if current_lines != 1: #Skips the header
                    self.update(df[-1:])

        def count_lines():
            total=0
            with open('feed.csv', 'r') as f:
                for line in f:
                    total+=1
            return total

        event_handler = PatternMatchingEventHandler(patterns=['feed.csv'], ignore_patterns=[], ignore_directories=True, case_sensitive=True)
        event_handler.on_modified = on_modified

        observer = Observer()
        observer.schedule(event_handler, path=".", recursive=False)
        observer.start()

        self.stop_event.wait()
        observer.stop()
        observer.join()

    def activate(self):
        self.overseer()



if __name__ == "__main__":
    stop_event = multiprocessing.Event()
    with open ("./logs/failure_log.csv", mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["Rule", "Time_of_Violation", "Severity"])
    with open("./logs/data_dump_log.csv", mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["Rule", "Time", "Roboustness"])
    print("Activating Rule")
    m = Monitor('feed.csv', ['out = (spd < 5);',None, 'spd'], stop_event, "./logs/failure_log.csv", "./logs/data_dump_log.csv")
    # m = Monitor('feed.csv', ['out = ((srf < 0) implies eventually[0:5] (srf > 0));',None,'srf'], stop_event, "./logs/failure_log.csv", "./logs/data_dump_log.csv")
    m.activate()
    time.sleep(15)
    stop_event.set()