import multiprocessing
import csv
from monitor import Monitor
import multiprocessing

class Booter:
    def __init__(self, feed_path, rule_path, stop_event):
        self.feed_path = feed_path
        self.rule_path = rule_path
        self.stop_event = stop_event
        self.failure_log = "./logs/failure_log.csv"
        self.data_dump_log = "./logs/data_dump_log.csv"
        with open (self.failure_log, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(["Rule", "Time_of_Violation", "Severity"])
        with open(self.data_dump_log, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(["Rule", "Time", "Roboustness"])

    def start(self):
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
    stop_event = multiprocessing.Event()
    b = Booter('feed.csv', 'rules.csv', stop_event)
    b.start()
    time.sleep(15)
    stop_event.set()
