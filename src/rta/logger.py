import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

class RTALogger:
    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)

    def export(self, monitors):
        """
        Outputs al montior results, aka failure logs, datadumps and plots
        """

        self.output_folder.mkdir(parents=True, exist_ok=True)

        failures = []
        robustness_results = []

        for monitor in monitors:
            failures.extend(monitor.failures)
            robustness_results.extend(monitor.robustness_results)

        self._write_csv(self.output_folder / "failure_log.csv", failures)
        self._write_csv(self.output_folder / "data_dump_log.csv", robustness_results)
        self._save_plots(monitors)

    def _write_csv(self, path, results):
        """
        Write rule, time and robustness comlumns
        """
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["rule", "time", "waypoint_index", "robustness"])
            writer.writeheader()
            writer.writerows(results)

    def _save_plots(self, monitors):
        """
        Save 1 plot per monitor
        """
        plot_dir = self.output_folder / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        for index, monitor in enumerate(monitors):
            monitor.update_graph()

            clean_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]',"_", monitor.rule).strip(" .")
            if not clean_name:
                clean_name = "rule"
            filename = f"{index}_{clean_name}.png"
            monitor.fig.savefig(plot_dir / filename)
            plt.close(monitor.fig)
            monitor.fig = None
            monitor.graph = None