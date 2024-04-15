""" DPMeta algorithm """

import sys, os, logging
from tvm.auto_scheduler.search_task import SearchTask

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import os
from src.utils import *
from src.space_ms import Space

LOGGER = logging.getLogger("autotvm")


class DropletMeta:
    """Tuner with droplet algorithm in Ansor.

    Parameters
    ----------
    json_file: str
        json format file
    target:
        hardware target
    log: str
        path to save json file
    trials: int
        number of samples, the default is 100
    pvalue: float
        statistical value to confidence level, the default is 0.05
    """

    def __init__(self, json_file, workload_file, target, log, pvalue=0.05) -> None:
        self.space = Space(json_file, workload_file, target)
        self.final_log = write_file([json_file], log)
        print("new file", json_file)
        self.pvalue = pvalue
        self.next = [(0, [0] * len(self.space.dims))]
        best_avg = get_ms_time(log)
        self.best_choice = [0, [0] * len(self.space.dims), best_avg]
        self.count, self.execution, self.found_best_pos = 1, 1, True
        self.visited, self.batch = set([0]), max(os.cpu_count(), 16)
        self.total_execution = 1
        if len(self.space.dims) > 0:
            self.total_execution = max(self.space.dims)
        self.dims, self.step = self.space.dims, 1

    def next_batch(self, batch_size):
        i, json_file_list = 0, []
        while i < len(self.next):
            if batch_size > 0 and self.count >= self.trials:
                break
            json_file_list.append(
                self.space.template(values=self.next[i][1], create=False)
            )
            i, self.count = i + 1, self.count + 1
        return self.space.run(json_file_list, self.final_log)

    def has_next(self):
        return len(self.next) > 0 and self.found_best_pos

    def tune(
        self,
        n_trial=100,
        measure_option=None,
        early_stopping=None,
        callbacks=(),
        si_prefix="G",
    ):
        self.trials = n_trial
        self.speculation()
        while self.has_next():
            ins, res = self.next_batch(self.batch)
            self.update(ins, res)

        print(self.best_choice)

    def num_to_bin(self, value, factor=1):
        bin_format = str(0) * (len(self.dims) - len(bin(value)[2:])) + bin(value)[2:]
        return [int(i) * factor for i in bin_format]

    def search_space(self, factor=1):
        search_space = []
        for i in range(1, 2 ** len(self.space.dims) - 1):
            if len(search_space) > 2 * self.batch:
                break
            search_space += [self.num_to_bin(i, factor)] + [self.num_to_bin(i, -factor)]
        return search_space

    def next_pos(self, new_positions):
        "returns the neighbors of the best solution"
        next_set = []
        for p in new_positions:
            if len(next_set) > self.batch:
                break
            new_p = [
                (x + y) % self.dims[i] if (x + y > 0) else 0
                for i, (x, y) in enumerate(zip(p, self.best_choice[1]))
            ]
            idx_p = self.space.knob2point(new_p)
            if idx_p not in self.visited:
                self.visited.add(idx_p)
                next_set.append((idx_p, new_p))
        return next_set

    def p_value(self, elem_1, elem_2):
        if len(elem_1) <= 1 or len(elem_2) <= 1:
            return True
        from scipy import stats  # pylint: disable=import-outside-toplevel

        return stats.ttest_ind(np.array(elem_1), np.array(elem_2)).pvalue <= self.pvalue

    def speculation(self):
        # Gradient descending direction prediction and search space filling
        while len(self.next) < self.batch and self.execution < self.total_execution:
            self.execution += self.step
            self.next += self.next_pos(self.search_space(self.execution))

    def update(self, inputs, results):
        self.found_best_pos, count_valids = False, 0
        for i, (_, res) in enumerate(zip(inputs, results)):
            try:
                if np.mean(self.best_choice[2]) > np.mean(res.costs):
                    self.best_choice = (self.next[i][0], self.next[i][1], res.costs)
                    self.found_best_pos = True
                count_valids += 1
            except TypeError:
                LOGGER.debug("Solution is not valid")
                continue
            else:
                continue

        self.next = self.next[self.batch : -1]
        if self.found_best_pos:
            self.next += self.next_pos(self.search_space())
            self.execution = 1
        self.speculation()
        # stop, because all neighborhoods are invalid.
        if count_valids == 0:
            self.next = []
            LOGGER.warning(
                f"Warning: early termination due to an all-invalid neighborhood \
                after iterations"
            )
