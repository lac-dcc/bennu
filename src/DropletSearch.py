import tvm, sys, os, json, threading
from tvm import autotvm, auto_scheduler
from copy import deepcopy
import tvm._ffi
from tvm.auto_scheduler import MeasureInput, MeasureResult, _ffi_api
from tvm.auto_scheduler.search_task import SearchTask
from tvm.auto_scheduler.measure import local_builder_build
from tvm.auto_scheduler.workload_registry import (
    workload_key_to_tensors,
    register_workload_tensors,
)
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.template_factory import Template_factory
from src.utils import *


class Space:
    __cfg = []
    __dims = []
    __total_dims = 0
    __config_space = {}

    def __init__(self, cfg) -> None:
        self.__cfg = cfg
        self.__dims = []
        self.__total_dims = []
        self.__config_space = {}
        self.create_space()

    def __repr__(self) -> str:
        s = "Information of the space:\n"
        for key in self.__config_space:
            s += f"{key}: [{self.__config_space[key]}, dims={len(self.__config_space[key])}]\n"
        s += f"Total space: {self.__total_dims}\n"
        return s

    def __str__(self) -> str:
        s = "Information of the space:\n"
        for key in self.__config_space:
            s += f"{key}: [{self.__config_space[key]}, dims={len(self.__config_space[key])}]\n"
        s += f"Total space: {self.__total_dims}\n"
        return s

    def create_space(self) -> None:
        SP_space = [4, 8, 16, 24, 32, 48, 64]
        PR_space = [64, 128, 256, 512]
        for i in range(len(self.__cfg)):
            f = self.__cfg[i]
            if f[0] == "SP" and f[3] != 1:
                for j in range(len(f[4])):
                    self.__config_space[f"{f[0]}_{i}_{j}"] = add_space(
                        SP_space, [f[4][j]], f[3]
                    )
            elif f[0] == "PR":
                start_value = int(f[3].split("$")[-1])
                if start_value != 0:
                    self.__config_space[f"{f[0]}_{i}"] = [
                        f"auto_unroll_max_step${v}"
                        for v in add_space(PR_space, [start_value])
                    ]
        self.__dims = []
        for key in self.__config_space:
            self.__dims.append(len(self.__config_space[key]))
        self.__total_dims = 1
        if len(self.__dims) > 0:
            for d in self.__dims:
                self.__total_dims *= d

    def get_value(self, key, pos):
        return self.__config_space[key][pos]

    def index(self, values) -> int:
        value = 0
        for i in range(len(values) - 1):
            value += values[i] * self.__dims[i]
        value += values[-1]
        return value

    @property
    def dims(self) -> list:
        return self.__dims

    @property
    def total_dims(self) -> int:
        return self.__total_dims


class Droplet:
    count = 0
    log = None
    task = None
    json_file = None
    trials = 100
    space = None
    pos = None
    batch = 10
    next = []
    final_log = None
    best_choice = []
    visited = []
    execution = 1
    total_execution = 1
    pvalue = 0.05
    trying = 0

    # params
    timeout = 20
    number = 3
    repeat = 3
    min_repeat_ms = 100
    cooldown_interval = 0.0
    enable_cpu_cache_flush = False
    device = 0
    n_parallel = os.cpu_count()
    build_func = "default"

    def __init__(
        self, json_file, workload_key, target, log, trials=100, pvalue=0.05
    ) -> None:
        self.json_file = json_file
        self.final_log = write_file(json_file, log)
        self.log = write_file(json_file)

        # register_workload_tensors(workload_key, workload_key_to_tensors(workload_key))

        self.task = SearchTask(workload_key=workload_key, target=target)
        self.trials = trials
        self.pvalue = pvalue
        self.space = Space(json_file["i"][1][1])
        self.next = [np.zeros(len(self.space.dims), dtype=int)]
        best_avg, _ = get_best_time(self.log)
        self.best_choice = [
            np.zeros(len(self.space.dims), dtype=int),
            np.mean(best_avg),
            best_avg,
        ]
        self.count = 1
        self.execution = 1
        self.trying = 0
        self.visited = [0]

        if len(self.space.dims) > 0:
            self.total_execution = max(self.space.dims)

        auto_scheduler.workload_registry.register_workload_tensors(
            self.task.workload_key, self.task.compute_dag.tensors
        )

        #print(self.space)

    def has_next(self):
        return (
            self.count < min(self.trials, self.space.total_dims)
            and len(self.next) > 0
            and self.trying < 1
        )

    def next_batch(self, batch_size):
        i, json_file_list = 0, []
        while batch_size > 0 and i < len(self.next) and self.count < self.trials:
            json_file_list.append(self.apply_optimization(self.next[i]))
            i += 1
            self.count += 1
        log = create_file(json_file_list)
        return self.run(log)

    def num_to_bin(self, value, factor=1):
        bin_format = (
            str(0) * (len(self.space.dims) - len(bin(value)[2:])) + bin(value)[2:]
        )
        return [int(i) * factor for i in bin_format]

    # def p_value(self, elem_1, elem_2):
    #    if len(elem_1) <= 1 or len(elem_2) <= 1:
    #        return True
    #    return stats.ttest_ind(np.array(elem_1), np.array(elem_2)).pvalue <= self.pvalue

    def search_space(self, factor=1):
        search_space = []
        for i in range(1, 2 ** len(self.space.dims) - 1):
            if len(search_space) > 2 * self.batch:
                break
            search_space += [self.num_to_bin(i, factor)] + [self.num_to_bin(i, -factor)]
        return search_space

    def speculation(self):
        # Gradient descending direction prediction and search space filling
        while len(self.next) < self.batch and self.execution < self.total_execution:
            #print(self.execution)
            self.next += self.next_pos(self.search_space(self.execution))
            self.execution += 1
            #print(self.execution)

    def combination_space(self, p1, p2):
        new_p = np.zeros(len(p1), dtype=int)
        for i in range(len(p1)):
            sum_values = p1[i] + p2[i]
            new_p[i] = sum_values % self.space.dims[i] if sum_values > 0 else 0
        return new_p

    def update(self, results):
        found_best_pos = False
        for i in range(len(results)):
            value = np.mean(results[i])
            if (
                value < self.best_choice[1]
            ):  # and self.p_value(self.best_choice[2], results[i]):
                self.best_choice = [self.next[i], value, results[i]]
                found_best_pos = True
        self.next = []
        self.trying += 1
        if found_best_pos:
            self.next += self.next_pos(self.search_space())
            self.execution = 1
            self.trying = 0
        self.speculation()

    def next_pos(self, new_positions):
        "returns the neighbors of the best solution"
        next_set = []
        for p in new_positions:
            if len(next_set) > self.batch:
                break
            new_p = self.combination_space(p, self.best_choice[0])
            index = self.space.index(new_p)
            if index not in self.visited:
                self.visited.append(index)
                next_set.append(new_p)
        return next_set

    def apply_optimization(self, values):
        j_file_modified = deepcopy(self.json_file)
        cfg = j_file_modified["i"][1][1]
        index = 0
        for i in range(len(cfg)):
            f = cfg[i]
            if f[0] == "SP" and f[3] != 1:
                new_f = []
                for j in range(len(f[4])):
                    new_f.append(self.space.get_value(f"{f[0]}_{i}_{j}", values[index]))
                    index += 1
                cfg[i] = ["SP", f[1], f[2], f[3], new_f, f[5]]
            elif f[0] == "PR":
                if f[3] != "auto_unroll_max_step$0":
                    cfg[i] = [
                        "PR",
                        f[1],
                        f[2],
                        self.space.get_value(f"{f[0]}_{i}", values[index]),
                    ]
                    index += 1
        return j_file_modified

    def tune(self):
        """
        tune function:
        input: task
        """
        #print("opa")
        self.speculation()
        while self.has_next():
            #print(self.next)
            res = self.next_batch(self.batch)
            self.update(res)

    def run(self, log):
        inputs, _ = auto_scheduler.RecordReader(log).read_lines()
        results = np.zeros((len(inputs), self.repeat), dtype=float)
        for i in range(len(inputs)):
            state = self.task.compute_dag.infer_bound_from_state(inputs[i].state)
            inp = [MeasureInput(self.task, state)]

            res = _ffi_api.Run(
                self.task,
                state.state_object,
                self.timeout,
                self.number,
                self.repeat,
                self.min_repeat_ms,
                self.cooldown_interval,
                self.enable_cpu_cache_flush,
                self.device,
                self.n_parallel,
                self.build_func,
            )
            _ffi_api.SaveRecords(self.final_log, inp, res)
            results[i] = [v.value for v in res[0].costs]
        return results
