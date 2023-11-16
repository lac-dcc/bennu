import tvm, sys, os, json
from tvm import autotvm, auto_scheduler
from copy import deepcopy
import tvm._ffi
from tvm.auto_scheduler import MeasureInput, MeasureResult, _ffi_api
from tvm.auto_scheduler.search_task import SearchTask
from tvm.auto_scheduler.measure import local_builder_build

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
        self.create_space()

    def __repr__(self):
        s = "Information of the space:\n"
        for key in self.__config_space:
            s += f"{key}: [{self.__config_space[key]}, dims={len(self.__config_space[key])}]\n"
        s += f"Total space: {self.__total_dims}\n"
        return s

    def __str__(self):
        s = "Information of the space:\n"
        for key in self.__config_space:
            s += f"{key}: [{self.__config_space[key]}, dims={len(self.__config_space[key])}]\n"
        s += f"Total space: {self.__total_dims}\n"
        return s

    def create_space(self):
        SP_space = [1, 2, 4, 8, 16, 32, 64]
        PR_space = [0, 4, 8, 16, 32, 64, 128, 256, 512]
        for i in range(len(self.__cfg)):
            f = self.__cfg[i]
            # print(f)
            if f[0] == "SP":
                for j in range(len(f[4])):
                    self.__config_space[f"{f[0]}_{i}_{j}"] = add_space(
                        SP_space, [f[4][j]]
                    )
            # TODO: Next opt
            # elif f[0] == "PR":
            #    self.config_space.append([f[3]] + PR_space)
        print(self.__config_space)
        self.__dims = []
        for key in self.__config_space:
            self.__dims.append(self.__config_space[key])
        if len(self.__dims) > 0:
            self.__total_dims = 1
            for d in self.__dims:
                self.__total_dims *= len(d)

    def get_value(self, key, pos):
        return self.__config_space[key][pos]

    @property
    def dims(self) -> list:
        return self.__dims

    @property
    def total_dims(self) -> int:
        return self.__total_dims


class Droplet:
    log = None
    task = None
    json_file = None
    trials = 100
    space = None
    pos = None
    batch = None

    def __init__(self, json_file, workload_key, target, trials=100) -> None:
        self.json_file = json_file
        self.log = write_file(json_file)
        self.task = SearchTask(workload_key=workload_key, target=target)
        self.trials = trials

        self.space = Space(json_file["i"][1][1])
        print("space =", self.space)

    def has_next(self):
        return len(self.batch) > 0

    def next_batch(self, batch_size):
        i = 0
        print(self.batch)
        while batch_size > 0 and i < len(self.batch):
            self.apply_optimization(self.batch[i])
            i += 1
        self.batch = self.batch[i:-1]
        print(self.batch)

    def update(self):
        # TODO: implement
        pass

    def load_history(self):
        # TODO: implement
        pass

    def apply_optimization(self, values):
        j_file_modified = deepcopy(self.json_file)
        cfg = j_file_modified["i"][1][1]
        for i in range(len(cfg)):
            f = cfg[i]
            if f[0] == "SP":
                new_f = []
                for j in range(len(f[4])):
                    key = f"{f[0]}_{i}_{j}"
                    # TODO: implement the logic to get the index
                    # new_f.append(self.space.get_value(key, values[]))
                    new_f.append(4)
                cfg[i] = ["SP", f[1], f[2], f[3], new_f, f[5]]
        # print(cfg)
        log = write_file(j_file_modified)
        self.run(log)

    def tune(self, repeat=3):
        self.batch = [np.zeros(len(self.space.dims))]

        for t in range(0, min(self.trials, self.space.total_dims)):
            self.update()
            self.next_batch(os.cpu_count())

            if not self.has_next():
                break

    def run(self, log, repeat=3):
        inputs, results = auto_scheduler.RecordReader(log).read_lines()

        filename = "/tmp/file_temp.json"
        if os.path.isfile(filename):
            os.remove(filename)

        auto_scheduler.workload_registry.register_workload_tensors(
            self.task.workload_key, self.task.compute_dag.tensors
        )

        timeout = 15
        number = 3
        repeat = 3
        min_repeat_ms = 100
        cooldown_interval = 0.0
        enable_cpu_cache_flush = False
        device = 0
        n_parallel = 12
        build_func = "default"

        for i in range(len(inputs)):
            state = self.task.compute_dag.infer_bound_from_state(inputs[i].state)
            inp = [MeasureInput(self.task, state)]

            res = _ffi_api.Run(
                self.task,
                state.state_object,
                timeout,
                number,
                repeat,
                min_repeat_ms,
                cooldown_interval,
                enable_cpu_cache_flush,
                device,
                n_parallel,
                build_func,
            )
            _ffi_api.SaveRecords(filename, inp, res)

            read_file(filename)
