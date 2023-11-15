import tvm, sys, os, json
from tvm import autotvm, auto_scheduler
import tvm._ffi
from tvm.auto_scheduler import MeasureInput, MeasureResult, _ffi_api
from tvm.auto_scheduler.search_task import SearchTask

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.template_factory import Template_factory
from src.utils import *

class Droplet:

    log = None
    task = None
    json_file = None

    def __init__(self, json_file, workload_key, target) -> None:
        self.json_file = json_file
        self.log = write_file(json_file)
        self.task = SearchTask(workload_key=workload_key, target=target)
    
    def run(self):
        inputs, results = auto_scheduler.RecordReader(self.log).read_lines()

        filename = "/tmp/file.json"
        if os.path.isfile(filename):
            os.remove(filename)
        
        auto_scheduler.workload_registry.register_workload_tensors(
            self.task.workload_key, self.task.compute_dag.tensors
        )

        states = []
        for i in range(len(inputs)):
            print(type(inputs[i]), type(inputs[i].state), inputs[i].task.workload_key)
            try:
                states.append([i, self.task.compute_dag.infer_bound_from_state(inputs[i].state)])
            except:
                continue
        for i, state in states:
            inp = [MeasureInput(self.task, state)]
            res = _ffi_api.Run(self.task, state.state_object)
            _ffi_api.SaveRecords(filename, inp, res)

            print(results[i])
            f = open(filename)
            for l in f.readlines():
                print(l.strip())