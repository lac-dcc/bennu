""" The class of Space used to optimize the Ansor parameters """

import os
import sys
import json
import numpy as np
from copy import deepcopy
from typing import Callable, Tuple, Union, List, Any
import tvm
import time
import random as rd

from tvm import meta_schedule as ms
from tvm.target import Target
from tvm.tir import Schedule
from tvm.meta_schedule.database import Workload, TuningRecord
from tvm.meta_schedule.utils import remove_build_dir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import *


class Space:
    """Space class

    Parameters
    ----------
    data: json data
        A json file template
    workload: json data
        A json file workload
    target: Target data
        Target device information
    """

    def __init__(self, data: json, workload: json, target: Target):
        self.cfg = deepcopy(data)
        self._id = data[0]
        self.workload = Workload.from_json(workload)
        self.target = target
        self.dev = self.get_device_type(target)
        self.total_dims, self.dims = 0, []
        self.config_space = {}
        self.start = []
        self.create_space()

    def __repr__(self) -> str:
        """Print the config space"""
        out = ""
        for key in self.config_space:
            out += f"{key}: dims={self.config_space[key]}\n"
        out += f"Total dimensions: {self.total_dims}\n"
        return out

    def __str__(self) -> str:
        """Print the config space"""
        out = ""
        for key in self.config_space:
            out += f"{key}: dims={self.config_space[key]}\n"
        out += f"Total dimensions: {self.total_dims}\n"
        return out

    def get_value(self, key, pos):
        """Return the space"""
        return self.config_space[key][pos]

    def add_space(self, space_list: list, element_list: list, limit=10000) -> list:
        """Return a list without repeat and with limited value"""
        new_list = element_list
        for elem in space_list:
            if elem not in new_list and elem <= limit:
                new_list.append(elem)
        return new_list

    def knob2point(self, knob):
        """Convert a array to point"""
        point = 0
        for j, k in enumerate(knob):
            point += int(np.prod(self.dims[:j])) * k
        return point

    def point2knob(self, point):
        """Convert point form (single integer) to knob (vector)"""
        knob = []
        for dim in self.dims:
            knob.append(point % dim)
            point //= dim
        return knob

    def power_of_two(self, min: int, max: int) -> list:
        """Return power of two array in interval"""
        return [2**i for i in range(min, max + 1)]

    def get_index(self, array: list, value: int):
        for i in range(len(array)):
            if array[i][0] == value:
                return i
        return -1

    def template(self, values=[], create=True):
        idx = -1
        config = deepcopy(self.cfg[1])
        # TODO: improve this array access, very confuse
        for counter, cfg in enumerate(config[0][0]):
            # print(counter, cfg)
            opt = cfg[0]
            if opt == "Annotate":
                ann_key = cfg[2]
                # TODO: which interval is interesting here
                if ann_key == ["meta_schedule.parallel"]:
                    interval = self.power_of_two(5, 9)
                elif ann_key == ["meta_schedule.vectorize"]:
                    interval = self.power_of_two(4, 8)
                elif ann_key == ["pragma_auto_unroll_max_step"]:
                    interval = self.power_of_two(7, 11)
                else:
                    continue
                idx += 1
                key = f"ann_{idx}"
                ann_value = cfg[1][1]
                if create:
                    self.config_space[key] = self.add_space(interval, [ann_value])
                else:
                    cfg[1][1] = self.get_value(key, values[idx])
            elif opt == "SampleCategorical":
                # TODO: study this opt
                pass
            elif opt == "SamplePerfectTile":
                # print(counter, config)
                tile = config[0][1]
                tile_idx = self.get_index(tile, counter)
                tile_val = tile[tile_idx][1]
                interval = self.power_of_two(1, 6)
                for i in range(len(tile_val)):
                    # don't optimize tile with size 1
                    # if tile_val[i] == 1:
                    #    continue
                    idx += 1
                    key = f"sp_{counter}_{idx}"
                    sp = tile_val[i]
                    if create:
                        self.config_space[key] = self.add_space(interval, [sp])
                    else:
                        config[0][1][tile_idx][1][i] = self.get_value(key, values[idx])
            elif opt == "TransformLayout":
                # Sol: removing transformLayout, until the bug is not resolved with Record.
                del config[0][0][counter]
        if create:
            return None

        # print(config)
        # rint("\n")
        return config

    def create_space(self):
        """Create the space using Meta's space"""
        self.template(create=True)
        # print(self.config_space)
        self.dims = []
        for key in self.config_space:
            self.dims.append(len(self.config_space[key]))
        self.total_dims = 1
        if len(self.dims) > 0:
            for dim in self.dims:
                self.total_dims *= dim

    def get_device_type(self, target: Target) -> str:
        """Get the device type string from a target.

        Parameters
        ----------
        target : Target
            The target to get the device type from.

        Returns
        -------
        device_type : str
            The device type string.
        """
        if target.kind.name == "llvm":
            return "cpu"
        elif target.kind.name == "cuda":
            return "cuda"
        else:
            raise RuntimeError(
                f"Unsupported target kind for device type: {target.kind.name}"
            )

    def save_log(
        self,
        path: str,
        record: ms.database.TuningRecord,
        results: ms.runner.RunnerResult,
    ) -> None:
        """Save the log file"""
        try:
            new_json = [self._id, record.as_json()]
        except:
            # TODO: Need to fix on 'TransformLayout' opt brings bug in Record's as_json function
            # print(record)
            return
        # update time
        new_json[1][1] = results
        write_file([new_json], path, "a")

    def run(
        self,
        json_file_list,
        final_log,
        timeout=10,
        number=2,
        repeat=3,
        min_repeat_ms=0,
        cpu_cache=False,
    ):
        """Execute a log file and save"""

        builder = ms.builder.LocalBuilder(timeout_sec=timeout)
        runner = ms.runner.LocalRunner(
            evaluator_config=ms.runner.EvaluatorConfig(
                number=number,
                repeat=repeat,
                min_repeat_ms=min_repeat_ms,
                enable_cpu_cache_flush=cpu_cache,
            ),
        )

        results = np.full(len(json_file_list), [10000], dtype=list)
        records = []
        for i, cfg in enumerate(json_file_list):
            # print(cfg)
            try:
                records.append(
                    TuningRecord.from_json(json.loads(json.dumps(cfg)), self.workload)
                )
            except:
                # TODO: Verify layer 22 of squeezenet has issue:
                # InternalError: Check failed: old_outputs.size() == new_outputs.size() (13 vs. 12)
                continue

        mods = []
        for record in records:
            sch = Schedule(self.workload.mod)
            # In some layers this is a heavy impact in time cost, so
            # I applied this only 25% of the samples.
            remove_postproc = True if rd.random() > 0.75 else False
            record.trace.apply_to_schedule(sch, remove_postproc=remove_postproc)
            # print(record.as_json())
            mods.append(sch.mod)

        builder_res = builder.build(
            [ms.builder.BuilderInput(mod, self.target) for mod in mods]
        )

        for i, record in enumerate(records):
            # print("test")
            try:
                inp = ms.runner.RunnerInput(
                    builder_res[i].artifact_path,
                    device_type=self.dev,
                    args_info=ms.arg_info.TensorInfo.from_prim_func(mods[i]["main"]),
                )
            except:
                # TODO: Study why this happen for some case
                # print(i, record.as_json())
                continue

            # run
            (runner_future,) = runner.run([inp])
            runner_res = runner_future.result()
            try:
                results[i] = [v.value for v in runner_res.run_secs]
            except:
                results[i] = [10000]

            # save the solution in json file
            self.save_log(final_log, record, results[i])

            # clean up
            remove_build_dir(builder_res[i].artifact_path)

        return results
