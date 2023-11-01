import os, sys
import tvm
from tvm import autotvm, te
from copy import deepcopy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.module.utils import *


class Template_autotvm:
    cfg = None
    sch = None
    args = []
    start_tensor = None
    stages = [None]
    stage_to_axes = dict()
    lengths = []

    def __init__(self, tensor, args) -> None:
        """
        Initializes the class constructor

        * \param tensor
        * \param args
        """
        self.start_tensor = tensor
        self.sch = te.create_schedule(tensor.op)
        self.cfg = autotvm.get_config()
        self.args = args
        self.stages = list(self.sch.stages)

        # generic way to update all stages to axes map
        for i in range(len(self.stages)):
            self.UpdateStageToAxesMap(i)

    def ret(self):
        """
        return function
        """
        return self.sch, self.args

    def UpdateStageToAxesMap(self, stage_id):
        """
        Update the axes into dict

        * \param stage_id The index of the stage
        """
        stage = self.stages[stage_id]
        if type(stage.op) == tvm.te.tensor.ComputeOp:
            self.stage_to_axes[stage_id] = []
            for axis in stage.op.axis:
                self.stage_to_axes[stage_id].append(axis)
            for axis in stage.op.reduce_axis:
                self.stage_to_axes[stage_id].append(axis)
        elif type(stage.op) == tvm.te.tensor.PlaceholderOp:
            # do nothing
            pass
        else:
            raise RuntimeError(f"Invalid op {stage.op}")

    def CHW(self, params):
        """
        CHW: CacheWriteStep

        * \param stage_id The index of the stage to be cache_write.
        * \param scope_name The scope name of the newly added stage.

        CacheWriteStep(int stage_id, String scope_name)
        """
        assert len(params) == 2
        stage_id, scope_name = params
        stage = self.stages[stage_id]

        tensor_array = []
        for i in range(stage.op.num_outputs):
            tensor_array.append(stage.origin_op.output(i))

        # Allocate cache write
        outs = self.sch.cache_write(tensor_array, scope_name)

        # update the list stages
        self.stages = list(self.sch.stages)
        self.UpdateStageToAxesMap(stage_id)

        # adding new stage
        new_stage = self.sch[outs[0].op]
        self.stages[stage_id] = new_stage
        self.UpdateStageToAxesMap(stage_id + 1)

    def print(self):
        """
        Print tensor function
        """
        print(tvm.lower(self.sch, self.args, simple_mode=True))

    def RE(self, params):
        """
        RE_fixed: Reorder step with a fixed list

        * \param stage_id The index of the stage to be reordered.
        * \param after_ids The expected indexes of the iterators after reorder.

        ReorderStep(int stage_id, const Array<Integer>& after_ids);
        """
        assert len(params) == 2

        stage_id, after_ids = params
        stage = self.stages[stage_id]
        axes = self.stage_to_axes[stage_id]

        assert len(after_ids) <= len(axes)

        new_axes = []
        for i in range(len(axes)):
            if i < len(after_ids):
                new_axes.append(axes[after_ids[i]])

        # Reorder with the new order
        stage.reorder(*new_axes)
        # Update the axes, values, and stage
        self.stage_to_axes[stage_id] = new_axes
        self.stages[stage_id] = stage

    def SP(self, params):
        """
        SP: SplitStep

        * \param stage_id The index of the stage to be split.
        * \param iter_id The index of the iterator to be split.
        * \param extent The extent length of the axis to split.
        * \param lengths The multiple split factors. Can be None to be filled by search policy.
        * \param inner_to_outer The split direction.

        SplitStep(int stage_id, int iter_id, Optional<PrimExpr> extent,
                    const Array<Optional<Integer>>& lengths, bool inner_to_outer);
        """

        assert len(params) == 5
        stage_id, iter_id, extent, lengths, inner_to_outer = params
        stage = self.stages[stage_id]
        axes = self.stage_to_axes[stage_id]

        search_space = [4, 8, 16, 32]
        # search_space = []

        order = []
        next_axis = axes[iter_id]
        for i in range(len(lengths)):
            name = f"SP_s{stage_id}_i{iter_id}_t{i}"
            search = add_space(search_space, [lengths[i]])
            self.cfg.define_knob(name, search)
            val = self.cfg[name].val
            add_unique(self.lengths, [lengths[i]])
            x, y = stage.split(next_axis, val)
            if inner_to_outer == 1:
                add(order, [x, y] if i == len(lengths) - 1 else [x])
                next_axis = y
            else:
                add(order, [x, y] if i == len(lengths) - 1 else [y])
                next_axis = x
        insert(axes, order, iter_id)
        self.stages[stage_id] = stage

    def AN(self, params):
        """
        AN: AnnotationStep

        * \brief The constructor.
        * \param stage_id The index of the stage to add annotation.
        * \param iter_id The index of the iterator to add annotation.
        * \param ann The annotation type of this step.

        AnnotationStep(int stage_id, int iter_id, IteratorAnnotation ann);
        """
        assert len(params) == 3

        stage_id, iter_id, ann = params

        assert stage_id < len(self.stages)
        stage = self.stages[stage_id]
        axes = self.stage_to_axes[stage_id]

        annotation_string = {
            0: "for",
            1: "unroll",
            2: "vectorize",
            3: "parallel",
            4: "vthread",
            5: "blockIdx.x",
            6: "threadIdx.x",
            7: "blockIdx.y",
            8: "threadIdx.y",
            9: "blockIdx.z",
            10: "threadIdx.z",
            11: "tensorize",
        }

        if ann == 1:  # unroll
            stage.unroll(axes[iter_id])
        elif ann == 2:  # vectorize
            stage.vectorize(axes[iter_id])
        elif ann == 3:  # parallel
            stage.parallel(axes[iter_id])
        elif ann in [4, 5, 6, 7, 8, 9, 10]:  # thread and block ids
            stage.bind(axes[iter_id], te.thread_axis(annotation_string[ann]))
        elif ann == 0:  # for
            pass  # do nothing in this case
        else:
            raise RuntimeError(f"Invalid annotation type {annotation_string[ann]}")

        # update stage
        self.stages[stage_id] = stage

    def FU(self, params):
        """
        FU_fixed: Fuse step with a list

        * \param stage_id The index of the stage to be fused.
        * \param fused_ids The index of the iterators to be fused.

        FuseStep(int stage_id, const Array<Integer>& fused_ids);
        """
        assert len(params) == 2
        stage_id, fused_ids = params

        assert stage_id < len(self.stages)
        stage = self.stages[stage_id]
        axes = self.stage_to_axes[stage_id]

        i, pos = 0, 0
        p = self.stage_to_axes[stage_id].copy()
        while i < len(fused_ids):
            if i == 0:
                t1 = p[fused_ids[i]]
                t2 = p[fused_ids[i + 1]]
                pos = fused_ids[i]
                pfused = stage.fuse(t1, t2)
                update(axes, [t1, t2], pfused, pos)
                i += 1
            else:
                tn = p[fused_ids[i]]
                fused = stage.fuse(pfused, tn)
                update(axes, [pfused, tn], fused, pos)
                pfused = fused
            i += 1

        # update stage
        self.stages[stage_id] = stage
        self.stage_to_axes[stage_id] = axes

    def PR(self, params):
        """
        PR: PragmaStep with fixed values

        * \param stage_id The index of the stage to be fused.
        * \param iter_id The index of the iterator to add pragma.
        * \param pragma_type The pragma string. Options: "auto_unroll_max_step", "auto_unroll_max_depth", "unroll_explicit"
        """
        assert len(params) == 3

        stage_id, iter_id, pragma_type = params

        assert stage_id < len(self.stages)
        stage = self.stages[stage_id]
        axes = self.stage_to_axes[stage_id]

        pragma, size = pragma_type.split("$")

        stage.pragma(axes[iter_id], pragma, int(size))

        if pragma == "auto_unroll_max_step":
            stage.pragma(axes[iter_id], "unroll_explicit", True)

        # update the stage
        self.stages[stage_id] = stage

    def FSP(self, params):
        """
        FSP: FollowSplitStep with values fixed

        * \param stage_id The index of the stage to be split.
        * \param iter_id The index of the iterator to be split.
        * \param src_step_id The index of the split step to be followed in the history.
        * \param n_split The number of split level.

        FollowSplitStep(int stage_id, int iter_id, int src_step_id, int n_split)

        Example: [3, 0, 1, 1]
        """
        assert len(params) == 4
        stage_id, iter_id, src_step_id, n_split = params

        assert stage_id < len(self.stages)
        stage = self.stages[stage_id]
        axes = self.stage_to_axes[stage_id]

        order = []
        next_axis = axes[iter_id]
        for i in range(n_split):
            # name = f"FSP_s{stage_id}_i{iter_id}_t{i}"
            # self.cfg.define_knob(name, self.lengths)
            # TODO: Need to work here
            x, y = stage.split(next_axis, 4)
            add(order, [x, y] if i == n_split - 1 else [x])
            next_axis = y
        insert(axes, order, iter_id)

    def FFSP(self, params):
        """
        FFSP: FollowFusedSplitStep

        * \param stage_id The index of the stage to be split.
        * \param iter_id The index of the iterator to be split.
        * \param src_step_ids An array of index for split step to be followed in the history.
        * \param level Use the length in this split level.
        * \param factor_or_nparts If this is true, use factor. Otherwise, use nparts.

        FollowFusedSplitStep(int stage_id, int iter_id, const Array<Integer>& src_step_ids, int level,
                   bool factor_or_nparts);
        """
        stage_id, iter_id, src_step_ids, level, factor_or_nparts = params
        # TODO: Implement FFSP opt
        pass

    def SA(self, params):
        """
        SA: StorageAlignStep

        * \param stage_id The index of the stage to be aligned.
        * \param iter_id The index of the iterator to be aligned.
        * \param factor The factor in alignment specification.
        * \param offset The offset in the alignment specification.

        StorageAlignStep(int stage_id, int iter_id, int factor, int offset)
        """
        assert len(params) == 4
        stage_id, iter_id, factor, offset = params
        stage = self.stages[stage_id]
        axes = self.stage_to_axes[stage_id]
        stage.storage_align(axes[iter_id], factor, offset)
        self.stages[stage_id] = stage

    def CA(self, params):
        """
        CA: Step with a list fixed
        * \param stage_id The index of the source stage.
        * \param target_stage_id The index of stage that this step will compute at to.
        * \param target_iter_id The index of iterator in target stage that this step will compute at to.

        ComputeAtStep(int stage_id, int target_stage_id, int target_iter_id);
        """
        assert len(params) == 3
        stage_id, target_stage_id, target_iter_id = params

        assert stage_id < len(self.stages)
        assert target_iter_id < len(self.stage_to_axes[target_stage_id])

        stage = self.stages[stage_id]
        target_stage = self.stages[target_stage_id]
        target_axes = self.stage_to_axes[target_stage_id]

        stage.compute_at(target_stage, target_axes[target_iter_id])

    def CI(self, params):
        """
        CI: ComputeInlineStep

        * \param stage_id The index of the stage to be marked compute inlined.

        ComputeInlineStep(int stage_id);
        """
        assert len(params) == 1
        stage_id = params[0]
        assert stage_id < len(self.stages)
        stage = self.stages[stage_id]
        stage.compute_inline()

    def CR(self, params):
        """
        CR: ComputeRootStep

        * \param stage_id The index of the stage to be marked compute at root.

        ComputeRootStep(int stage_id);
        """
        assert len(params) == 1
        stage_id = params[0]
        assert stage_id < len(self.stages)
        stage = self.stages[stage_id]
        stage.compute_root()

    def CHR(self, params):
        """
        CHR: CacheReadStep

        * \param stage_id The index of the stage to be cache_read.
        * \param scope_name The scope name of the newly added stage.
        * \param reader_stage_ids The indices of reader stages.

        CacheReadStep(int stage_id, String scope_name, const Array<Integer>& reader_stage_ids);
        """
        assert len(params) == 3
        stage_id, scope_name, reader_stage_ids = params
        stage = self.stages[stage_id]

        readers = []
        for i in reader_stage_ids:
            readers.append(stage[i].origin_op)

        # Allocate cache read
        out = self.sch.cache_read(stage.origin_op.output(0), scope_name, readers)

        # update the list stages
        self.stages = list(self.sch.stages)
        self.UpdateStageToAxesMap(stage_id)

        # adding new stage
        new_stage = self.sch[out.op]
        self.stages[stage_id] = new_stage
        self.UpdateStageToAxesMap(stage_id + 1)

    def RF(self, params):
        """
        RF: RfactorStep

        * \param stage_id The index of the stage to be factored.
        * \param iter_id The index of the iterator to be factored.
        * \param factor_iter_id The position where the new iterator is placed.
        */
        RfactorStep(int stage_id, int iter_id, int factor_iter_id);
        """
        assert len(params) == 3
        stage_id, iter_id, factor_iter_id = params
        stage = self.stages[stage]
        axes = self.stage_to_axes[stage]

        tensor = stage.origin_op.output(0)
        axis = axes[iter_id]
        outs = self.sch.rfactor(tensor, axis, factor_iter_id)

        # update the list stages
        self.stages = list(self.sch.stages)
        self.UpdateStageToAxesMap(stage_id)

        # adding new stage
        new_stage = self.sch[outs[0].op]
        self.stages[stage_id] = new_stage
        self.UpdateStageToAxesMap(stage_id + 1)
