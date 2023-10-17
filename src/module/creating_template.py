import tvm
from tvm import autotvm, te
from src.module.utils import *


class Template_autotvm:

    cfg = None
    sch = None
    start_tensor = None
    tensor = None
    axis = None
    args = []
    # search_space = [2, 4, 8, 16, 24, 32, 64, 96, 128] # TODO: Find best values
    search_space = [4]
    stage_to_axes = dict()  # TODO: creating axes in different stage

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

        # Initialize the axis
        self.tensor = tensor
        self.axis = []
        for t in self.sch[tensor].op.axis:
            self.axis.append(t)
        for t in self.sch[tensor].op.reduce_axis:
            self.axis.append(t)
        # FIXME: It's 2 because we have 3 operations
        # and the op 2 is computer type node
        # We need to do this generic
        self.UpdateStageToAxesMap(2)

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
        stage = self.sch.stages[stage_id]

        if type(stage.op) == tvm.te.tensor.ComputeOp:
            self.stage_to_axes[stage_id] = []
            for axis in stage.op.axis:
                self.stage_to_axes[2].append(axis)
            for axis in stage.op.reduce_axis:
                self.stage_to_axes[2].append(axis)
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
        stage = self.sch.stages[stage_id]

        name = f"CHW"
        self.cfg.define_knob(name, ["None", scope_name])

        # Rewrite the tensor
        self.tensor = None

        if self.cfg[name].val == scope_name:
            # cache size
            bn = 32

            # Allocate write cache
            cache = self.sch.cache_write(self.start_tensor, scope_name)
            _, no, _, _ = self.sch[self.start_tensor].tile(
                self.start_tensor.op.axis[0], self.start_tensor.op.axis[1], bn, bn
            )

            # Write cache is computed at no
            self.sch[cache].compute_at(self.sch[self.start_tensor], no)
            self.tensor = cache
        else:
            self.tensor = self.start_tensor

        # Update the axis
        self.axis = []
        for t in self.sch[self.tensor].op.axis:
            self.axis.append(t)
        for t in self.sch[self.tensor].op.reduce_axis:
            self.axis.append(t)

    def print(self):
        """
        Print tensor function
        """
        print(tvm.lower(self.sch, self.args, simple_mode=True))

    def RE_fixed(self, params):
        """
        RE_fixed: Reorder step with a fixed list

        * \param stage_id The index of the stage to be reordered.
        * \param after_ids The expected indexes of the iterators after reorder.

        ReorderStep(int stage_id, const Array<Integer>& after_ids);
        """
        assert len(params) == 2

        stage_id, after_ids = params

        new_order = []
        for i in range(len(self.axis)):
            if i < len(after_ids):
                new_order.append(self.axis[after_ids[i]])
            else:
                new_order.append(self.axis[i])
        # Reorder with the new order
        self.sch[self.tensor].reorder(*new_order)
        self.axis = new_order

    def RE(self, size_order):
        """
        RE: ReorderStep

        * \param stage_id The index of the stage to be reordered.
        * \param after_ids The expected indexes of the iterators after reorder.

        ReorderStep(int stage_id, const Array<Integer>& after_ids);
        """
        if len(self.axis) == 0:
            return
        name = f"RE_{0}"

        self.cfg.define_knob(name, [i for i in range(size_order)])

        perms = permutation(self.axis, size_order)
        for i, p in enumerate(perms):
            if self.cfg[name].val == i:
                self.sch[self.tensor].reorder(*p)

    def SP_fixed(self, list_SP):
        """
        SP_fixed: SplitStep

        * \param stage_id The index of the stage to be split.
        * \param iter_id The index of the iterator to be split.
        * \param extent The extent length of the axis to split.
        * \param lengths The multiple split factors. Can be None to be filled by search policy.
        * \param inner_to_outer The split direction.

        SplitStep(int stage_id, int iter_id, Optional<PrimExpr> extent,
                    const Array<Optional<Integer>>& lengths, bool inner_to_outer);
        """
        order = []
        for iter_id in range(len(list_SP)):
            split_size = len(list_SP[iter_id])
            for i in range(split_size):
                if i == 0:
                    x0, y0 = self.sch[self.tensor].split(
                        self.axis[iter_id], factor=list_SP[iter_id][i]
                    )
                    add(order, [x0, y0] if i == split_size - 1 else [x0])
                    yp = y0
                else:
                    x, y = self.sch[self.tensor].split(yp, factor=list_SP[iter_id][i])
                    add(order, [x, y] if i == split_size - 1 else [x])
                    yp = y
        self.axis = order  # update the tensor's axis

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
        stage = self.sch.stages[stage_id]

        order = []
        next_axis = self.axis[iter_id]
        for i in range(len(lengths)):
            name = f"SP_s{stage_id}_i{iter_id}_t{i}"
            self.cfg.define_knob(name, self.search_space)  
            x, y = stage.split(next_axis, self.cfg[name].val)
            if inner_to_outer == 1:
                add(order, [x, y] if i == len(lengths) - 1 else [x])
                next_axis = y
            else:
                add(order, [x, y] if i == len(lengths) - 1 else [y])
                next_axis = x
            
        insert(self.axis, order, iter_id)

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
            self.sch[self.tensor].unroll(self.axis[iter_id])
        elif ann == 2:  # vectorize
            self.sch[self.tensor].vectorize(self.axis[iter_id])
        elif ann == 3:  # parallel
            self.sch[self.tensor].parallel(self.axis[iter_id])
        elif ann in [4, 5, 6, 7, 8, 9, 10]:  # thread and block ids
            self.sch[self.tensor].bind(
                self.axis[iter_id], te.thread_axis(annotation_string[ann])
            )
        elif ann == 0:  # for
            pass  # do nothing in this case
        else:
            raise RuntimeError(f"Invalid annotation type {annotation_string[ann]}")

    def FU(self):
        """
        FU: FuseStep

        * \param stage_id The index of the stage to be fused.
        * \param fused_ids The index of the iterators to be fused.

        FuseStep(int stage_id, const Array<Integer>& fused_ids);
        """
        # TODO: Grow up the number of fusion, currently only between two tensor
        # is possible.
        name = f"FU_{0}"
        size_fusion = len(self.axis) - 1
        self.cfg.define_knob(name, [i for i in range(size_fusion)])

        for i in range(size_fusion):
            if self.cfg[name].val == i:
                fused = self.sch[self.tensor].fuse(self.axis[i], self.axis[i + 1])
                update(self.axis, [self.axis[i], self.axis[i + 1]], fused, i)

    def FU_fixed(self, params):
        """
        FU_fixed: Fuse step with a list

        * \param stage_id The index of the stage to be fused.
        * \param fused_ids The index of the iterators to be fused.

        FuseStep(int stage_id, const Array<Integer>& fused_ids);
        """
        assert len(params) == 2
        stage_id, fused_ids = params

        i, pos = 0, 0
        p = self.axis.copy()
        while i < len(fused_ids):
            if i == 0:
                t1 = p[fused_ids[i]]
                t2 = p[fused_ids[i + 1]]
                pos = fused_ids[i]
                pfused = self.sch[self.tensor].fuse(t1, t2)
                update(self.axis, [t1, t2], pfused, pos)
                i += 1
            else:
                tn = p[fused_ids[i]]
                fused = self.sch[self.tensor].fuse(pfused, tn)
                update(self.axis, [pfused, tn], fused, pos)
                pfused = fused
            i += 1

    def PR(self, var, pragma_type):
        """
        PR: PragmaStep

        * \param stage_id The index of the stage to be fused.
        * \param iter_id The index of the iterator to add pragma.
        * \param pragma_type The pragma string.
        pragma_type options: "auto_unroll_max_step", "auto_unroll_max_depth", "unroll_explicit"
        """
        assert pragma_type in [
            "auto_unroll_max_step",
            "auto_unroll_max_depth",
            "unroll_explicit",
        ]

        name = f"PR_{var}_{pragma_type}"
        pragma_size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.cfg.define_knob(name, [i for i in pragma_size])

        for i in pragma_size:
            if self.cfg[name].val == i:
                self.sch[self.tensor].pragma(
                    self.axis[var], pragma_type, self.cfg[name].val
                )

    def PR_fixed(self, params):
        """
        PR: PragmaStep with fixed values

        * \param stage_id The index of the stage to be fused.
        * \param iter_id The index of the iterator to add pragma.
        * \param pragma_type The pragma string.
        pragma options: "auto_unroll_max_step", "auto_unroll_max_depth", "unroll_explicit"
        """
        assert len(params) == 3

        stage_id, iter_id, pragma_type = params
        stage = self.sch.stages[stage_id]

        pragma, size = pragma_type.split("$")

        stage.pragma(self.axis[iter_id], pragma, int(size))

        if pragma == "auto_unroll_max_step":
            stage.pragma(self.axis[iter_id], "unroll_explicit", True)

    def FSP(self):
        """
        FSP: FollowSplitStep

        * \param stage_id The index of the stage to be split.
        * \param iter_id The index of the iterator to be split.
        * \param src_step_id The index of the split step to be followed in the history.
        * \param n_split The number of split level.

        FollowSplitStep(int stage_id, int iter_id, int src_step_id, int n_split)
        """
        split_factor = [0, 1, 2, 3, 4]
        order = self.axis.copy()

        for src_step_id in range(len(self.axis)):
            for n_split in split_factor:
                name = f"FSP_{src_step_id}_{n_split}"
                self.cfg.define_knob(name, self.search_space)
                if n_split != 0:
                    for i in range(n_split):
                        if i == 0:
                            x0, y0 = self.sch[self.tensor].split(
                                self.axis[src_step_id], self.cfg[name].val
                            )
                            insert(
                                order,
                                [x0, y0] if i == n_split - 1 else [x0],
                                src_step_id,
                            )
                            yp = y0
                        else:
                            x, y = self.sch[self.tensor].split(yp, self.cfg[name].val)
                            insert(
                                order, [x, y] if i == n_split - 1 else [x], src_step_id
                            )
                            yp = y
        self.axis = order  # update the tensor's axis

    def FSP_fixed(self, params):
        """
        FSP: FollowSplitStep with values fixed

        * \param stage_id The index of the stage to be split.
        * \param iter_id The index of the iterator to be split.
        * \param src_step_id The index of the split step to be followed in the history.
        * \param n_split The number of split level.

        FollowSplitStep(int stage_id, int iter_id, int src_step_id, int n_split)
        """
        assert len(params) == 4
        stage_id, iter_id, src_step_id, n_split = params

        order = self.axis.copy()

        if n_split != 0:
            for i in range(n_split):
                name = f"FSP_{src_step_id}_{i}"
                self.cfg.define_knob(name, self.search_space)
                if i == 0:
                    x0, y0 = self.sch[self.tensor].split(
                        self.axis[src_step_id], self.cfg[name].val
                    )
                    insert(order, [x0, y0] if i == n_split - 1 else [x0], src_step_id)
                    yp = y0
                else:
                    x, y = self.sch[self.tensor].split(yp, self.cfg[name].val)
                    insert(order, [x, y] if i == n_split - 1 else [x], src_step_id)
                    yp = y
        self.axis = order  # update the tensor's axis

    def FFSP(self):
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
        pass

    def FFSP_fixed(self, params):
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
        assert len(params) == 5
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
        # TODO: Implement SA opt
        pass

    def CA(self, params):
        """
        CA: ComputeAtStep
        * \param stage_id The index of the source stage.
        * \param target_stage_id The index of stage that this step will compute at to.
        * \param target_iter_id The index of iterator in target stage that this step will compute at to.

        ComputeAtStep(int stage_id, int target_stage_id, int target_iter_id);

        ['CA', 2, 3, 1]
        """
        assert len(params) == 3
        stage_id, target_stage_id, target_iter_id = params
        # TODO: Implement CA generic opt
        pass

    def CA_fixed(self, params):
        """
        CA:     Step with a list fixed
        * \param stage_id The index of the source stage.
        * \param target_stage_id The index of stage that this step will compute at to.
        * \param target_iter_id The index of iterator in target stage that this step will compute at to.

        ComputeAtStep(int stage_id, int target_stage_id, int target_iter_id);

        ['CA', 2, 3, 1]
        """
        assert len(params) == 3
        stage_id, target_stage_id, target_iter_id = params
        stage = self.sch.stages[stage_id]

        if target_stage_id == len(self.sch.stages):
            stage.compute_at(self.sch.stages[-1], self.axis[target_iter_id])
        else:
            stage.compute_at(
                self.sch.stages[target_stage_id], self.axis[target_iter_id]
            )

    def CI(self, stage_id):
        """
        CI: ComputeInlineStep

        * \param stage_id The index of the stage to be marked compute inlined.

        ComputeInlineStep(int stage_id);
        """
        assert stage_id < len(self.sch.stages)
        stage = self.sch.stages[stage_id]
        stage.compute_inline()

    def CR(self, stage_id):
        """
        CR: ComputeRootStep

        * \param stage_id The index of the stage to be marked compute at root.

        ComputeRootStep(int stage_id);
        """
        assert stage_id < len(self.sch.stages)
        stage = self.sch.stages[stage_id]
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
        # TODO: Implement CHR opt
        pass

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
        # TODO: Implement RF opt
        pass
