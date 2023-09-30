import tvm
from tvm import autotvm, te
from src.module.utils import *

class Template_autotvm():

    cfg = None
    sch = None
    tensor = None
    axis = None
    cache = None
    cache_axis = None
    args = []
    #search_space = [2, 4, 8, 16, 24, 32, 64, 96, 128] # TODO: Find best values 
    search_space = [4] # TODO: Find best values 
    annotation = []

    def __init__(self, tensor, args) -> None:
        '''
            Initializes the class constructor

            * \param tensor 
            * \param args
        '''
        self.sch = te.create_schedule(tensor.op)
        self.tensor = tensor
        self.cfg = autotvm.get_config()
        self.args = args

        self.axis = []
        for t in self.sch[self.tensor].op.axis:
            self.axis.append(t)
        for t in self.sch[self.tensor].op.reduce_axis:
            self.axis.append(t)

    def ret(self):
        '''
            return function
        '''
        return self.sch, self.args

    def CHW(self, list_CHW):
        '''
            CHW: CacheWriteStep

            * \param stage_id The index of the stage to be cache_write.
            * \param scope_name The scope name of the newly added stage.
            
            CacheWriteStep(int stage_id, String scope_name)

            TODO: Not finished yet.
        '''
        assert len(list_CHW) == 2
        stage_id, scope_name = list_CHW
        name = f'CHW'
        self.cfg.define_knob(name, ["None", scope_name])
        if self.cfg[name].val == scope_name:
            bn = 32 # size of cache in 32 blocks

            self.sch = te.create_schedule(self.tensor.op)

            # Allocate write cache
            self.cache = self.sch.cache_write(self.tensor, "local")
            _, n, _, _ = self.sch[self.tensor].tile(self.tensor.op.axis[0], self.tensor.op.axis[1], bn, bn)

            # Write cache is computed at no
            self.sch[self.cache].compute_at(self.sch[self.tensor], n)

            # New axes
            self.cache_axis = []
            for t in self.sch[self.cache].op.axis:
                self.cache_axis.append(t)
            for t in self.sch[self.cache].op.reduce_axis:
                self.cache_axis.append(t)

    def print(self):
        '''
            Print tensor function
        '''
        print(tvm.lower(self.sch, self.args, simple_mode=True))

    def RE_fixed(self, list_order):
        '''
            RE_fixed: Reorder step with a fixed list

            * \param stage_id The index of the stage to be reordered.
            * \param after_ids The expected indexes of the iterators after reorder.

            ReorderStep(int stage_id, const Array<Integer>& after_ids);
        '''
        if self.cfg[f'CHW'].val != 'local':
            axis = self.axis
        else:
            axis = self.cache_axis

        assert len(list_order) <= len(axis)
        print(self.values_space)

        p, count = [], 0
        for ord in list_order:
            p.append(axis[ord])
            count += 1
        for i in range(count, len(axis)):
            p.append(axis[i])
        
        if self.cfg[f'CHW'].val != 'local':
            self.sch[self.tensor].reorder(*p)
            self.axis = p
        else:
            self.sch[self.cache].reorder(*p)
            self.cache_axis = p

    def RE(self, size_order):
        '''
            RE: ReorderStep

            * \param stage_id The index of the stage to be reordered.
            * \param after_ids The expected indexes of the iterators after reorder.

            ReorderStep(int stage_id, const Array<Integer>& after_ids);
        ''' 
        if len(self.axis) == 0:
            return
        name = f'RE_{0}'
        
        self.cfg.define_knob(name, [i for i in range(size_order)])

        perms = permutation(self.axis, size_order)
        for i, p in enumerate(perms):
            if self.cfg[name].val == i:
                self.sch[self.tensor].reorder(*p)

    def SP_fixed(self, list_SP):
        '''
            SP_fixed: SplitStep

            * \param stage_id The index of the stage to be split.
            * \param iter_id The index of the iterator to be split.
            * \param extent The extent length of the axis to split.
            * \param lengths The multiple split factors. Can be None to be filled by search policy.
            * \param inner_to_outer The split direction.
            
            SplitStep(int stage_id, int iter_id, Optional<PrimExpr> extent,
                        const Array<Optional<Integer>>& lengths, bool inner_to_outer);
        '''
        order = []
        self.values_space = []
        for iter_id in range(len(list_SP)):
            split_size = len(list_SP[iter_id])
            for i in range(split_size):
                if i == 0:
                    x0, y0 = self.sch[self.tensor].split(self.axis[iter_id], factor=list_SP[iter_id][i])
                    add(order, [x0, y0] if i == split_size-1 else [x0])
                    yp = y0
                else:
                    x, y = self.sch[self.tensor].split(yp, factor=list_SP[iter_id][i])
                    add(order, [x, y] if i == split_size-1 else [x])
                    yp = y
        self.axis = order # update the tensor's axis 
    
    def SP(self, list_iter_id):
        '''
            SP: SplitStep
    
            * \param stage_id The index of the stage to be split.
            * \param iter_id The index of the iterator to be split.
            * \param extent The extent length of the axis to split.
            * \param lengths The multiple split factors. Can be None to be filled by search policy.
            * \param inner_to_outer The split direction.
            
            SplitStep(int stage_id, int iter_id, Optional<PrimExpr> extent,
                        const Array<Optional<Integer>>& lengths, bool inner_to_outer);
        '''
        order = []

        if self.cfg[f'CHW'].val != 'local':
            axis = self.axis
            tensor = self.tensor
        else:
            axis = self.cache_axis
            tensor = self.cache
            
        self.values_space = []
        for iter_id in range(len(list_iter_id)):
            split_size = list_iter_id[iter_id]
            if split_size == 0:
                k = axis[iter_id]
                add(order, [k])
            else:
                for i in range(split_size):
                    name = f'SP_{iter_id}_{i}'
                    self.cfg.define_knob(name, self.search_space)
                    if i == 0:
                        x0, y0 = self.sch[tensor].split(axis[iter_id], self.cfg[name].val)
                        self.values_space.append(self.cfg[name].val)
                        add(order, [x0, y0] if i == split_size-1 else [x0])
                        yp = y0
                    else:
                        x, y = self.sch[tensor].split(yp, self.cfg[name].val)
                        self.values_space.append(self.cfg[name].val)
                        add(order, [x, y] if i == split_size-1 else [x])
                        yp = y
        
        if self.cfg[f'CHW'].val != 'local':
            self.axis = order
        else:
            self.cache_axis = order

    def AN(self):
        '''
            AN: AnnotationStep

            * \brief The constructor.
            * \param stage_id The index of the stage to add annotation.
            * \param iter_id The index of the iterator to add annotation.
            * \param ann The annotation type of this step.
            
            AnnotationStep(int stage_id, int iter_id, IteratorAnnotation ann);
        '''
        pass

    def FU(self):
        '''
            FU: FuseStep
            
            * \param stage_id The index of the stage to be fused.
            * \param fused_ids The index of the iterators to be fused.
            
            FuseStep(int stage_id, const Array<Integer>& fused_ids);
        '''
        # TODO: Grow up the number of fusion, currently only between two tensor 
        # is possible.
        name = f'FU_{0}'
        size_fusion = len(self.axis)-1
        self.cfg.define_knob(name, [i for i in range(size_fusion)])

        for i in range(size_fusion):
            if self.cfg[name].val == i:
                fused = self.sch[self.tensor].fuse(self.axis[i], self.axis[i+1])
                update(self.axis, [self.axis[i], self.axis[i+1]], fused, i)

    def FU_fixed(self, list_fuse):
        '''
            FU_fixed: Fuse step with a list

            * \param stage_id The index of the stage to be fused.
            * \param fused_ids The index of the iterators to be fused.
            
            FuseStep(int stage_id, const Array<Integer>& fused_ids);
        '''
        i, pos = 0, 0
        p = self.axis.copy()
        while i < len(list_fuse):
            if i == 0:
                t1 = p[list_fuse[i]]
                t2 = p[list_fuse[i+1]]
                pos = list_fuse[i]
                pfused = self.sch[self.tensor].fuse(t1, t2)
                update(self.axis, [t1, t2], pfused, pos)
                i += 1
            else:
                tn = p[list_fuse[i]]
                fused = self.sch[self.tensor].fuse(pfused, tn)
                update(self.axis, [pfused, tn], fused, pos)
                pfused = fused
            i += 1

    def PR(self, var, pragma_type):
        '''
            PR: PragmaStep

            * \param stage_id The index of the stage to be fused.
            * \param iter_id The index of the iterator to add pragma.
            * \param pragma_type The pragma string.
            pragma_type options: "auto_unroll_max_step", "auto_unroll_max_depth", "unroll_explicit"
        '''
        assert pragma_type in ["auto_unroll_max_step", "auto_unroll_max_depth", "unroll_explicit"]

        name = f'PR_{var}_{pragma_type}'
        pragma_size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.cfg.define_knob(name, [i for i in pragma_size])
        
        for i in pragma_size:
            if self.cfg[name].val == i:
                self.sch[self.tensor].pragma(self.axis[var], pragma_type, self.cfg[name].val)

    def PR_fixed(self, list_pragma):
        '''
            PR: PragmaStep with fixed values

            * \param stage_id The index of the stage to be fused.
            * \param iter_id The index of the iterator to add pragma.
            * \param pragma_type The pragma string.
            pragma_type options: "auto_unroll_max_step", "auto_unroll_max_depth", "unroll_explicit"
        '''
        assert len(list_pragma) == 3
        assert var < len(self.axis)

        var, pragma_type, size = list_pragma
        self.sch[self.tensor].pragma(self.axis[var], pragma_type, size)

    def FSP(self):
        '''
            FSP: FollowSplitStep
            
            * \param stage_id The index of the stage to be split.
            * \param iter_id The index of the iterator to be split.
            * \param src_step_id The index of the split step to be followed in the history.
            * \param n_split The number of split level.
            
            FollowSplitStep(int stage_id, int iter_id, int src_step_id, int n_split)
        '''
        pass

    def FSP_fixed(self, list_FSP):
        '''
            FSP: FollowSplitStep with values fixed
            
            * \param stage_id The index of the stage to be split.
            * \param iter_id The index of the iterator to be split.
            * \param src_step_id The index of the split step to be followed in the history.
            * \param n_split The number of split level.
            
            FollowSplitStep(int stage_id, int iter_id, int src_step_id, int n_split)
        '''
        assert len(list_FSP) == 4
        stage_id, iter_id, src_step_id, n_split = list_FSP

        order = self.axis.copy()

        if n_split == 0:
            k = self.axis[src_step_id]
            insert(order, [k], src_step_id)
        else:
            for i in range(n_split):
                name = f'FSP_{src_step_id}_{i}'
                self.cfg.define_knob(name, self.search_space)
                if i == 0:
                    x0, y0 = self.sch[self.tensor].split(self.axis[src_step_id], self.cfg[name].val)
                    insert(order, [x0, y0] if i == n_split-1 else [x0], src_step_id)
                    yp = y0
                else:
                    x, y = self.sch[self.tensor].split(yp, self.cfg[name].val)
                    insert(order, [x, y] if i == n_split-1 else [x], src_step_id)
                    yp = y
        self.axis = order # update the tensor's axis 

    def FFSP(self):
        '''
            FFSP: FollowFusedSplitStep

            * \param stage_id The index of the stage to be split.
            * \param iter_id The index of the iterator to be split.
            * \param src_step_ids An array of index for split step to be followed in the history.
            * \param level Use the length in this split level.
            * \param factor_or_nparts If this is true, use factor. Otherwise, use nparts.
            
            FollowFusedSplitStep(int stage_id, int iter_id, const Array<Integer>& src_step_ids, int level,
                       bool factor_or_nparts);
        '''
        pass
    
    def FFSP_fixed(self, list_FFSP):
        '''
            FFSP: FollowFusedSplitStep

            * \param stage_id The index of the stage to be split.
            * \param iter_id The index of the iterator to be split.
            * \param src_step_ids An array of index for split step to be followed in the history.
            * \param level Use the length in this split level.
            * \param factor_or_nparts If this is true, use factor. Otherwise, use nparts.
            
            FollowFusedSplitStep(int stage_id, int iter_id, const Array<Integer>& src_step_ids, int level,
                       bool factor_or_nparts);
        '''
        assert len(list_FFSP) == 5
        stage_id, iter_id, src_step_ids, level, factor_or_nparts = list_FFSP
        pass

    def SA(self):
        '''
            SA: StorageAlignStep
      
            * \param stage_id The index of the stage to be aligned.
            * \param iter_id The index of the iterator to be aligned.
            * \param factor The factor in alignment specification.
            * \param offset The offset in the alignment specification.

            StorageAlignStep(int stage_id, int iter_id, int factor, int offset)
        '''
        pass

    def CA(self):
        '''
            CA: ComputeAtStep
            * \param stage_id The index of the source stage.
            * \param target_stage_id The index of stage that this step will compute at to.
            * \param target_iter_id The index of iterator in target stage that this step will compute at to.
            
            ComputeAtStep(int stage_id, int target_stage_id, int target_iter_id);
        '''
        pass
    
    def CA_fixed(self, list_CA):
        '''
            CA:     Step with a list fixed
            * \param stage_id The index of the source stage.
            * \param target_stage_id The index of stage that this step will compute at to.
            * \param target_iter_id The index of iterator in target stage that this step will compute at to.
            
            ComputeAtStep(int stage_id, int target_stage_id, int target_iter_id);

            ['CA', 2, 3, 1]
        '''
        assert len(list_CA) == 3
        stage_id, target_stage_id, target_iter_id = list_CA

        print(self.axis)
        # TODO: Verify why this get error?
        # self.sch[self.tensor].compute_at(self.sch[self.tensor], self.axis[target_iter_id])
        pass

    def CI(self):
        '''
            CI: ComputeInlineStep

            * \param stage_id The index of the stage to be marked compute inlined.
            
            ComputeInlineStep(int stage_id);
        '''
        self.sch[self.tensor].compute_inline()

    def CR(self):
        '''
            CR: ComputeRootStep

            * \param stage_id The index of the stage to be marked compute at root.
            
            ComputeRootStep(int stage_id);
        '''
        self.sch[self.tensor].compute_root()

    def CHR(self):
        '''
            CHR: CacheReadStep

            * \param stage_id The index of the stage to be cache_read.
            * \param scope_name The scope name of the newly added stage.
            * \param reader_stage_ids The indices of reader stages.
            
            CacheReadStep(int stage_id, String scope_name, const Array<Integer>& reader_stage_ids);
        '''
        pass

    def RF(self):
        '''
            RF: RfactorStep

            * \param stage_id The index of the stage to be factored.
            * \param iter_id The index of the iterator to be factored.
            * \param factor_iter_id The position where the new iterator is placed.
            */
            RfactorStep(int stage_id, int iter_id, int factor_iter_id);
        '''
        pass
