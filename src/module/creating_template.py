import sys, os

import tvm
from tvm import autotvm, te
from src.module.utils import *

''' cfg
Config: [[], 
            [
            ['SP', 2, 0, 1000, [5, 25, 4], 1], 
            ['SP', 2, 4, 700, [1, 35, 4], 1], ['SP', 2, 8, 800, [8], 1], 
            ['RE', 2, [0, 4, 1, 5, 8, 2, 6, 9, 3, 7]], 
            ['FU', 2, [0, 1, 2]], 
            ['AN', 2, 0, 3], 
            ['PR', 2, 0, 'auto_unroll_max_step$512'], 
            ['AN', 2, 7, 2]]]  
'''

class Template_autotvm():

    cfg = None
    sch = None
    tensor = None
    args = []
    search_space = [2, 4, 8, 16, 24, 32, 64, 96, 128] # TODO: Find best values 
    axis = None
    annotation = []

    def __init__(self, tensor, args) -> None:
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

    def CHW(self):
        '''
            CHW: CacheWriteStep
            TODO: Not finished yet.
        '''
        name = f'CHW'
        self.cfg.define_knob(name, [0, 1])
        if self.cfg[name].val != 0:
            new_CC = te.create_schedule(self.tensor.op)
            CC = new_CC.cache_write(self.tensor, 'local')

            #xo, yo, xi, yi = self.sch[self.tensor].tile(self.axis[0], self.axis[1], x_factor=32, y_factor=32)
            #self.sch[CC].compute_at(self.sch[self.tensor], xo)

            #self.annotation.append(['CHW', CC])
            #self.tensor = CC
            #CC = self.sch.cache_write(self.tensor, "global")
            print(CC)
        

    def print(self):
        '''
            Print tensor function
        '''
        print(tvm.lower(self.sch, self.args, simple_mode=True))

    def RE_fixed(self, list_order):
        '''
            RE_fixed: Reorder step with a list
        '''
        assert len(list_order) <= len(self.axis)
        p = []
        for ord in list_order:
            p.append(self.axis[ord])
        self.sch[self.tensor].reorder(*p)
        self.axis = p

    def RE(self, size_order):
        '''
            RE: ReorderStep
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
            SP_fixed:
        '''
        order = []
        for iter_id in range(len(list_SP)):
            split_size = len(list_SP[iter_id])
            for i in range(split_size):
                if i == 0:
                    x0, y0 = self.sch[self.tensor].split(self.axis[iter_id], factor=list_SP[iter_id][i])
                    if i == split_size-1:
                        add(order, [x0, y0])
                    else:
                        add(order, [x0])
                    yp = y0
                else:
                    x, y = self.sch[self.tensor].split(yp, factor=list_SP[iter_id][i])
                    if i == split_size-1:
                        add(order, [x, y])
                    else:
                        add(order, [x])
                    yp = y
        self.axis = order # update the tensor's axis 
    
    def SP(self, list_iter_id):
        '''
            SP: SplitStep
        '''
        order = []
        for iter_id in range(len(list_iter_id)):
            split_size = list_iter_id[iter_id]
            if split_size == 0:
                k = self.axis[iter_id]
                add(order, [k])
            else:
                for i in range(split_size):
                    name = f'SP_{iter_id}_{i}'
                    self.cfg.define_knob(name, self.search_space)

                    if i == 0:
                        x0, y0 = self.sch[self.tensor].split(self.axis[iter_id], self.cfg[name].val)
                        if i == split_size-1:
                            add(order, [x0, y0])
                        else:
                            add(order, [x0])
                        yp = y0
                    else:
                        x, y = self.sch[self.tensor].split(yp, self.cfg[name].val)
                        if i == split_size-1:
                            add(order, [x, y])
                        else:
                            add(order, [x])
                        yp = y
        self.axis = order # update the tensor's axis 

    def AN(self):
        '''
            AN: AnnotationStep
        '''
        pass

    def FU(self):
        '''
            FU: FuseStep
            Describe: fuse can fuse two consecutive axes of one computation.
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
            pragma_type options: "auto_unroll_max_step", "auto_unroll_max_depth", "unroll_explicit"
        '''
        name = f'PR_{var}_{pragma_type}'
        pragma_size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.cfg.define_knob(name, [i for i in pragma_size])
        
        for i in pragma_size:
            if self.cfg[name].val == i:
                self.sch[self.tensor].pragma(self.axis[var], pragma_type, self.cfg[name].val)

    def PR_fixed(self, list_pragma):
        '''
            PR: PragmaStep with fixed values
        '''
        var, pragma_type, size = list_pragma
        assert var < len(self.axis)
        self.sch[self.tensor].pragma(self.axis[var], pragma_type, size)

    def FSP(self):
        '''
            FSP: FollowSplitStep
            stage_id, iter_id, src_step_id, n_split
        '''
        pass

    def FSP_fixed(self, list_FSP):
        '''
            FSP: FollowSplitStep
            stage_id, iter_id, src_step_id, n_split

            ['FSP', 3, 0, 1, 1], 
            ['FSP', 3, 2, 2, 1], 
            ['RE', 3, [0, 2, 1, 3]], 
        '''
        stage_id = list_FSP[0]
        iter_id = list_FSP[1]
        src_step_id = list_FSP[2]
        n_split = list_FSP[3]

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
                    if i == n_split-1:
                        insert(order, [x0, y0], src_step_id)
                    else:
                        insert(order, [x0], src_step_id) 
                    yp = y0
                else:
                    x, y = self.sch[self.tensor].split(yp, self.cfg[name].val)
                    if i == n_split-1:
                        insert(order, [x, y], src_step_id)
                    else:
                        insert(order, [x], src_step_id)
                    yp = y
        self.axis = order # update the tensor's axis 

    def FFSP(self):
        '''
            FFSP: FollowFusedSplitStep
        '''
        pass

    def SA(self):
        '''
            SA: StorageAlignStep
        '''
        pass

    def CA(self):
        '''
            CA: ComputeAtStep
        '''
        pass

    def CI(self):
        '''
            CI: ComputeInlineStep
        '''
        pass

    def CR(self):
        '''
            CR: ComputeRootStep
        '''
        pass

    def CHR(self):
        '''
            CHR: CacheReadStep
        '''
        pass

    def RF(self):
        '''
            RF: RfactorStep
        '''
        pass
