## example input

'''
    Time spent: [0.0173736, 0.0208746, 0.0257531]
    Config: [[], 
    [['SP', 2, 0, 1000, [5, 25, 4], 1], ['SP', 2, 4, 700, [1, 35, 4], 1], ['SP', 2, 8, 800, [8], 1], 
    ['RE', 2, [0, 4, 1, 5, 8, 2, 6, 9, 3, 7]], 
    ['FU', 2, [0, 1, 2]], 
    ['AN', 2, 0, 3], 
    ['PR', 2, 0, 'auto_unroll_max_step$512'], 
    ['AN', 2, 7, 2]]]    
'''

import sys, os

import tvm
from tvm import autotvm
from src.module.utils import *

class Template_autotvm():

    cfg = None
    sch = None
    tensor = None
    args = []
    search_space = [4]
    axis = None
    order = []

    def __init__(self, s, t, a) -> None:
        self.sch = s
        self.tensor = t
        self.cfg = autotvm.get_config()
        self.args = a

        self.axis = []
        for t in self.sch[self.tensor].op.axis:
            self.axis.append(t)
        for t in self.sch[self.tensor].op.reduce_axis:
            self.axis.append(t)
        #print(self.axis)

    def add(self, list, elements):
        for e in elements:
            if e not in list:
                list.append(e)

    def ret(self):
        '''
            return function
        '''
        return self.sch, self.args
    

    def limited_interval(self, max_value, interval):
        new_interval = []
        for elem in interval:
            if max_value <= elem:
                continue
            new_interval.append(elem)
        return new_interval


    def CHW(self):
        '''
            CHW: CacheWriteStep
        '''
        name = f'CHW'
        self.cfg.define_knob(name, [0, 1])
        if self.cfg[name].val != 0:
            CC = self.sch.cache_write(self.tensor, 'local')
            #print(CC)
            #print("----")
            print(self.tensor)
        

    def print(self):
        '''
            Print tensor function
        '''
        print(tvm.lower(self.sch, self.args, simple_mode=True))

    def RE_fixed(self, list_order):
        '''
            RE_fixed: Reorder step with a list
        '''
        assert len(list_order) == len(self.order)
        p = []
        for ord in list_order:
            p.append(self.order[ord])
        self.sch[self.tensor].reorder(*p)

    def RE(self, size_order):
        '''
            RE: ReorderStep
        ''' 
        if len(self.order) == 0:
            return
        name = f'RE_0'
        
        self.cfg.define_knob(name, [i for i in range(size_order)])

        perms = permutation(self.order, size_order)
        for i, p in enumerate(perms):
            if self.cfg[name].val == i:
                self.sch[self.tensor].reorder(*p)


    def SP(self, list_iter_id):
        '''
            SP: SplitStep
        '''
        self.order = []
        for iter_id in range(len(list_iter_id)):
            split_size = list_iter_id[iter_id]-1
            if split_size == 0:
                k = self.axis[iter_id]
                self.add(self.order, [k])
            else:
                for i in range(split_size):
                    name = f'SP_{iter_id}_{i}'
                    self.cfg.define_knob(name, self.search_space)

                    if i == 0:
                        x0, y0 = self.sch[self.tensor].split(self.axis[iter_id], self.cfg[name].val)
                        if i == split_size-1:
                            self.add(self.order, [x0, y0])
                        else:
                            self.add(self.order, [x0])
                        yp = y0
                    else:
                        x, y = self.sch[self.tensor].split(yp, self.cfg[name].val)
                        if i == split_size-1:
                            self.add(self.order, [x, y])
                        else:
                            self.add(self.order, [x])
                        yp = y
        self.axis = self.order # update the tensor's axis 

def AN(self):
    '''
        AN: AnnotationStep
    '''
    pass

def FU(self):
    '''
        FU: FuseStep
    '''
    name = f'FU_0'
    size_fusion = len(self.axis)-1
    self.cfg.define_knob(name, [i for i in range(size_fusion)])

    for i in range(size_fusion+1):
        if cfg[name].val == i:
            self.sch[self.tensor].compute_at(self.axis[i], self.axis[i+1])

def FU_fixed(self, list_fuse):
    '''
        FU_fixed: Fuse step with a list
    '''
    for i in range(len(list_fuse)-1):
        fused = self.sch[self.tensor].fuse(self.axis[list_fuse[i]], self.axis[list_fuse[i+1]])
        # TODO: update self.axis

def PR(self):
    '''
        PR: PragmaStep
    '''
    pass

def FSP(self):
    '''
        FSP: FollowSplitStep
    '''
    pass

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
