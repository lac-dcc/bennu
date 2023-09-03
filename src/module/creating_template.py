'''
AN: AnnotationStep
FU: FuseStep
PR: PragmaStep
RE: ReorderStep

FSP: FollowSplitStep
FFSP: FollowFusedSplitStep
SA: StorageAlignStep
CA: ComputeAtStep
CI: ComputeInlineStep
CR: ComputeRootStep
CHR: CacheReadStep

RF: RfactorStep
'''

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
                #print(e, "inserted", list)
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
            self.tensor = self.sch.cache_write(self.tensor, 'local')
        

    def print(self):
        '''
            Print tensor function
        '''
        print(tvm.lower(self.sch, self.args, simple_mode=True))

    
    def RE(self, size_order):
        '''
            RE: ReorderStep
        ''' 
        if len(self.order) == 0:
            return
        name = f'RE_0'
        
        self.cfg.define_knob(name, [i for i in range(size_order)])

        #yo, xo, k, yi, xi = self.order
        #try:
        #self.sch[self.tensor].reorder(*self.order)
        #except:
        #    print("Received error")
        #    print(self.order, len(self.order))

        perms = permutation(self.order, size_order)
        for i, p in enumerate(perms):
            if self.cfg[name].val == i:
                self.sch[self.tensor].reorder(*p)

        '''
        x0, x1, re, x2, y2 = self.order
        if self.cfg[name].val == 0:
            self.sch[self.tensor].reorder(x0, x1, re, x2, y2)
        elif self.cfg[name].val == 1:
            self.sch[self.tensor].reorder(x0, x1, re, x2, y2)
        '''
        #self.order = []


    def SP(self, iter_id, split_size):
        '''
            SP: SplitStep
        '''

        '''
        for i in range(split_size):
            name = f'SP_{iter_id}_{i}'
            self.cfg.define_knob(name, self.search_space)

            if i == 0:
                x0, y0 = self.sch[self.tensor].split(self.axis[iter_id], self.cfg[name].val)
                print(x0, y0)
                if i == split_size-1:
                    self.order.append(x0)
                    self.order.append(y0)
                else:
                    self.order.append(x0)
                yp = y0
            else:
                x, y = self.sch[self.tensor].split(yp, self.cfg[name].val)
                if i == split_size-1:
                    self.order.append(x)
                    self.order.append(y)
                else:
                    self.order.append(x)
                yp = y
        print(self.order)
        '''
        self.cfg.define_knob("tile_y", [4, 8, 16])
        self.cfg.define_knob("tile_x", [4, 8, 16])

        y, x = self.sch[self.tensor].op.axis
        k = self.sch[self.tensor].op.reduce_axis[0]

        yo, yi = self.sch[self.tensor].split(y, self.cfg["tile_y"].val)
        xo, xi = self.sch[self.tensor].split(x, self.cfg["tile_x"].val)

        self.order = []
        self.add(self.order, [yo, xo, k, yi, xi])

        #self.order.append(yo)
        #self.order.append(xo)
        #self.order.append(k)
        #self.order.append(yi)
        #elf.order.append(xi)

        #self.sch[self.tensor].reorder(yo, xo, k, yi, xi)
        # schedule according to config
        '''
        if split_size == 3:
            name = f'SP_{iter_id}_0'
            x0, y0 = self.sch[self.tensor].split(self.axis[iter_id], self.cfg[name].val)
            name = f'SP_{iter_id}_1'
            x1, y1 = self.sch[self.tensor].split(y0, self.cfg[name].val)
            name = f'SP_{iter_id}_2'
            x2, y2 = self.sch[self.tensor].split(y1, self.cfg[name].val)

            reduce_axis = self.sch[self.tensor].op.reduce_axis

            # TODO: best order.
            self.order.append([x0, reduce_axis[0], x1, x2, y2])
            # this order is not the best, but get good result
            #if reduce_axis is None:
            #    self.order.append([x0, x1, x2, y2])
            #    #self.sch[t].reorder(x0, x1, x2, y2)
            #else:
            #    self.order.append([x0, reduce_axis[0], x1, x2, y2])
            #    #self.sch[t].reorder(x0, reduce_axis[0], x1, x2, y2)
        elif split_size == 1:
            name = f'SP_{iter_id}_0'
            x0, y0 = self.sch[self.tensor].split(self.axis[iter_id], self.cfg[name].val)
            # TODO: best order.
            # this order is not the best, but get good result
            #if reduce_axis is None:
            #    self.order.append([x0, y0])
            #    #self.sch[t].reorder(x0, y0)
            #else:
            #    self.order.append([x0, reduce_axis[0], y0])
            #    #self.sch[t].reorder(x0, reduce_axis[0], y0)
        '''
