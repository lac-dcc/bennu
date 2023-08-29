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
    ([0.023814, 0.0269994, 0.032609], [[], 
    [['CHW', 2, 'local'], ['SP', 2, 0, 1000, [20, 1, 2], 1], ['SP', 2, 4, 700, [1, 700, 1], 1], ['SP', 2, 8, 800, [5], 1], ['RE', 2, [0, 4, 1, 5, 8, 2, 6, 9, 3, 7]], ['FSP', 3, 0, 1, 2], ['FSP', 3, 3, 2, 2], ['RE', 3, [0, 3, 1, 4, 2, 5]], ['CA', 2, 3, 3], ['FU', 3, [0, 1, 2]], ['AN', 3, 0, 3], ['PR', 2, 0, 'auto_unroll_max_step$512'], ['AN', 2, 9, 2]]])
'''

import tvm
from tvm import autotvm

class Template_autotvm():

    cfg = None
    sch = None
    tensor = None
    args = [None]
    search_space = [1, 2, 4, 8, 16, 32, 46, 64]
    axis = None
    order = []


    def __init__(self, s, t, c, a) -> None:
        self.sch = s
        self.tensor = t
        self.cfg = c
        self.args = a

        self.axis = []
        for t in self.sch[self.tensor].op.axis:
            self.axis.append(t)
        for t in self.sch[self.tensor].op.reduce_axis:
            self.axis.append(t)
        print(self.axis)

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
        print(tvm.lower(self.sch, self.args))

    
    def RE(self):
        '''
            RE: ReorderStep
        ''' 
        name = f'RE_0'
        self.cfg.define_knob(name, [0, 1, 2, 3])

        print(len(self.order))

        '''
        x0, x1, re, x2, y2 = self.order
        if self.cfg[name].val == 0:
            self.sch[self.tensor].reorder(x0, x1, re, x2, y2)
        elif self.cfg[name].val == 1:
            self.sch[self.tensor].reorder(x0, x1, re, x2, y2)
        '''


    def SP(self, iter_id, lengths):
        '''
            SP: SplitStep
        '''
        for i in range(lengths):
            name = f'SP_{iter_id}_{i}'
            self.cfg.define_knob(name, self.search_space)

        # schedule according to config
        if lengths == 3:
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
        elif lengths == 1:
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

