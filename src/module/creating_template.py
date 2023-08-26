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
CHW: CacheWriteStep
RF: RfactorStep
'''

## example input

'''
    ([0.023814, 0.0269994, 0.032609], [[], 
    [['CHW', 2, 'local'], ['SP', 2, 0, 1000, [20, 1, 2], 1], ['SP', 2, 4, 700, [1, 700, 1], 1], ['SP', 2, 8, 800, [5], 1], ['RE', 2, [0, 4, 1, 5, 8, 2, 6, 9, 3, 7]], ['FSP', 3, 0, 1, 2], ['FSP', 3, 3, 2, 2], ['RE', 3, [0, 3, 1, 4, 2, 5]], ['CA', 2, 3, 3], ['FU', 3, [0, 1, 2]], ['AN', 3, 0, 3], ['PR', 2, 0, 'auto_unroll_max_step$512'], ['AN', 2, 9, 2]]])
'''

import tvm
from tvm import autotvm

class Template_ansor():

    cfg = None
    sch = None
    tensors = [None]
    args = [None]
    search_space = [1, 2, 4, 8, 16, 32, 46, 64]
    axis = None
    order = []


    def __init__(self, s, t, c, a) -> None:
        self.sch = s
        self.tensors = t
        self.cfg = c
        self.args = a
    
    def ret(self):
        return self.sch, self.args

    def space(self, values):
        type = values[0]
        if type == "SP":
            assert len(values) == 6
            stage_id = values[1]
            iter_id = values[2]
            loop_extent = values[3]
            lengths = values[4]
            inner_to_outer = values[5]
            self.SP(type, stage_id, iter_id, loop_extent, lengths, inner_to_outer)
        elif type == "CHW":
            assert len(values) == 3
            stage_id = values[1]
            scope_name = values[2]
            self.CHW(type, stage_id, scope_name)
        else:
            raise("Not implemented space search")

    def limited_interval(self, max_value, interval):
        new_interval = []
        for elem in interval:
            if max_value <= elem:
                continue
            new_interval.append(elem)
        return new_interval


    def CHW(self, type, stage_id, scope_name):
        name = type + "_" + str(stage_id)
        self.cfg.define_knob(name, [0, 1])
        if self.cfg[name].val != 0:
            self.tensors[stage_id] = self.sch.cache_write(self.tensors[stage_id], scope_name)


    def print(self):
        print(tvm.lower(self.sch, self.args))

    
    def RE(self, type, iter_id):
        '''
            RE: ReorderStep
        ''' 
        pass


    def SP(self, type, stage_id, iter_id, loop_extent, lengths, inner_to_outer):
        '''
            SP: SplitStep
            ("SP", stage_id, iter_id, loop_extent, lengths, inner_to_outer)
        '''
        t = self.tensors[stage_id]
        axis = self.sch[t].op.axis
        reduce_axis = self.sch[t].op.reduce_axis if len(self.sch[t].op.reduce_axis) > 0 else None

        print(axis)
        print(reduce_axis)

        for i in range(len(lengths)):
            name = type + "_%d_%d" % (iter_id, i)
            # define search space
            self.cfg.define_knob(name, self.search_space)

        # schedule according to config
        if len(lengths) == 3:
            name = type + "_%d_%d" % (iter_id, 0)
            x0, y0 = self.sch[t].split(axis[iter_id], self.cfg[name].val)
            name = type + "_%d_%d" % (iter_id, 1)
            x1, y1 = self.sch[t].split(y0, self.cfg[name].val)
            name = type + "_%d_%d" % (iter_id, 2)
            x2, y2 = self.sch[t].split(y1, self.cfg[name].val)
            # TODO: best order.
            # this order is not the best, but get good result
            if reduce_axis == None:
                self.order.append([x0, x1, x2, y2])
                #self.sch[t].reorder(x0, x1, x2, y2)
            else:
                self.order.append([x0, reduce_axis[0], x1, x2, y2])
                #self.sch[t].reorder(x0, reduce_axis[0], x1, x2, y2)
        elif len(lengths) == 1:
            name = type + "_%d_%d" % (iter_id, 0)
            x0, y0 = self.sch[t].split(axis[iter_id], self.cfg[name].val)
            # TODO: best order.
            # this order is not the best, but get good result
            if reduce_axis == None:
                self.sch[t].reorder(x0, y0)
            else:
                self.sch[t].reorder(x0, reduce_axis[0], y0)

