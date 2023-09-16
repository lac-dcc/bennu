from src.module.creating_template import Template_autotvm

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

def Template_factory(cfg, s, tensors, args):
    print(cfg)
    ta = Template_autotvm(s, tensors, args)
    list_SP = []
    for i in range(len(cfg)):
        field = cfg[i]
        if field[0] == 'SP':
            list_SP.append(len(field[4]))
            if i == len(cfg)-1 or cfg[i+1][0] != 'SP':
                ta.SP(list_SP)
        elif field[0] == 'RE':
            ta.RE_fixed(field[2])
        elif field[0] == 'FU':
            ta.FU_fixed(field[2])
        elif field[0] == 'PR':
            var = field[2]
            pragma = field[3].split("$")[0]
            size_pragma = int(field[3].split("$")[1])
            ta.PR_fixed([var, pragma, size_pragma])
        else:
            print(f'Method {field[0]} not implemented!')
        # TODO: Complete with other parameters
    return ta.ret()