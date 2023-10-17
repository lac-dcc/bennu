from src.module.creating_template import Template_autotvm

'''
    Build the template based on Auto Schedule
'''
def Template_factory(cfg, tensors, args):
    ta = Template_autotvm(tensors, args)
    list_SP = []
    for i in range(len(cfg)):
        field = cfg[i]
        if field[0] == 'SP':
            list_SP.append(len(field[4]))
            if i == len(cfg)-1 or cfg[i+1][0] != 'SP':
                ta.SP(list_SP)
        elif field[0] == 'RE':
            ta.RE_fixed(field[1:])
        elif field[0] == 'FU':
            ta.FU_fixed(field[1:])
        elif field[0] == 'PR':
            ta.PR_fixed(field[1:])
        elif field[0] == 'AN':
            ta.AN(field[1:])
        elif field[0] == 'FSP':
            ta.FSP_fixed(field[1:])
        elif field[0] == 'FFSP':
            ta.FFSP_fixed(field[1:])
        elif field[0] == 'SA':
            ta.SA(field[1:])
        elif field[0] == 'CA':
            #TODO: Temporarily commented. I'm working on a fix for the crash issue.
            #ta.CA_fixed(field[1:])
            pass
        elif field[0] == 'CI':
            ta.CI(field[1:])
        elif field[0] == 'CR':
            ta.CR(field[1:])
        elif field[0] == 'CHR':
            ta.CHR(field[1:])
        elif field[0] == 'CHW':
            ta.CHW(field[1:])
        elif field[0] == 'RF':
            ta.RF(field[1:])
        else:
            raise RuntimeError(f"Invalid template type {field[0]}")
    return ta.ret()