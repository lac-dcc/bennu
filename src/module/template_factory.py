import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.module.creating_template import Template_autotvm


def Template_factory(cfg, tensors, args):
    """
    Build the template based on Auto Schedule
    """
    ta = Template_autotvm(tensors, args)
    for field in cfg:
        if field[0] == "SP":
            ta.SP(field[1:])
        elif field[0] == "RE":
            ta.RE_fixed(field[1:])
        elif field[0] == "FU":
            ta.FU_fixed(field[1:])
        elif field[0] == "PR":
            ta.PR_fixed(field[1:])
        elif field[0] == "AN":
            ta.AN(field[1:])
        elif field[0] == "FSP":
            ta.FSP_fixed(field[1:])
        elif field[0] == "FFSP":
            ta.FFSP_fixed(field[1:])
        elif field[0] == "SA":
            ta.SA(field[1:])
        elif field[0] == "CA":
            ta.CA_fixed(field[1:])
        elif field[0] == "CI":
            ta.CI(field[1:])
        elif field[0] == "CR":
            ta.CR(field[1:])
        elif field[0] == "CHR":
            ta.CHR(field[1:])
        elif field[0] == "CHW":
            ta.CHW(field[1:])
        elif field[0] == "RF":
            ta.RF(field[1:])
        else:
            raise RuntimeError(f"Invalid template type {field[0]}")
    return ta.ret()
