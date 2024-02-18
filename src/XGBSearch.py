import sys, os

# from tvm.auto_scheduler.space import Space
from tvm.auto_scheduler.search_task import SearchTask
from tvm.autotvm.tuner.xgboost_tuner import XGBTuner
from tvm.autotvm.tuner.sa_model_optimizer import SimulatedAnnealingOptimizer
from tvm.autotvm.tuner.model_based_tuner import ModelOptimizer

from tvm.autotvm.tuner.xgboost_cost_model import XGBoostCostModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import os
from src.utils import *
from src.space import Space

class XGBSearch:
    
    def __init__(
        self,
        task,
        plan_size=64,
        feature_type="itervar",
        loss_type="reg",
        num_threads=None,
        optimizer="sa",
        diversity_filter_ratio=None,
        log_interval=50,
    ):
        cost_model = XGBoostCostModel(
            task,
            feature_type=feature_type,
            loss_type=loss_type,
            num_threads=num_threads,
            log_interval=log_interval // 2,
        )
        if optimizer == "sa":
            optimizer = SimulatedAnnealingOptimizer(task, log_interval=log_interval)
        else:
            assert isinstance(optimizer, ModelOptimizer), (
                "Optimizer must be " "a supported name string" "or a ModelOptimizer object."
            )

        super(XGBTuner, self).__init__(
            task, cost_model, optimizer, plan_size, diversity_filter_ratio
        )
    
    def tune():
        pass