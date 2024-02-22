import sys, os, logging
import heapq

# from tvm.auto_scheduler.space import Space
from tvm.auto_scheduler.search_task import SearchTask

# from tvm.autotvm.tuner.xgboost_tuner import XGBTuner
# from tvm.autotvm.tuner.sa_model_optimizer import SimulatedAnnealingOptimizer
from tvm.autotvm.tuner.model_based_tuner import ModelOptimizer

# from tvm.autotvm.tuner.xgboost_cost_model import XGBoostCostModel
from tvm.autotvm.tuner.model_based_tuner import submodular_pick, FeatureCache
from tvm.autotvm.tuner.xgboost_cost_model import (
    CustomCallback,
    xgb_average_recalln_curve_score,
)
from tvm.autotvm import feature
from tvm.autotvm.env import GLOBAL_SCOPE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import os
from src.utils import *
from src.space import Space

logger = logging.getLogger("autotvm")
xgb = None

from tvm.contrib.popen_pool import PopenPoolExecutor, StatusKind


class XGBSearch:
    def __init__(
        self,
        json_file,
        target,
        log,
        plan_size=64,
        feature_type="itervar",
        loss_type="reg",
        num_threads=None,
        optimizer="sa",
        diversity_filter_ratio=None,
        log_interval=50,
    ):

        # added variables
        workload_key = json_file["i"][0][0]
        self.task = SearchTask(workload_key=workload_key, target=target)
        self.space = Space(json_file, self.task, True)
        self.final_log = log
        self.batch = max(16, os.cpu_count())
        self.count = 0

        # if doesn't have opt just copy the solution from Ansor template
        if self.space.total_dims == 1:
            self.final_log = write_file([json_file], log)

        self.pool = None

        cost_model = XGBoostCostModel(
            self.task,
            feature_type=feature_type,
            loss_type=loss_type,
            num_threads=num_threads,
            log_interval=log_interval // 2,
            space=self.space,
        )
        if optimizer == "sa":
            optimizer = SimulatedAnnealingOptimizer(
                self.task, space=self.space, log_interval=log_interval
            )
        else:
            assert isinstance(optimizer, ModelOptimizer), (
                "Optimizer must be "
                "a supported name string"
                "or a ModelOptimizer object."
            )

        # super(XGBTuner, self).__init__(
        #    self.task, cost_model, optimizer, plan_size, diversity_filter_ratio
        # )
        # space
        self.target = target
        self.plan_size = plan_size

        self.cost_model = cost_model
        self.model_optimizer = optimizer
        self.diversity_filter_ratio = diversity_filter_ratio

        if self.diversity_filter_ratio:
            assert self.diversity_filter_ratio >= 1, (
                "Diversity filter ratio " "must be larger than one"
            )

        # trial plan
        self.next = []
        self.trial_pt = 0
        self.visited = set()

        # observed samples
        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.train_ct = 0

    def next_batch(self, batch_size):
        ret, index_list = [], []
        while len(ret) < batch_size and self.has_next():
            while self.trial_pt < len(self.next):
                index = self.next[self.trial_pt]
                if index not in self.visited and self.space.is_index_valid(index):
                    break
                self.trial_pt += 1

            if self.trial_pt >= len(self.next) - int(0.05 * self.plan_size):
                # if the trial list is empty or
                # the tuner is doing the last 5% trials (e-greedy), choose randomly
                index = self.space.get_rand_index(to_exclude=self.visited)
            ret.append(self.space.apply_opt(self.space.point2knob(index)))
            index_list.append(index)
            self.visited.add(index)
            self.count += 1
        log = write_file(ret)
        return self.space.run(log, self.final_log, index_list=index_list)

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            # index = inp.config.index
            index = inp.index
            if res.error_no == 0:
                self.xs.append(index)
                # flops = inp.task.flop / np.mean(res.costs)
                flops = 1.0 / np.mean(res.costs)
                self.flops_max = max(self.flops_max, flops)
                self.ys.append(flops)
            else:
                self.xs.append(index)
                self.ys.append(0.0)
            # Usually the update function is called during the tune loop
            # after the index is already added to the visited set.
            # However, adding the index to visited again here enables us
            # to also use this update function to resume tuning progress in
            # case of interruption.
            assert self.space.is_index_valid(index)
            self.visited.add(index)
        # if we have enough new training samples
        if (
            len(self.xs) >= self.plan_size * (self.train_ct + 1)
            and self.flops_max > 1e-6
        ):
            self.cost_model.fit(self.xs, self.ys, self.plan_size)
            if self.diversity_filter_ratio:
                candidate = self.model_optimizer.find_maximums(
                    self.cost_model,
                    self.plan_size * self.diversity_filter_ratio,
                    self.visited,
                )
                scores = self.cost_model.predict(candidate)
                knobs = [self.space.point2knob(x) for x in candidate]
                pick_index = submodular_pick(
                    0 * scores, knobs, self.plan_size, knob_weight=1
                )
                maximums = np.array(candidate)[pick_index]
            else:
                maximums = self.model_optimizer.find_maximums(
                    self.cost_model, self.plan_size, self.visited
                )

            self.next = maximums
            self.trial_pt = 0
            self.train_ct += 1

    def load_history(self, data_set, min_seed_records=500):
        # set in_tuning as True to make the feature extraction consistent
        GLOBAL_SCOPE.in_tuning = True

        # fit base model
        base_model = self.cost_model.spawn_base_model()
        success = base_model.fit_log(data_set, self.plan_size, min_seed_records)

        if not success:
            GLOBAL_SCOPE.in_tuning = False
            return

        # use base model to select initial points
        if not self.next:
            # no plan yet, use base model to select initial trials
            maximums = self.model_optimizer.find_maximums(
                base_model, self.plan_size, self.visited
            )
            self.next = maximums
            self.trial_pt = 0

        self.cost_model.load_basemodel(base_model)
        GLOBAL_SCOPE.in_tuning = False

    def has_next(self):
        return len(self.visited) < len(self.space) and self.count < self.trials

    def tune(
        self,
        n_trial=100,
        measure_option=None,
        early_stopping=None,
        callbacks=(),
        si_prefix="G",
    ):
        self.trials = n_trial
        while self.has_next():
            inp, res = self.next_batch(self.batch)
            self.update(inp, res)


class SimulatedAnnealingOptimizer:
    """parallel simulated annealing optimization algorithm

    Parameters
    ----------
    task: Task
        The tuning task
    n_iter: int
        The number of iterations of simulated annealing
    temp: float or Array of float
        If is a single float, then use a constant temperature.
        If is an Array, then perform linear cooling from temp[0] to temp[1]
    early_stop: int, optional
        Stop iteration if the optimal set do not change in `early_stop` rounds
    log_interval: int, optional
        Print log every `log_interval` iterations
    """

    def __init__(
        self,
        task,
        space=None,
        n_iter=500,
        temp=(1, 0),
        persistent=True,
        parallel_size=128,
        early_stop=50,
        log_interval=50,
    ):
        super(SimulatedAnnealingOptimizer, self).__init__()
        self.task = task
        self.space = space
        self.n_iter = n_iter
        self.temp = temp
        self.persistent = persistent
        self.parallel_size = min(parallel_size, len(self.space))
        self.early_stop = early_stop or 1e9
        self.log_interval = log_interval
        self.points = None

    def find_maximums(self, model, num, exclusive):
        tic = time.time()
        temp, n_iter, early_stop, log_interval = (
            self.temp,
            self.n_iter,
            self.early_stop,
            self.log_interval,
        )

        if self.persistent and self.points is not None:
            points = self.points
        else:
            points = self.space.sample_ints(self.parallel_size)

        scores = model.predict(points)

        # build heap and insert initial points
        heap_items = [(float("-inf"), -1 - i) for i in range(num)]
        heapq.heapify(heap_items)
        in_heap = set(exclusive)
        in_heap.update([x[1] for x in heap_items])

        for s, p in zip(scores, points):
            if s > heap_items[0][0] and p not in in_heap:
                pop = heapq.heapreplace(heap_items, (s, p))
                in_heap.remove(pop[1])
                in_heap.add(p)

        k = 0
        k_last_modify = 0

        if isinstance(temp, (tuple, list, np.ndarray)):
            t = temp[0]
            cool = 1.0 * (temp[0] - temp[1]) / (n_iter + 1)
        else:
            t = temp
            cool = 0

        while k < n_iter and k < k_last_modify + early_stop:
            new_points = np.empty_like(points)
            for i, p in enumerate(points):
                new_points[i] = self.task.config_space.random_walk(p)

            new_scores = model.predict(new_points)

            ac_prob = np.exp(np.minimum((new_scores - scores) / (t + 1e-5), 1))
            ac_index = np.random.random(len(ac_prob)) < ac_prob

            points[ac_index] = new_points[ac_index]
            scores[ac_index] = new_scores[ac_index]

            for s, p in zip(new_scores, new_points):
                if s > heap_items[0][0] and p not in in_heap:
                    pop = heapq.heapreplace(heap_items, (s, p))
                    in_heap.remove(pop[1])
                    in_heap.add(p)
                    k_last_modify = k

            k += 1
            t -= cool

            if log_interval and k % log_interval == 0:
                t_str = f"{t:.2f}"
                logger.debug(
                    "SA iter: %d\tlast_update: %d\tmax-0: %.2f\tmax-1: %.2f\ttemp: %s\t"
                    "elapsed: %.2f",
                    k,
                    k_last_modify,
                    heap_items[0][0],
                    np.max([v for v, _ in heap_items]),
                    t_str,
                    time.time() - tic,
                )

        heap_items.sort(key=lambda item: -item[0])
        heap_items = [x for x in heap_items if x[0] >= 0]
        logger.debug(
            "SA iter: %d\tlast_update: %d\telapsed: %.2f",
            k,
            k_last_modify,
            time.time() - tic,
        )
        logger.debug("SA Maximums: %s", heap_items)

        if self.persistent:
            self.points = points

        return [x[1] for x in heap_items]


# Global variables for passing arguments to extract functions.
_extract_space = None
_extract_target = None
_extract_task = None


def _extract_popen_initializer(space, target, task):
    global _extract_space, _extract_target, _extract_task
    _extract_space = space
    _extract_target = target
    _extract_task = task


def _extract_itervar_feature_index(args):
    """extract iteration var feature for an index in extract_space"""
    config = _extract_space.get(args)
    with _extract_target:
        sch, fargs = _extract_task.instantiate(config)

    fea = feature.get_itervar_feature_flatten(sch, fargs, take_log=True)
    fea = np.concatenate((fea, list(config.get_other_option().values())))
    return fea


def _extract_itervar_feature_log(arg):
    """extract iteration var feature for log items"""
    inp, res = arg
    config = inp.config
    with inp.target:
        sch, args = inp.task.instantiate(config)
    fea = feature.get_itervar_feature_flatten(sch, args, take_log=True)
    x = np.concatenate((fea, list(config.get_other_option().values())))

    if res.error_no == 0:
        y = inp.task.flop / np.mean(res.costs)
    else:
        y = 0.0
    return x, y


def _extract_knob_feature_index(args):
    """extract knob feature for an index in extract_space"""
    config = _extract_space.get(args)

    return config.get_flatten_feature()


def _extract_knob_feature_log(arg):
    """extract knob feature for log items"""
    inp, res = arg
    config = inp.config
    x = config.get_flatten_feature()

    if res.error_no == 0:
        with inp.target:  # necessary, for calculating flops of this task
            inp.task.instantiate(config)
        y = inp.task.flop / np.mean(res.costs)
    else:
        y = 0.0
    return x, y


def _extract_curve_feature_index(args):
    """extract sampled curve feature for an index in extract_space"""
    config = _extract_space.get(args)
    with _extract_target:
        sch, fargs = _extract_task.instantiate(config)

    fea = feature.get_buffer_curve_sample_flatten(sch, fargs, sample_n=20)
    fea = np.concatenate((fea, list(config.get_other_option().values())))
    return np.array(fea)


def _extract_curve_feature_log(arg):
    """extract sampled curve feature for log items"""
    inp, res = arg
    config = inp.config
    with inp.target:
        sch, args = inp.task.instantiate(config)
    fea = feature.get_buffer_curve_sample_flatten(sch, args, sample_n=20)
    x = np.concatenate((fea, list(config.get_other_option().values())))

    if res.error_no == 0:
        y = inp.task.flop / np.mean(res.costs)
    else:
        y = 0.0
    return x, y


def _binarize_evals(evals):
    """binarize evaluation labels"""
    bin_evals = []
    for evalset in evals:
        # binarize labels in xgb.dmatrix copy
        barray = evalset[0].get_data().copy()
        blabel = evalset[0].get_label().copy()
        blabel[blabel < 0.5] = 0.0
        blabel[blabel >= 0.5] = 1.0
        # pylint: disable=R1721
        bin_evals.append(
            tuple([xgb.DMatrix(barray, blabel)] + [e for e in evalset[1:]])
        )
    return bin_evals


class XGBoostCostModel:
    """XGBoost as cost model

    Parameters
    ----------
    task: Task
        The tuning task
    feature_type: str, optional
        If is 'itervar', use features extracted from IterVar (loop variable).
        If is 'knob', use flatten ConfigEntity directly.
        If is 'curve', use sampled curve feature (relation feature).

        Note on choosing feature type:
        For single task tuning, 'itervar' and 'knob' are good.
                                'itervar' is more accurate but 'knob' is much faster.
                                There are some constraints on 'itervar', if you meet
                                problems with feature extraction when using 'itervar',
                                you can switch to 'knob'.

        For cross-shape tuning (e.g. many convolutions with different shapes),
                               'itervar' and 'curve' has better transferability,
                               'knob' is faster.
        For cross-device or cross-operator tuning, you can use 'curve' only.
    loss_type: str
        If is 'reg', use regression loss to train cost model.
                     The cost model predicts the normalized flops.
        If is 'rank', use pairwise rank loss to train cost model.
                     The cost model predicts relative rank score.
        If is 'rank-binary', use pairwise rank loss with binarized labels to train cost model.
                     The cost model predicts relative rank score.
    num_threads: int, optional
        The number of threads.
    log_interval: int, optional
        If is not none, the cost model will print training log every `log_interval` iterations.
    upper_model: XGBoostCostModel, optional
        The upper model used in transfer learning
    """

    def __init__(
        self,
        task,
        feature_type,
        space=None,
        loss_type="reg",
        num_threads=None,
        log_interval=25,
        upper_model=None,
    ):
        global xgb
        self.pool = None
        super(XGBoostCostModel, self).__init__()
        try:
            if xgb is None:
                xgb = __import__("xgboost")
        except ImportError:
            raise ImportError(
                "XGBoost is required for XGBoostCostModel. "
                "Please install its python package first. "
                "Help: (https://xgboost.readthedocs.io/en/latest/) "
            )

        self.task = task
        self.target = task.target
        self.space = space

        self.fea_type = feature_type
        self.loss_type = loss_type
        self.num_threads = num_threads
        self.log_interval = log_interval

        self.loss_type = loss_type

        if loss_type == "reg":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.00,
                "alpha": 0,
                "objective": "reg:linear",
            }
        elif loss_type in ("rank", "rank-binary"):
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.00,
                "alpha": 0,
                "objective": "rank:pairwise",
            }
        else:
            raise RuntimeError("Invalid loss type: " + loss_type)

        self.xgb_params["verbosity"] = 0
        if num_threads:
            self.xgb_params["nthread"] = num_threads
        self.bst = None

        if feature_type == "itervar":
            self.feature_extract_func = _extract_itervar_feature_index
        elif feature_type == "knob":
            self.feature_extract_func = _extract_knob_feature_index
        elif feature_type == "curve":
            self.feature_extract_func = _extract_curve_feature_index
        else:
            raise RuntimeError("Invalid feature type " + feature_type)

        if upper_model:  # share a same feature cache with upper model
            self.feature_cache = upper_model.feature_cache
        else:
            self.feature_cache = FeatureCache()
        self.upper_model = upper_model
        self.feature_extra_ct = 0
        self.pool = None
        self.base_model = None

        self._sample_size = 0
        self._reset_pool(self.space, self.target, self.task)

    def _reset_pool(self, space, target, task):
        """reset processing pool for feature extraction"""

        if self.upper_model:  # base model will reuse upper model's pool,
            self.upper_model._reset_pool(space, target, task)
            return

        self._close_pool()

        self.pool = PopenPoolExecutor(
            max_workers=self.num_threads,
            initializer=_extract_popen_initializer,
            initargs=(space, target, task),
        )

    def _close_pool(self):
        if self.pool:
            self.pool = None

    def _get_pool(self):
        if self.upper_model:
            return self.upper_model._get_pool()
        return self.pool

    def _base_model_discount(self):
        return 1.0 / (2 ** (self._sample_size / 64.0))

    def fit(self, xs, ys, plan_size):
        tic = time.time()
        self._reset_pool(self.space, self.target, self.task)

        x_train = self._get_feature(xs)
        y_train = np.array(ys)
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-8)

        valid_index = y_train > 1e-6
        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])
        self._sample_size = len(x_train)

        if self.base_model:
            discount = self._base_model_discount()
            if discount < 0.05:  # discard base model
                self.base_model.upper_model = None
                self.base_model = None
            else:
                dtrain.set_base_margin(
                    discount * self.base_model.predict(xs, output_margin=True)
                )

        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=8000,
            callbacks=[
                CustomCallback(
                    stopping_rounds=20,
                    metric=f"tr-a-recall@{plan_size}",
                    evals=[(dtrain, "tr")],
                    maximize=True,
                    fevals=[xgb_average_recalln_curve_score(plan_size)],
                    verbose_eval=self.log_interval,
                    loss_type=self.loss_type,
                )
            ],
        )

        logger.debug(
            "XGB train: %.2f\tobs: %d\terror: %d\tn_cache: %d",
            time.time() - tic,
            len(xs),
            len(xs) - np.sum(valid_index),
            self.feature_cache.size(self.fea_type),
        )

    def fit_log(self, records, plan_size, min_seed_records=500):
        tic = time.time()

        # filter data, only pick the data with a same task
        data = []
        for inp, res in records:
            if inp.task.name == self.task.name:
                data.append((inp, res))

        logger.debug("XGB load %d entries from history log file", len(data))

        # extract feature
        self._reset_pool(self.space, self.target, self.task)
        pool = self._get_pool()
        if self.fea_type == "itervar":
            feature_extract_func = _extract_itervar_feature_log
        elif self.fea_type == "knob":
            feature_extract_func = _extract_knob_feature_log
        elif self.fea_type == "curve":
            feature_extract_func = _extract_curve_feature_log
        else:
            raise RuntimeError("Invalid feature type: " + self.fea_type)
        result = pool.map_with_error_catching(feature_extract_func, data)
        result = list(result)  # store results so we can iterate through them twice

        # get maximum feature length
        fea_len = -1
        for res in result:
            if res.status != StatusKind.COMPLETE:
                continue
            x, _ = res.value
            fea_len = max(fea_len, x.shape[0])

        xs, ys = [], []
        for res in result:
            if res.status != StatusKind.COMPLETE:
                continue
            x, y = res.value
            # Features may not be the same size, pad them until they are
            if fea_len > len(x):
                xs.append(np.pad(x, (0, fea_len - len(x))))
            else:
                xs.append(x)
            ys.append(y)

        if len(xs) < min_seed_records:  # no enough samples
            return False

        xs, ys = np.array(xs), np.array(ys)
        x_train = xs
        y_train = ys
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-8)

        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])

        plan_size *= 2
        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=400,
            callbacks=[
                CustomCallback(
                    stopping_rounds=100,
                    metric=f"tr-a-recall@{plan_size}",
                    evals=[(dtrain, "tr")],
                    maximize=True,
                    fevals=[xgb_average_recalln_curve_score(plan_size)],
                    verbose_eval=self.log_interval,
                    loss_type=self.loss_type,
                )
            ],
        )

        logger.debug("XGB train: %.2f\tobs: %d", time.time() - tic, len(xs))

        return True

    def predict(self, xs, output_margin=False):
        feas = self._get_feature(xs)
        dtest = xgb.DMatrix(feas)

        if self.base_model:
            dtest.set_base_margin(
                self._base_model_discount()
                * self.base_model.predict(xs, output_margin=True)
            )

        return self.bst.predict(dtest, output_margin=output_margin)

    def load_basemodel(self, base_model):
        self.base_model = base_model
        self.base_model._close_pool()
        self.base_model.upper_model = self

    def spawn_base_model(self):
        return XGBoostCostModel(
            self.task,
            self.fea_type,
            self.loss_type,
            self.num_threads,
            self.log_interval,
            self,
        )

    def _get_feature(self, indexes):
        """get features for indexes, run extraction if we do not have cache for them"""
        # free feature cache
        if self.feature_cache.size(self.fea_type) >= 100000:
            self.feature_cache.clear(self.fea_type)

        fea_cache = self.feature_cache.get(self.fea_type)

        indexes = np.array(indexes)
        need_extract = [x for x in indexes if x not in fea_cache]

        if need_extract:
            pool = self._get_pool()
            feas = pool.map_with_error_catching(self.feature_extract_func, need_extract)
            for i, fea in zip(need_extract, feas):
                fea_cache[i] = fea.value if fea.status == StatusKind.COMPLETE else None

        feature_len = -1
        for idx in indexes:
            if fea_cache[idx] is not None:
                feature_len = max(fea_cache[idx].shape[-1], feature_len)

        ret = np.empty((len(indexes), feature_len), dtype=np.float32)
        for i, ii in enumerate(indexes):
            t = fea_cache[ii]
            if t is not None and t.shape[0] < feature_len:
                t = np.pad(t, (0, feature_len - t.shape[0]))
            ret[i, :] = t if t is not None else 0
        return ret

    def __del__(self):
        self._close_pool()
