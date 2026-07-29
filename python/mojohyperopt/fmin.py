"""Sequential optimizer driver matching Hyperopt's common `fmin` contract."""

from __future__ import annotations

import os
import time

import numpy as np

from . import tpe
from .base import (
    JOB_STATE_DONE,
    JOB_STATE_ERROR,
    JOB_STATE_RUNNING,
    STATUS_FAIL,
    STATUS_OK,
    Domain,
    Trials,
    trial_doc,
)
from .space import evaluate


def space_eval(space, hp_assignment):
    return evaluate(space, hp_assignment)


def generate_trial(tid, space):
    return trial_doc(int(tid), dict(space))


def generate_trials_to_calculate(points):
    trials = Trials()
    trials.insert_trial_docs(
        [generate_trial(tid, point) for tid, point in enumerate(points)]
    )
    return trials


def _seed(rstate) -> int:
    if hasattr(rstate, "integers"):
        return int(rstate.integers(2**31 - 1))
    return int(rstate.randint(2**31 - 1))


def _result(value):
    if isinstance(value, dict):
        result = dict(value)
        status = result.get("status")
        if status not in (STATUS_OK, STATUS_FAIL):
            raise ValueError("objective result requires status 'ok' or 'fail'")
        if status == STATUS_OK and "loss" not in result:
            raise ValueError("successful objective result requires a loss")
        if "loss" in result:
            result["loss"] = float(result["loss"])
        return result
    return {"loss": float(value), "status": STATUS_OK}


def fmin(
    fn,
    space,
    algo=None,
    max_evals=None,
    timeout=None,
    loss_threshold=None,
    trials=None,
    rstate=None,
    allow_trials_fmin=True,
    pass_expr_memo_ctrl=None,
    catch_eval_exceptions=False,
    verbose=True,
    return_argmin=True,
    points_to_evaluate=None,
    max_queue_len=1,
    show_progressbar=True,
    early_stop_fn=None,
    trials_save_file="",
):
    if algo is None:
        algo = tpe.suggest
    if max_evals is None:
        max_evals = np.iinfo(np.int64).max
    if timeout is not None and timeout < 0:
        raise ValueError("timeout must be None or non-negative")
    if trials_save_file:
        raise NotImplementedError("trials_save_file is not supported")
    if pass_expr_memo_ctrl:
        raise NotImplementedError("pass_expr_memo_ctrl is not supported")
    if max_queue_len < 1:
        raise ValueError("max_queue_len must be positive")
    if trials is None:
        trials = (
            Trials()
            if points_to_evaluate is None
            else generate_trials_to_calculate(points_to_evaluate)
        )
    if allow_trials_fmin and type(trials) is not Trials and hasattr(trials, "fmin"):
        return trials.fmin(
            fn,
            space,
            algo=algo,
            max_evals=max_evals,
            timeout=timeout,
            loss_threshold=loss_threshold,
            rstate=rstate,
            verbose=verbose,
        )
    if rstate is None:
        environment_seed = os.environ.get("HYPEROPT_FMIN_SEED", "")
        rstate = np.random.default_rng(
            int(environment_seed) if environment_seed else None
        )

    domain = Domain(fn, space, pass_expr_memo_ctrl=pass_expr_memo_ctrl)
    start = time.perf_counter()
    early_stop_args = []

    for trial in list(trials.trials):
        if trial["state"] == 0 and len(trials) <= max_evals:
            raw = {
                label: values[0]
                for label, values in trial["misc"]["vals"].items()
                if values
            }
            trial["state"] = JOB_STATE_RUNNING
            try:
                trial["result"] = _result(fn(space_eval(space, raw)))
                trial["state"] = JOB_STATE_DONE
            except Exception as error:
                trial["state"] = JOB_STATE_ERROR
                trial["misc"]["error"] = (str(type(error)), str(error))
                if not catch_eval_exceptions:
                    raise

    while len(trials) < int(max_evals):
        if timeout is not None and time.perf_counter() - start >= timeout:
            break
        tid = trials.new_trial_ids(1)[0]
        docs = algo([tid], domain, trials, _seed(rstate))
        if not docs:
            break
        trial = docs[0]
        trials.insert_trial_docs([trial])
        raw = {
            label: values[0]
            for label, values in trial["misc"]["vals"].items()
            if values
        }
        trial["state"] = JOB_STATE_RUNNING
        try:
            trial["result"] = _result(fn(space_eval(space, raw)))
            trial["state"] = JOB_STATE_DONE
        except Exception as error:
            trial["state"] = JOB_STATE_ERROR
            trial["misc"]["error"] = (str(type(error)), str(error))
            if not catch_eval_exceptions:
                raise
        if (
            loss_threshold is not None
            and trial["result"].get("status") == STATUS_OK
            and trial["result"].get("loss", float("inf")) < loss_threshold
        ):
            break
        if early_stop_fn is not None:
            stop, early_stop_args = early_stop_fn(trials, *early_stop_args)
            if stop:
                break

    successful = [
        trial
        for trial in trials.trials
        if trial["result"].get("status") == STATUS_OK
    ]
    if not successful:
        if return_argmin:
            raise Exception("There are no successful evaluation tasks")
        return None
    return trials.argmin if return_argmin else space_eval(space, trials.argmin)
