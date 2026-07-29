import functools
import inspect
import warnings

import numpy as np
import pytest

import mojohyperopt as mh

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import hyperopt as upstream


class _NumPy2AnnealingAlgo(upstream.anneal.AnnealingAlgo):
    """Upstream 0.2.7 with its scalar conversion made NumPy 2 compatible."""

    def choose_ltv(self, label, size):
        tids = self.node_tids[label]
        vals = self.node_vals[label]
        losses = [self.tid_losses_dct[tid] for tid in tids]
        if size == 1:
            tid_set = set(tids)
            for tid in self.best_tids:
                if tid in tid_set:
                    index = tids.index(tid)
                    return losses[index], tid, vals[index]
        good_index = self.rng.geometric(1.0 / self.avg_best_idx, size=size) - 1
        good_index = np.clip(good_index, 0, len(tids) - 1).astype("int32")
        picks = np.argsort(losses)[good_index]
        picked_loss = np.asarray(losses)[picks]
        picked_tids = np.asarray(tids)[picks]
        picked_vals = np.asarray(vals)[picks]
        if size == 1:
            self.best_tids.append(int(picked_tids.item()))
        return picked_loss, picked_tids, picked_vals


def _upstream_anneal_numpy2(new_ids, domain, trials, seed, *args, **kwargs):
    (new_id,) = new_ids
    return _NumPy2AnnealingAlgo(domain, trials, seed, *args, **kwargs)(new_id)


def test_public_signatures_match_upstream():
    assert inspect.signature(mh.fmin) == inspect.signature(upstream.fmin)
    assert inspect.signature(mh.tpe.suggest) == inspect.signature(upstream.tpe.suggest)
    assert inspect.signature(mh.anneal.suggest) == inspect.signature(
        upstream.anneal.suggest
    )


def test_space_eval_nested_choice_and_constants():
    space = {
        "model": mh.hp.choice(
            "kind",
            [
                {"name": "linear", "alpha": mh.hp.loguniform("alpha", -4, 1)},
                {
                    "name": "tree",
                    "depth": mh.hp.randint("depth", 2, 9),
                    "leaf": [mh.hp.quniform("leaf", 1, 20, 1), "fixed"],
                },
            ],
        ),
        "flag": True,
    }
    assignment = {"kind": 1, "depth": 6, "leaf": 8.0}
    assert mh.space_eval(space, assignment) == {
        "model": {"name": "tree", "depth": 6, "leaf": [8.0, "fixed"]},
        "flag": True,
    }


def test_duplicate_labels_are_rejected():
    space = {
        "x": mh.hp.uniform("same", 0, 1),
        "y": mh.hp.uniform("same", 0, 1),
    }
    with pytest.raises(ValueError, match="duplicate label"):
        mh.fmin(lambda value: 0, space, max_evals=1, verbose=False)


def test_trials_properties_and_scalar_objective():
    trials = mh.Trials()
    argmin = mh.fmin(
        lambda x: (x - 1.5) ** 2,
        mh.hp.uniform("x", -5, 5),
        algo=mh.tpe.suggest,
        max_evals=45,
        trials=trials,
        rstate=np.random.default_rng(3),
        verbose=False,
    )
    assert len(trials) == 45
    assert set(argmin) == {"x"}
    assert trials.best_trial["result"]["loss"] == min(trials.losses())
    assert trials.statuses() == [mh.STATUS_OK] * 45
    assert len(trials.vals["x"]) == 45
    assert set(trials.best_trial["misc"]) >= {"idxs", "vals", "tid"}


def test_objective_result_dictionary_and_return_value():
    space = {"x": mh.hp.uniform("x", -2, 2)}
    trials = mh.Trials()
    value = mh.fmin(
        lambda point: {
            "loss": (point["x"] - 0.25) ** 2,
            "status": mh.STATUS_OK,
            "metric": abs(point["x"]),
        },
        space,
        algo=mh.anneal.suggest,
        max_evals=20,
        trials=trials,
        rstate=np.random.default_rng(7),
        return_argmin=False,
        verbose=False,
    )
    assert isinstance(value["x"], float)
    assert trials.best_trial["result"]["metric"] >= 0
    assert value == mh.space_eval(space, trials.argmin)


def test_points_to_evaluate_are_run_first():
    seen = []
    trials = mh.generate_trials_to_calculate([{"x": -1.0}, {"x": 2.0}])
    mh.fmin(
        lambda x: seen.append(x) or x * x,
        mh.hp.uniform("x", -5, 5),
        max_evals=2,
        trials=trials,
        verbose=False,
    )
    assert seen == [-1.0, 2.0]
    assert trials.losses() == [1.0, 4.0]


def test_failure_can_be_caught_and_search_continues():
    calls = 0

    def objective(x):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("first trial fails")
        return x * x

    trials = mh.Trials()
    mh.fmin(
        objective,
        mh.hp.uniform("x", -1, 1),
        max_evals=5,
        trials=trials,
        catch_eval_exceptions=True,
        verbose=False,
    )
    assert trials.trials[0]["state"] == mh.JOB_STATE_ERROR
    assert sum(result.get("status") == mh.STATUS_OK for result in trials.results) == 4


def test_early_stop_contract():
    def stop(trials, limit=0):
        limit += 1
        return limit >= 3, [limit]

    trials = mh.Trials()
    mh.fmin(
        lambda x: x * x,
        mh.hp.uniform("x", -1, 1),
        max_evals=20,
        trials=trials,
        early_stop_fn=stop,
        verbose=False,
    )
    assert len(trials) == 3


def test_all_standard_distribution_nodes_run():
    space = {
        "u": mh.hp.uniform("u", -1, 1),
        "ui": mh.hp.uniformint("ui", 1, 5),
        "qu": mh.hp.quniform("qu", -2, 2, 0.5),
        "lu": mh.hp.loguniform("lu", -3, 1),
        "qlu": mh.hp.qloguniform("qlu", -3, 1, 0.1),
        "n": mh.hp.normal("n", 0, 2),
        "qn": mh.hp.qnormal("qn", 0, 2, 0.25),
        "ln": mh.hp.lognormal("ln", 0, 1),
        "qln": mh.hp.qlognormal("qln", 0, 1, 0.1),
        "ri": mh.hp.randint("ri", 3, 9),
        "pc": mh.hp.pchoice("pc", [(0.2, "a"), (0.8, "b")]),
    }
    trials = mh.Trials()
    mh.fmin(
        lambda point: point["u"] ** 2,
        space,
        algo=mh.tpe.suggest,
        max_evals=25,
        trials=trials,
        rstate=np.random.default_rng(12),
        verbose=False,
    )
    values = mh.space_eval(space, trials.argmin)
    assert isinstance(values["ui"], int)
    assert values["qu"] * 2 == pytest.approx(round(values["qu"] * 2))
    assert 3 <= values["ri"] < 9
    assert values["pc"] in ("a", "b")


@pytest.mark.parametrize("algorithm", [mh.tpe.suggest, mh.anneal.suggest])
def test_conditional_spaces_only_record_active_branch(algorithm):
    space = mh.hp.choice(
        "kind",
        [
            {"kind": "left", "x": mh.hp.uniform("x", -2, 2)},
            {"kind": "right", "y": mh.hp.uniform("y", 4, 8)},
        ],
    )
    trials = mh.Trials()
    mh.fmin(
        lambda point: (point.get("x", point.get("y")) - 1) ** 2,
        space,
        algo=algorithm,
        max_evals=35,
        trials=trials,
        rstate=np.random.default_rng(18),
        verbose=False,
    )
    for trial in trials.trials:
        labels = set(trial["misc"]["vals"])
        assert ("x" in labels) != ("y" in labels)


def _run_ours(algorithm):
    space = {
        "x": mh.hp.uniform("x", -5, 5),
        "y": mh.hp.loguniform("y", -3, 2),
    }
    trials = mh.Trials()
    mh.fmin(
        lambda point: (point["x"] - 1.25) ** 2
        + (np.log(point["y"]) + 0.5) ** 2,
        space,
        algo=algorithm,
        max_evals=80,
        trials=trials,
        rstate=np.random.default_rng(22),
        verbose=False,
    )
    return trials.best_trial["result"]["loss"]


def _run_upstream(algorithm):
    space = {
        "x": upstream.hp.uniform("x", -5, 5),
        "y": upstream.hp.loguniform("y", -3, 2),
    }
    trials = upstream.Trials()
    upstream.fmin(
        lambda point: (point["x"] - 1.25) ** 2
        + (np.log(point["y"]) + 0.5) ** 2,
        space,
        algo=algorithm,
        max_evals=80,
        trials=trials,
        rstate=np.random.default_rng(22),
        verbose=False,
        show_progressbar=False,
    )
    return trials.best_trial["result"]["loss"]


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (mh.tpe.suggest, upstream.tpe.suggest),
        (mh.anneal.suggest, _upstream_anneal_numpy2),
    ],
)
def test_search_quality_parity_with_upstream(ours, theirs):
    our_loss = _run_ours(ours)
    upstream_loss = _run_upstream(theirs)
    assert our_loss < 0.2
    assert upstream_loss < 0.2


def test_partial_tpe_configuration_is_supported():
    configured = functools.partial(
        mh.tpe.suggest, n_startup_jobs=5, n_EI_candidates=40, gamma=0.4
    )
    trials = mh.Trials()
    mh.fmin(
        lambda x: (x + 0.75) ** 2,
        mh.hp.uniform("x", -3, 3),
        algo=configured,
        max_evals=30,
        trials=trials,
        rstate=np.random.default_rng(5),
        verbose=False,
    )
    assert trials.best_trial["result"]["loss"] < 0.02


def test_documented_anneal_batch_api_and_class():
    space = mh.hp.uniform("x", -1, 1)
    domain = mh.Domain(lambda x: x * x, space)
    trials = mh.Trials()
    algo = mh.anneal.AnnealingAlgo(domain, trials, seed=4)
    docs = algo.batch([0, 1])
    assert [doc["tid"] for doc in docs] == [0, 1]
    docs = mh.anneal.suggest_batch([2], domain, trials, seed=5)
    assert docs[0]["tid"] == 2


def test_documented_status_constants_and_trial_helpers():
    assert mh.STATUS_STRINGS == ("new", "running", "suspended", "ok", "fail")
    assert mh.JOB_STATES == (0, 1, 2, 3)
    doc = mh.generate_trial(9, {"x": 1.5})
    assert doc["tid"] == 9
    assert doc["misc"]["vals"] == {"x": [1.5]}
