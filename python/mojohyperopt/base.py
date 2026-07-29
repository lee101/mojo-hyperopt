"""In-memory trial records compatible with Hyperopt's covered public surface."""

from __future__ import annotations

from typing import Any

from .space import collect_distributions

STATUS_NEW = "new"
STATUS_RUNNING = "running"
STATUS_SUSPENDED = "suspended"
STATUS_OK = "ok"
STATUS_FAIL = "fail"
STATUS_STRINGS = (STATUS_NEW, STATUS_RUNNING, STATUS_SUSPENDED, STATUS_OK, STATUS_FAIL)

JOB_STATE_NEW = 0
JOB_STATE_RUNNING = 1
JOB_STATE_DONE = 2
JOB_STATE_ERROR = 3
JOB_STATES = (JOB_STATE_NEW, JOB_STATE_RUNNING, JOB_STATE_DONE, JOB_STATE_ERROR)


class Trials:
    def __init__(self, exp_key=None, refresh=True):
        self._trials: list[dict[str, Any]] = []
        self.attachments: dict[str, Any] = {}
        self.exp_key = exp_key

    @property
    def trials(self):
        return self._trials

    def __len__(self):
        return len(self._trials)

    def refresh(self):
        return None

    def insert_trial_docs(self, docs):
        self._trials.extend(docs)
        return docs

    def new_trial_ids(self, n):
        start = max((trial["tid"] for trial in self._trials), default=-1) + 1
        return list(range(start, start + n))

    @property
    def results(self):
        return [trial["result"] for trial in self._trials]

    @property
    def vals(self):
        labels: set[str] = set()
        for trial in self._trials:
            labels.update(trial["misc"]["vals"])
        return {
            label: [
                trial["misc"]["vals"][label][0]
                for trial in self._trials
                if trial["misc"]["vals"].get(label)
            ]
            for label in sorted(labels)
        }

    def losses(self):
        return [result.get("loss") for result in self.results]

    def statuses(self):
        return [result.get("status") for result in self.results]

    @property
    def best_trial(self):
        successful = [
            trial
            for trial in self._trials
            if trial["result"].get("status") == STATUS_OK
            and trial["result"].get("loss") is not None
        ]
        if not successful:
            raise ValueError("no successful trials")
        return min(successful, key=lambda trial: float(trial["result"]["loss"]))

    @property
    def argmin(self):
        return {
            label: values[0]
            for label, values in self.best_trial["misc"]["vals"].items()
            if values
        }


class Domain:
    def __init__(self, fn, space, pass_expr_memo_ctrl=None):
        if pass_expr_memo_ctrl:
            raise NotImplementedError("pass_expr_memo_ctrl is not supported")
        self.fn = fn
        self.space = space
        self.params = collect_distributions(space)

    def new_result(self):
        return {"status": STATUS_NEW}


def trial_doc(tid: int, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": JOB_STATE_NEW,
        "tid": tid,
        "spec": None,
        "result": {"status": STATUS_NEW},
        "misc": {
            "tid": tid,
            "cmd": ("domain_attachment", "FMinIter_Domain"),
            "workdir": None,
            "idxs": {label: [tid] for label in raw},
            "vals": {label: [value] for label, value in raw.items()},
        },
        "exp_key": None,
        "owner": None,
        "version": 0,
        "book_time": None,
        "refresh_time": None,
    }
