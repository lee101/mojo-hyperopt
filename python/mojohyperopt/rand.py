"""Random-search startup sampler used by TPE."""

from __future__ import annotations

import numpy as np

from .base import trial_doc
from .space import sample_prior


def suggest(new_ids, domain, trials, seed):
    rng = np.random.default_rng(seed)
    docs = []
    for new_id in new_ids:
        _, raw = sample_prior(domain.space, rng)
        docs.append(trial_doc(int(new_id), raw))
    return docs
