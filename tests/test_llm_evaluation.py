from types import SimpleNamespace

import torch

from lp_llm.evaluate import sliding_nll


def test_sliding_window_scores_each_token_once_after_first():
    seen = []

    class Model:
        def __call__(self, inputs, labels):
            seen.append(int((labels != -100).sum()))
            return SimpleNamespace(loss=torch.tensor(2.0))

    nll, count = sliding_nll(Model(), torch.arange(10).reshape(1, 10), 6, 2, "cpu")
    assert (nll, count) == (18.0, 9)
    assert seen == [6, 2, 2]
