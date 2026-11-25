import math

import pytest

from skillpath.eval.metrics import f1_at_k, ndcg_at_k, precision_at_k, recall_at_k

RANKED = ["a", "b", "c", "d"]
REL = {"a": 2, "c": 1, "e": 2}  # "e" is relevant but never retrieved


def test_precision_and_recall():
    assert precision_at_k(RANKED, REL, 2) == 0.5
    assert precision_at_k(RANKED, REL, 4) == 0.5
    assert recall_at_k(RANKED, REL, 4) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, REL, 4, threshold=2) == 0.5   # only a and e count
    assert recall_at_k(RANKED, {}, 4) == 0.0


def test_f1():
    p, r = 0.5, 2 / 3
    assert f1_at_k(RANKED, REL, 4) == pytest.approx(2 * p * r / (p + r))


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k(["a", "e", "c"], REL, 3) == pytest.approx(1.0)
    dcg = 3 / math.log2(2) + 0 + 1 / math.log2(4)
    idcg = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, REL, 3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(RANKED, {}, 3) == 0.0
