import numpy as np
import pandas as pd
import pytest

from skillpath.profile import Profile
from skillpath.recsys import hybrid
from skillpath.recsys.hybrid import FEATURES, HybridRecommender, select_weights


class _Fixed:
    """Stand-in for Model 1 or 2: returns preset scores; has job vectors for the feedback feature."""
    def __init__(self, scores, vectors=None):
        self._s, self.job_vectors = np.asarray(scores, dtype=float), vectors

    def scores(self, profile):
        return self._s


@pytest.fixture
def rec(monkeypatch, toy_jobs):
    monkeypatch.setattr(hybrid, "load_jobs", lambda: toy_jobs)
    import skillpath.recsys.rules as rules
    monkeypatch.setattr(rules, "related_skills", lambda *a, **k: pd.DataFrame({"skill": ["Apache Airflow"]}))
    vectors = np.array([[1, 0], [1, 0], [0, 1], [0.8, 0.6]], dtype=np.float32)
    return lambda **kw: HybridRecommender(_Fixed([0.2, 0.1, 0.9, 0.3]), _Fixed([0.5, 0.4, 0.6, 0.7], vectors), **kw)


PROFILE = Profile(skills=["Python", "Apache Spark"], target_role="Data Engineer")


def test_features_are_scaled_and_correct(rec):
    f = rec().features(PROFILE)
    assert f["tfidf"].between(0, 1).all() and f["semantic"].between(0, 1).all()
    assert f.loc[0, "skill"] == pytest.approx(0.5)          # j1 tech skills: Python, Spark, Airflow, AWS
    assert f.loc[0, "related"] == pytest.approx(0.5)        # of the missing {Airflow, AWS}, Airflow is related
    assert f["role"].tolist() == [1.0, 1.0, 0.5, 1.0]       # j3 is Software Engineer: adjacent to Data Engineer


def test_role_and_skill_features_can_outrank_raw_text_score(rec):
    # j3 has the best text scores but the wrong family and no skill overlap
    assert rec(variant="text").recommend(PROFILE, k=1).iloc[0]["job_id"] == "j3"
    assert rec(weights={"role": 2.0, "skill": 1.0}).recommend(PROFILE, k=4).iloc[-1]["job_id"] == "j3"


def test_text_variant_ignores_skill_and_role(rec):
    r = rec(variant="text")
    assert r.weights["skill"] == r.weights["role"] == r.weights["related"] == 0
    assert r.recommend(PROFILE, k=1).iloc[0]["job_id"] in ("j3", "j4")   # pure text ranking
    assert rec(variant="norole").weights["role"] == 0
    with pytest.raises(ValueError):
        rec(variant="bogus")


def test_feedback_removes_disliked_and_boosts_similar_to_liked(rec):
    r = rec()
    out = r.recommend(PROFILE, k=4, feedback={"j1": -1})
    assert "j1" not in out["job_id"].tolist()
    liked = r.features(PROFILE, feedback={"j3": 1})["feedback"]
    assert liked[2] == pytest.approx(1.0) and liked[0] == 0.0   # j3 itself vs an orthogonal job


def test_explain_contributions_sum_to_score(rec):
    r = rec()
    e = r.explain(PROFILE, "j1")
    assert sum(e["contributions"].values()) == pytest.approx(e["score"], abs=1e-3)
    assert e["matched_skills"] == ["Apache Spark", "Python"] and "AWS" in e["missing_skills"]


def test_select_weights_prefers_more_signals_within_tolerance():
    table = pd.DataFrame([
        {"tfidf": 0.5, "semantic": 0.0, "skill": 0.5, "related": 0.0, "role": 1.0, "feedback": 0.0, "NDCG@10": 0.995},
        {"tfidf": 0.5, "semantic": 0.5, "skill": 0.5, "related": 0.25, "role": 1.0, "feedback": 0.0, "NDCG@10": 0.990},
        {"tfidf": 1.0, "semantic": 1.0, "skill": 1.0, "related": 0.25, "role": 1.0, "feedback": 0.0, "NDCG@10": 0.950},
    ])
    best = select_weights(table, "NDCG@10", tol=0.01)
    assert best["semantic"] == 0.5 and best["related"] == 0.25    # row 2: all signals, within 0.01 of the best
    assert select_weights(table, "NDCG@10", tol=0.001)["semantic"] == 0.0   # tighter tolerance: plain argmax
    assert set(best) == set(FEATURES)
