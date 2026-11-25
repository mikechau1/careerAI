import pandas as pd
import pytest

from skillpath.data.roles import role_relation
from skillpath.eval import harness, judge
from skillpath.eval.profiles import PROFILES_JSON, load_profiles
from skillpath.profile import Profile


def test_role_relation():
    assert role_relation("Data Engineer", "Data Engineer") == 2
    assert role_relation("Data Engineer", "Data Scientist") == 1
    assert role_relation("Data Engineer", "Security Engineer") == 0
    assert role_relation(None, "Data Engineer") == 0


def test_grade_combines_role_and_skill_fit():
    p = Profile(skills=["Python", "Apache Spark", "SQL"], target_role="Data Engineer")
    listed = ["Python", "Apache Spark", "Snowflake"]            # 2/3 fit
    assert judge.grade(p, "Data Engineer", listed, listed) == 3
    assert judge.grade(p, "Data Scientist", listed, listed) == 2  # adjacent + good fit: still relevant
    assert judge.grade(p, "Security Engineer", listed, listed) == 1
    assert judge.grade(p, "Data Engineer", ["Kubernetes", "Go", "Rust"], []) == 2


def test_grade_falls_back_to_all_skills_when_few_are_listed():
    p = Profile(skills=["Python", "SQL"], target_role="Data Analyst")
    assert judge.grade(p, "Data Analyst", ["Python"], ["Python", "SQL", "Tableau"]) == 3


class _FixedModel:
    def __init__(self, ids):
        self.ids = ids

    def recommend(self, profile, k=10):
        return pd.DataFrame({"job_id": self.ids[:k]})


def test_evaluate_pools_and_scores(monkeypatch, toy_jobs):
    monkeypatch.setattr(harness, "load_jobs", lambda: toy_jobs)
    p = Profile(skills=["Python", "Apache Spark", "SQL"], target_role="Data Engineer", profile_id="p1")
    models = {"good": _FixedModel(["j1", "j2", "j3"]), "bad": _FixedModel(["j3", "j1", "j2"])}
    per_profile, summary = harness.evaluate(models, [p], ks=(1, 3), pool_depth=3, save_qrels=False)
    assert summary.loc["good", "P@1"] == 1.0 and summary.loc["bad", "P@1"] == 0.0
    assert summary.loc["good", "NDCG@3"] > summary.loc["bad", "NDCG@3"]
    assert summary.loc["good", "R@3"] == summary.loc["bad", "R@3"] == 1.0   # same pool, same items


@pytest.mark.skipif(not PROFILES_JSON.exists(), reason="eval profiles not built")
def test_committed_profiles_cover_every_role_family():
    from skillpath.data.roles import ROLE_FAMILIES
    profiles = load_profiles()
    assert 10 <= len(profiles) <= 20
    assert {p.target_role for p in profiles} == {name for name, _ in ROLE_FAMILIES}
    assert all(len(p.skills) >= 5 for p in profiles)
