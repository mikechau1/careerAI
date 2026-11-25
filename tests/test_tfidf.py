import pandas as pd
import pytest

from skillpath.profile import Profile
from skillpath.recsys.base import JOBS, RESULT_COLUMNS, job_document, top_k
from skillpath.recsys.tfidf import TfidfRecommender


def test_job_document_puts_title_and_skills_first(toy_jobs):
    doc = job_document(toy_jobs, max_desc_chars=10).iloc[0]
    assert doc.startswith("Data Engineer. Data Engineer. Skills: Python, Apache Spark")
    assert doc.endswith("Build Spar")


def test_tfidf_ranks_matching_job_first(toy_jobs):
    docs = pd.concat([toy_jobs] * 3, ignore_index=True)  # min_df=3 needs repeated terms
    rec = TfidfRecommender.fit(docs)
    scores = rec.scores(Profile(text="React TypeScript frontend developer", skills=["React"]))
    assert docs.loc[scores.argmax(), "title"] == "Frontend Developer"


def test_top_k_sorts_and_filters(toy_jobs):
    out = top_k(toy_jobs, [0.1, 0.9, 0.5, 0.3], k=2)
    assert out["job_id"].tolist() == ["j2", "j3"] and list(out.columns) == RESULT_COLUMNS
    assert top_k(toy_jobs, [0.1, 0.9, 0.5, 0.3], k=5, role_family="Software Engineer")["job_id"].tolist() == ["j3"]


@pytest.mark.skipif(not JOBS.exists(), reason="processed jobs not built")
def test_real_tfidf_recommends_data_engineering_jobs():
    rec = TfidfRecommender.load()
    p = Profile(skills=["Python", "Apache Spark", "Apache Airflow", "SQL"], target_role="Data Engineer",
                text="Data engineer building pipelines")
    out = rec.recommend(p, k=10)
    assert len(out) == 10 and (out["role_family"] == "Data Engineer").mean() >= 0.6
