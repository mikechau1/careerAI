import numpy as np
import pytest

from skillpath.config import CHROMA_DIR
from skillpath.profile import Profile
from skillpath.recsys import semantic


def test_scores_are_cosine_against_every_job(monkeypatch, toy_jobs):
    # fake embedder: 2-d unit vectors; "react" queries point along y, others along x
    monkeypatch.setattr(semantic, "embed", lambda texts: np.array(
        [[0.0, 1.0] if "react" in t.lower() else [1.0, 0.0] for t in texts], dtype=np.float32))
    monkeypatch.setattr(semantic, "load_jobs", lambda: toy_jobs)
    vectors = semantic.embed(toy_jobs["description"].tolist())
    rec = semantic.SemanticRecommender(vectors)
    out = rec.recommend(Profile(text="React developer"), k=1)
    assert out.iloc[0]["job_id"] == "j3" and out.iloc[0]["score"] == pytest.approx(1.0)


def test_query_text_puts_role_and_skills_first():
    q = Profile(text="long resume " * 500, skills=["Python", "SQL"], target_role="Data Engineer").query_text(max_chars=50)
    assert q.startswith("Target role: Data Engineer. Skills: Python, SQL.")
    assert len(q) < 120


@pytest.mark.skipif(not (CHROMA_DIR / "chroma.sqlite3").exists(), reason="run `python -m skillpath.recsys.semantic --index`")
def test_real_index_covers_all_jobs_and_searches_courses():
    rec = semantic.SemanticRecommender.load()
    assert rec.job_vectors.shape[1] == 384
    assert np.allclose(np.linalg.norm(rec.job_vectors[:50], axis=1), 1, atol=1e-3)
    p = Profile(skills=["Kubernetes", "Terraform", "AWS"], target_role="Cloud/DevOps Engineer", text="DevOps engineer")
    assert (rec.recommend(p, k=10)["role_family"] == "Cloud/DevOps Engineer").mean() >= 0.5
    courses = rec.search_courses("introduction to kubernetes", k=3)
    assert len(courses) == 3 and courses["score"].between(0, 1).all()
