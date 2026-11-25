"""Shared pieces for the job recommenders (Models 1-3).

Every recommender exposes the same call:

    recommend(profile: Profile, k: int = 10, role_family: str | None = None) -> pd.DataFrame

It returns the top-k jobs sorted by `score` (higher is better), with the columns in
RESULT_COLUMNS. `role_family` is an optional hard filter, for when the user wants only
jobs in one family. Evaluation never sets it, so models are compared on ranking alone.
"""

from functools import lru_cache

import pandas as pd

from skillpath.config import DATA_DIR, PROCESSED_DIR

JOBS = PROCESSED_DIR / "jobs.parquet"
MODELS_DIR = DATA_DIR / "models"  # cached fitted models (git-ignored)
RESULT_COLUMNS = ["job_id", "title", "role_family", "company", "location", "score"]


@lru_cache(maxsize=1)
def load_jobs() -> pd.DataFrame:
    """The processed tech jobs (M1-04) with canonical skills (M1-07)."""
    cols = ["job_id", "title", "role_family", "company", "location", "level", "description",
            "skills_listed", "skills_canonical"]
    return pd.read_parquet(JOBS, columns=cols)


def job_document(jobs: pd.DataFrame, max_desc_chars: int | None = None) -> pd.Series:
    """The text each model indexes for a job: title (twice, for weight), skills, description."""
    desc = jobs["description"] if max_desc_chars is None else jobs["description"].str[:max_desc_chars]
    skills = jobs["skills_canonical"].map(", ".join)
    return jobs["title"] + ". " + jobs["title"] + ". Skills: " + skills + ". " + desc


def top_k(jobs: pd.DataFrame, scores, k: int, role_family: str | None = None) -> pd.DataFrame:
    out = jobs.assign(score=scores)
    if role_family:
        out = out[out["role_family"] == role_family]
    return out.nlargest(k, "score")[RESULT_COLUMNS].reset_index(drop=True)
