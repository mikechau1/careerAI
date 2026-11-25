"""M2-01, Model 1 (traditional baseline): TF-IDF + cosine similarity.

    from skillpath.recsys.tfidf import TfidfRecommender
    rec = TfidfRecommender.load()          # fits on first use, then caches to data/models/
    rec.recommend(profile, k=10)

    python -m skillpath.recsys.tfidf --fit   # (re)fit and cache explicitly

Each job is indexed as its title (twice), its canonical skills and its description (see
base.job_document). The query is Profile.query_text(): target role, skills, then the
resume text. Vectors are L2-normalized, so a dot product equals cosine similarity.
"""

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from skillpath.profile import Profile
from skillpath.recsys.base import MODELS_DIR, job_document, load_jobs, top_k

CACHE = MODELS_DIR / "tfidf.joblib"


class TfidfRecommender:
    name = "tfidf"

    def __init__(self, vectorizer: TfidfVectorizer, matrix, job_ids: np.ndarray):
        self.vectorizer, self.matrix, self.job_ids = vectorizer, matrix, job_ids

    @classmethod
    def fit(cls, jobs: pd.DataFrame | None = None) -> "TfidfRecommender":
        jobs = load_jobs() if jobs is None else jobs
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.5, max_features=300_000,
            sublinear_tf=True, token_pattern=r"(?u)\b[\w+#.]*\w[\w+#]*\b",  # keeps c++, c#, node.js
        )
        matrix = vectorizer.fit_transform(job_document(jobs))
        return cls(vectorizer, matrix, jobs["job_id"].to_numpy())

    @classmethod
    def load(cls) -> "TfidfRecommender":
        """The cached model, refit if the cache is missing or was built from a different job set."""
        if CACHE.exists():
            parts = joblib.load(CACHE)
            if list(parts["job_ids"]) == load_jobs()["job_id"].tolist():
                return cls(parts["vectorizer"], parts["matrix"], parts["job_ids"])
        rec = cls.fit()
        rec.save()
        return rec

    def save(self):
        # store the fitted parts, not the instance, so the cache doesn't depend on how the module was run
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "job_ids": self.job_ids}, CACHE)

    def scores(self, profile: Profile) -> np.ndarray:
        """Cosine similarity of the profile to every job, in load_jobs() order."""
        q = self.vectorizer.transform([profile.query_text()])
        return (self.matrix @ q.T).toarray().ravel()

    def recommend(self, profile: Profile, k: int = 10, role_family: str | None = None) -> pd.DataFrame:
        return top_k(load_jobs(), self.scores(profile), k, role_family)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true", help="refit and overwrite the cache")
    args = parser.parse_args()
    t0 = time.time()
    rec = TfidfRecommender.fit() if args.fit else TfidfRecommender.load()
    if args.fit:
        rec.save()
    print(f"TF-IDF over {rec.matrix.shape[0]:,} jobs x {rec.matrix.shape[1]:,} terms ({time.time() - t0:.0f}s)")
    demo = Profile(skills=["Python", "Apache Spark", "Apache Airflow", "SQL", "AWS"], target_role="Data Engineer",
                   text="Data engineer building batch and streaming pipelines.")
    print(rec.recommend(demo, k=10).to_string(index=False))


if __name__ == "__main__":
    main()
