"""M2-02, Model 2 (semantic): sentence embeddings + ChromaDB vector retrieval.

    python -m skillpath.recsys.semantic --index   # embed jobs + courses into data/chroma/ (re-runnable)

    from skillpath.recsys.semantic import SemanticRecommender
    rec = SemanticRecommender.load()
    rec.recommend(profile, k=10)                  # same interface as Model 1
    rec.search_courses("intro to docker", k=5)    # course search, used by M2-06 and the agents

ChromaDB is the persistent vector store; the app and agents query it with metadata filters.
For ranking, the recommender also pulls every job vector into memory once and scores all
47k jobs exactly (a 47k x 384 dot product takes milliseconds). Model 3 needs a score for
every candidate, not just Chroma's top-n.

Embedding model: EMBEDDING_MODEL in .env (default all-MiniLM-L6-v2, 384 dims). It reads up
to 256 word pieces, so documents put the title and skills first and the description after.
"""

import argparse
import time

import chromadb
import numpy as np
import pandas as pd

from skillpath.config import CHROMA_DIR, EMBEDDING_MODEL, PROCESSED_DIR
from skillpath.profile import Profile
from skillpath.recsys.base import job_document, load_jobs, top_k

JOBS_COLLECTION, COURSES_COLLECTION = "jobs", "courses"
BATCH = 2048
_model = None


def embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # slow import; only when needed
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(texts: list[str], show_progress: bool = False) -> np.ndarray:
    return embedding_model().encode(texts, batch_size=64, normalize_embeddings=True,
                                    show_progress_bar=show_progress, convert_to_numpy=True).astype(np.float32)


def client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _rebuild(name: str, ids: list[str], docs: list[str], metas: list[dict]):
    db = client()
    if name in [c.name if hasattr(c, "name") else c for c in db.list_collections()]:
        db.delete_collection(name)
    col = db.create_collection(name, metadata={"hnsw:space": "cosine"})
    vectors = embed(docs, show_progress=True)
    for i in range(0, len(ids), BATCH):
        col.add(ids=ids[i:i + BATCH], embeddings=vectors[i:i + BATCH].tolist(),
                documents=docs[i:i + BATCH], metadatas=metas[i:i + BATCH])
    print(f"indexed {col.count():,} into '{name}'")


def index_jobs():
    jobs = load_jobs()
    metas = jobs[["title", "role_family", "company"]].fillna("").to_dict("records")
    _rebuild(JOBS_COLLECTION, jobs["job_id"].tolist(), job_document(jobs, max_desc_chars=1000).tolist(), metas)


def course_document(courses: pd.DataFrame) -> pd.Series:
    return courses["title"] + ". Skills: " + courses["skills"].map(", ".join) + ". " + courses["description"].str[:1000]


def index_courses():
    courses = pd.read_parquet(PROCESSED_DIR / "courses.parquet")
    metas = courses[["title", "provider", "difficulty", "url"]].fillna("").to_dict("records")
    for m, tech in zip(metas, courses["is_tech"]):
        m["is_tech"] = bool(tech)
    _rebuild(COURSES_COLLECTION, courses["course_id"].tolist(), course_document(courses).tolist(), metas)


class SemanticRecommender:
    name = "semantic"

    def __init__(self, job_vectors: np.ndarray):
        self.job_vectors = job_vectors  # aligned with load_jobs() row order

    @classmethod
    def load(cls) -> "SemanticRecommender":
        col = client().get_collection(JOBS_COLLECTION)
        got = col.get(include=["embeddings"], limit=col.count())
        by_id = dict(zip(got["ids"], np.asarray(got["embeddings"], dtype=np.float32)))
        ids = load_jobs()["job_id"]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise RuntimeError(f"{len(missing)} jobs are not indexed; run `python -m skillpath.recsys.semantic --index`")
        return cls(np.stack([by_id[i] for i in ids]))

    def scores(self, profile: Profile) -> np.ndarray:
        """Cosine similarity of the profile to every job, in load_jobs() order."""
        return self.job_vectors @ embed([profile.query_text(max_chars=800)])[0]

    def recommend(self, profile: Profile, k: int = 10, role_family: str | None = None) -> pd.DataFrame:
        return top_k(load_jobs(), self.scores(profile), k, role_family)

    @staticmethod
    def search_courses(text: str, k: int = 5, tech_only: bool = True) -> pd.DataFrame:
        """Nearest courses to a free-text query, straight from Chroma."""
        res = client().get_collection(COURSES_COLLECTION).query(
            query_embeddings=embed([text]).tolist(), n_results=k, where={"is_tech": True} if tech_only else None)
        rows = [{"course_id": i, **m, "score": 1 - d}
                for i, m, d in zip(res["ids"][0], res["metadatas"][0], res["distances"][0])]
        return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="(re)build the jobs and courses collections")
    args = parser.parse_args()
    t0 = time.time()
    if args.index:
        index_courses()
        index_jobs()
        print(f"indexing took {time.time() - t0:.0f}s")
    rec = SemanticRecommender.load()
    demo = Profile(skills=["Python", "Apache Spark", "Apache Airflow", "SQL", "AWS"], target_role="Data Engineer",
                   text="Data engineer building batch and streaming pipelines.")
    print(rec.recommend(demo, k=10).to_string(index=False))
    print(rec.search_courses("containers and kubernetes for beginners", k=5)[["title", "provider", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
