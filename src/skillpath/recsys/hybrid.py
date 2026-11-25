"""M2-07, Model 3 (hybrid): a weighted mix of Models 1 and 2 plus skill, rule, role and feedback signals.

    from skillpath.recsys.hybrid import HybridRecommender
    rec = HybridRecommender.load()
    rec.recommend(profile, k=10, feedback={"<job_id>": +1, "<job_id>": -1})
    rec.explain(profile, job_id)            # per-feature score breakdown, for the explainer agent

    python -m skillpath.recsys.hybrid --tune   # grid-search the weights on the eval set

Every job gets these features, each scaled to [0, 1]:
  tfidf     Model 1 cosine, min-max scaled over all jobs
  semantic  Model 2 cosine, min-max scaled over all jobs
  skill     skill coverage: the share of the job's technical skills the user has
  related   learnability: the share of the job's skills the user lacks that the association
            rules link to the user's skills (M2-05). Gaps that are a natural next step
            count against a job less.
  role      1 if the job is in the user's target role family, 0.5 if adjacent, 0 otherwise
  feedback  similarity to jobs the user liked; jobs the user disliked are removed

score = sum(weight[f] * feature[f]). The weights are tuned on the eval profiles (M2-04).

Ablations (variant=...): the eval judge (M2-04) grades on role family and listed-skill fit,
and two features here mirror those signals. So the scores need reading in layers:
  "text"    Models 1 + 2 only. Shares no feature with the judge: the fair comparison.
  "norole"  everything except the role feature
  "full"    everything (the app default). Its eval score is inflated by construction.
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from skillpath.config import ROOT_DIR
from skillpath.data.roles import role_relation
from skillpath.profile import Profile
from skillpath.recsys.base import RESULT_COLUMNS, load_jobs
from skillpath.skills.taxonomy import load_taxonomy

FEATURES = ["tfidf", "semantic", "skill", "related", "role", "feedback"]
DEFAULT_WEIGHTS = {"tfidf": 0.5, "semantic": 0.5, "skill": 0.5, "related": 0.25, "role": 1.0, "feedback": 0.5}  # = tuned (eval/hybrid_weights.json)
WEIGHTS_FILE = ROOT_DIR / "eval" / "hybrid_weights.json"  # committed, so everyone runs the tuned model
TEXT_ONLY_WEIGHTS = {"tfidf": 0.5, "semantic": 1.0}         # best text-only setting in the tuning grid


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class HybridRecommender:
    name = "hybrid"

    def __init__(self, tfidf, semantic, weights: dict | None = None, variant: str = "full"):
        self.tfidf, self.semantic = tfidf, semantic
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        if variant == "norole":       # ablation: no target-role feature
            self.weights["role"] = 0.0
        elif variant == "text":       # ablation: Models 1 + 2 only (no feature the judge also uses)
            self.weights = {**{f: 0.0 for f in FEATURES}, **TEXT_ONLY_WEIGHTS}
        elif variant != "full":
            raise ValueError(f"unknown variant {variant!r}")
        self.name = "hybrid" if variant == "full" else f"hybrid-{variant}"
        jobs = load_jobs()
        tax = load_taxonomy()
        self._job_tech = [frozenset(s for s in skills if tax.is_technical(s)) for skills in jobs["skills_canonical"]]
        self._families = jobs["role_family"].to_numpy()
        self._ids = jobs["job_id"].to_numpy()
        self._pos = {j: i for i, j in enumerate(self._ids)}

    @classmethod
    def load(cls, variant: str = "full") -> "HybridRecommender":
        from skillpath.recsys.semantic import SemanticRecommender
        from skillpath.recsys.tfidf import TfidfRecommender
        weights = json.loads(WEIGHTS_FILE.read_text()) if WEIGHTS_FILE.exists() else None
        return cls(TfidfRecommender.load(), SemanticRecommender.load(), weights, variant)

    # --- features ------------------------------------------------------------------
    def features(self, profile: Profile, feedback: dict[str, int] | None = None) -> pd.DataFrame:
        """One row per job (load_jobs() order) with every feature in [0, 1]."""
        have = frozenset(profile.skills)
        try:
            from skillpath.recsys.rules import related_skills
            related = frozenset(related_skills(profile.skills, top_n=40, role_family=profile.target_role)["skill"])
        except FileNotFoundError:  # rules not mined yet
            related = frozenset()
        skill, rel = np.zeros(len(self._ids)), np.zeros(len(self._ids))
        for i, req in enumerate(self._job_tech):
            if req:
                skill[i] = len(req & have) / len(req)
                missing = req - have
                rel[i] = len(missing & related) / len(missing) if missing else 1.0
        role = np.array([role_relation(profile.target_role, f) / 2 for f in self._families])
        feats = pd.DataFrame({
            "tfidf": _minmax(self.tfidf.scores(profile)),
            "semantic": _minmax(self.semantic.scores(profile)),
            "skill": skill, "related": rel, "role": role,
            "feedback": self._feedback_feature(feedback),
        })
        return feats

    def _feedback_feature(self, feedback: dict[str, int] | None) -> np.ndarray:
        liked = [self._pos[j] for j, v in (feedback or {}).items() if v > 0 and j in self._pos]
        if not liked:
            return np.zeros(len(self._ids))
        sims = self.semantic.job_vectors @ self.semantic.job_vectors[liked].T   # cosine to each liked job
        return np.clip((sims.max(axis=1) - 0.5) / 0.5, 0, 1)                  # 0.5 -> 0, 1.0 -> 1

    def _score(self, feats: pd.DataFrame, weights: dict | None = None) -> np.ndarray:
        w = weights or self.weights
        return sum(w[f] * feats[f].to_numpy() for f in FEATURES)

    # --- public API ----------------------------------------------------------------
    def recommend(self, profile: Profile, k: int = 10, role_family: str | None = None,
                  feedback: dict[str, int] | None = None, breakdown: bool = False) -> pd.DataFrame:
        feats = self.features(profile, feedback)
        jobs = load_jobs()
        out = pd.concat([jobs, feats.add_prefix("f_")], axis=1).assign(score=self._score(feats))
        disliked = {j for j, v in (feedback or {}).items() if v < 0}
        out = out[~out["job_id"].isin(disliked)]
        if role_family:
            out = out[out["role_family"] == role_family]
        cols = RESULT_COLUMNS + ([f"f_{f}" for f in FEATURES] if breakdown else [])
        return out.nlargest(k, "score")[cols].reset_index(drop=True)

    def explain(self, profile: Profile, job_id: str, feedback: dict[str, int] | None = None) -> dict:
        """Weighted contribution of each feature to one job's score, plus the skills behind it."""
        i = self._pos[job_id]
        f = self.features(profile, feedback).iloc[i]
        req, have = self._job_tech[i], set(profile.skills)
        return {"job_id": job_id, "score": float(sum(self.weights[k] * f[k] for k in FEATURES)),
                "contributions": {k: round(float(self.weights[k] * f[k]), 4) for k in FEATURES},
                "matched_skills": sorted(req & have), "missing_skills": sorted(req - have)}


# --- tuning -------------------------------------------------------------------------
GRID = {"tfidf": [0.0, 0.5, 1.0], "semantic": [0.0, 0.5, 1.0], "skill": [0.0, 0.5, 1.0],
        "related": [0.0, 0.25], "role": [0.0, 0.5, 1.0]}
TUNE_TOLERANCE = 0.01


def tune(k: int = 10, use_role: bool = True) -> tuple[dict, pd.DataFrame]:
    """Grid search over GRID, maximizing mean NDCG@k on the eval profiles (heuristic judge).

    Features are computed once per profile, so each weight setting is just a matrix product."""
    from skillpath.eval import judge
    from skillpath.eval.profiles import load_profiles

    rec = HybridRecommender.load()
    jobs = load_jobs()
    listed, canon, fam = jobs["skills_listed"].to_numpy(), jobs["skills_canonical"].to_numpy(), jobs["role_family"].to_numpy()
    discounts = 1 / np.log2(np.arange(2, k + 2))
    cached = []
    for p in load_profiles():
        # grade EVERY job once, so the ideal ranking (the NDCG denominator) is fixed per profile and
        # doesn't depend on which jobs a weight setting happens to retrieve
        gains = 2.0 ** np.array([judge.grade(p, fam[i], list(listed[i]), list(canon[i])) for i in range(len(jobs))]) - 1
        ideal = float(np.sort(gains)[::-1][:k] @ discounts)
        cached.append((rec.features(p), gains, ideal))
    grid = dict(GRID, role=GRID["role"] if use_role else [0.0])
    results = []
    for combo in itertools.product(*grid.values()):
        w = dict(zip(grid, combo), feedback=0.0)
        if not any(combo):
            continue
        ndcgs = [float(gains[np.argsort(-rec._score(feats, w))[:k]] @ discounts) / ideal for feats, gains, ideal in cached]
        results.append({**w, f"NDCG@{k}": float(np.mean(ndcgs))})
    table = pd.DataFrame(results).sort_values(f"NDCG@{k}", ascending=False).reset_index(drop=True)
    return select_weights(table, f"NDCG@{k}"), table


def select_weights(table: pd.DataFrame, metric: str, tol: float = TUNE_TOLERANCE) -> dict:
    """Among settings within `tol` of the best score, prefer the one that uses the most signals.

    With 18 profiles, settings that differ by less than `tol` are a tie. The plain argmax tends
    to zero out a signal (e.g. drop the semantic model) on noise, and a model that uses every
    signal holds up better on new users."""
    near = table[table[metric] >= table[metric].max() - tol].copy()
    near["signals"] = (near[[f for f in FEATURES if f != "feedback"]] > 0).sum(axis=1)
    pick = near.sort_values(["signals", metric], ascending=False).iloc[0]
    best = {f: float(pick[f]) for f in FEATURES}
    best["feedback"] = DEFAULT_WEIGHTS["feedback"]  # no feedback in offline eval, so keep the default
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="grid-search weights and save them")
    args = parser.parse_args()
    if args.tune:
        best, table = tune()
        WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        WEIGHTS_FILE.write_text(json.dumps(best, indent=2))
        print(table.head(10).to_string(index=False))
        print(f"\nbest weights -> {WEIGHTS_FILE}: {best}")
    rec = HybridRecommender.load()
    demo = Profile(skills=["Python", "Apache Spark", "Apache Airflow", "SQL", "AWS"], target_role="Data Engineer",
                   text="Data engineer building batch and streaming pipelines.")
    print(rec.recommend(demo, k=10, breakdown=True).to_string(index=False))


if __name__ == "__main__":
    main()
