"""Offline evaluation harness: score recommenders on the fixed profiles.

    python -m skillpath.eval.harness --models tfidf semantic

For each profile:
1. Each model returns its top POOL_DEPTH jobs. No role filter is set, so models are
   compared on ranking alone.
2. Every job in the pool (the union of all models' lists) gets a relevance grade from the
   heuristic judge, unless eval/qrels_manual.csv has a hand label for it.
3. The pooled labels are saved to eval/qrels.csv so they can be inspected and corrected.
4. P@K, R@K, F1@K and NDCG@K are computed per model.

Recall is measured against the relevant jobs in the pool, TREC-style. Searching all 47k jobs
for every relevant one isn't feasible, so recall means "how much of what any model found".
"""

import argparse

import pandas as pd

from skillpath.config import ROOT_DIR
from skillpath.eval import judge
from skillpath.eval.metrics import f1_at_k, ndcg_at_k, precision_at_k, recall_at_k
from skillpath.eval.profiles import load_profiles
from skillpath.recsys.base import load_jobs

QRELS = ROOT_DIR / "eval" / "qrels.csv"
QRELS_MANUAL = ROOT_DIR / "eval" / "qrels_manual.csv"
POOL_DEPTH = 50
KS = (5, 10)


def get_models(names: list[str]) -> dict:
    models = {}
    for name in names:
        if name == "tfidf":
            from skillpath.recsys.tfidf import TfidfRecommender
            models[name] = TfidfRecommender.load()
        elif name == "semantic":
            from skillpath.recsys.semantic import SemanticRecommender
            models[name] = SemanticRecommender.load()
        elif name.startswith("hybrid"):
            from skillpath.recsys.hybrid import HybridRecommender
            models[name] = HybridRecommender.load(variant=name.split("-", 1)[1] if "-" in name else "full")
        else:
            raise ValueError(f"unknown model {name!r}")
    return models


def _manual_labels() -> dict[tuple[str, str], int]:
    if not QRELS_MANUAL.exists():
        return {}
    m = pd.read_csv(QRELS_MANUAL)
    return {(r.profile_id, r.job_id): int(r.grade) for r in m.itertuples()}


def judge_pool(profile, job_ids: set[str], jobs_by_id: pd.DataFrame, manual: dict) -> dict[str, int]:
    out = {}
    for jid in job_ids:
        if (profile.profile_id, jid) in manual:
            out[jid] = manual[(profile.profile_id, jid)]
        else:
            row = jobs_by_id.loc[jid]
            out[jid] = judge.grade(profile, row["role_family"], list(row["skills_listed"]), list(row["skills_canonical"]))
    return out


def evaluate(models: dict, profiles=None, ks=KS, pool_depth=POOL_DEPTH, save_qrels: bool = True):
    """Returns (per-profile metrics, summary averaged over profiles)."""
    profiles = profiles or load_profiles()
    jobs_by_id = load_jobs().set_index("job_id")
    manual = _manual_labels()
    rows, qrels = [], []
    for p in profiles:
        ranked = {name: m.recommend(p, k=pool_depth)["job_id"].tolist() for name, m in models.items()}
        labels = judge_pool(p, set().union(*ranked.values()), jobs_by_id, manual)
        binary = {j: g for j, g in labels.items() if g >= judge.RELEVANT_GRADE}
        qrels += [{"profile_id": p.profile_id, "job_id": j, "grade": g,
                   "source": "manual" if (p.profile_id, j) in manual else "heuristic"} for j, g in labels.items()]
        for name, ids in ranked.items():
            row = {"profile_id": p.profile_id, "target_role": p.target_role, "model": name}
            for k in ks:
                row[f"P@{k}"] = precision_at_k(ids, binary, k)
                row[f"R@{k}"] = recall_at_k(ids, binary, k)
                row[f"F1@{k}"] = f1_at_k(ids, binary, k)
                row[f"NDCG@{k}"] = ndcg_at_k(ids, labels, k)
            rows.append(row)
    if save_qrels:
        QRELS.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(qrels).sort_values(["profile_id", "grade"], ascending=[True, False]).to_csv(QRELS, index=False)
    per_profile = pd.DataFrame(rows)
    summary = per_profile.drop(columns=["profile_id", "target_role"]).groupby("model").mean().round(3)
    return per_profile, summary.loc[list(models)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["tfidf", "semantic"])
    args = parser.parse_args()
    _, summary = evaluate(get_models(args.models))
    print(summary.to_string())


if __name__ == "__main__":
    main()
