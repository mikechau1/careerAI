"""M2-05: association rule mining on job skills (FP-Growth).

    python -m skillpath.recsys.rules                       # mine rules -> processed/skill_rules.parquet
                                                           #   + docs/results/arm_top_rules.md
    from skillpath.recsys.rules import related_skills
    related_skills(["Docker", "AWS"])                      # global rules
    related_skills(["Machine Learning", "Python"], role_family="AI/ML Engineer")   # role-specific

Each tech job is one transaction: its set of canonical technical skills (M1-07). A rule
{Docker, AWS} -> {Kubernetes} with confidence 0.81 and lift 9.2 means: 81% of postings
that ask for Docker and AWS also ask for Kubernetes, 9.2x more often than Kubernetes
appears overall.

Rules are mined twice:
- globally over all 47k tech jobs (min support 2%), and
- within each role family (min support 5% of that family's jobs).
Global confidence is diluted by unrelated postings. P(PyTorch | ML, Python) is low across
all tech jobs, but high among AI/ML Engineer jobs. So when the user has a target role,
related_skills() uses that role's rules and falls back to the global ones.

related_skills() turns the rules into skill recommendations. It finds the rules whose
antecedents the user already has and whose consequent the user lacks. These become a
boost for skill-gap priorities (M2-03) and a "learnability" feature in Model 3 (M2-07).
"""

import inspect
import time
from functools import lru_cache

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from skillpath.config import PROCESSED_DIR, ROOT_DIR
from skillpath.recsys.base import load_jobs
from skillpath.skills.taxonomy import load_taxonomy

RULES = PROCESSED_DIR / "skill_rules.parquet"
REPORT = ROOT_DIR / "docs" / "results" / "arm_top_rules.md"
GLOBAL = "ALL"
MIN_SUPPORT = 0.02        # global: an itemset must appear in at least 2% of tech jobs (~950 postings)
MIN_SUPPORT_ROLE = 0.05   # per role family: 5% of that family's postings
MIN_CONFIDENCE = 0.4
MIN_LIFT = 1.5
MIN_LIFT_ROLE = 1.1      # within a family the core skills are already common, so lift stays near 1
MAX_LEN = 3


TEMPLATE_MIN_SKILLS = 6  # identical skill sets this large are the same posting, reposted


def transactions(role_family: str | None = None) -> list[list[str]]:
    """One transaction per job: its technical skills. Aggregators repost one employer's
    posting dozens of times under new titles (54 copies of one Capital One GenAI job came via
    "Jobs for Humanity"), which would turn template quirks into 100%-confidence rules. So
    identical skill sets with at least TEMPLATE_MIN_SKILLS skills are counted once."""
    tax = load_taxonomy()
    jobs = load_jobs()
    if role_family:
        jobs = jobs[jobs["role_family"] == role_family]
    tx, seen = [], set()
    for skills in jobs["skills_canonical"]:
        tech = [s for s in skills if tax.is_technical(s)]
        key = frozenset(tech)
        if len(key) >= TEMPLATE_MIN_SKILLS:
            if key in seen:
                continue
            seen.add(key)
        tx.append(tech)
    return tx


def mine(min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE, min_lift=MIN_LIFT,
         role_family: str | None = None) -> pd.DataFrame:
    tx = transactions(role_family)
    enc = TransactionEncoder()
    onehot = pd.DataFrame(enc.fit(tx).transform(tx), columns=enc.columns_)
    itemsets = fpgrowth(onehot, min_support=min_support, use_colnames=True, max_len=MAX_LEN)
    kwargs = {"metric": "confidence", "min_threshold": min_confidence}
    if "num_itemsets" in inspect.signature(association_rules).parameters:  # mlxtend >= 0.23.2
        kwargs["num_itemsets"] = len(onehot)
    rules = association_rules(itemsets, **kwargs)
    rules = rules[(rules["lift"] >= min_lift) & (rules["consequents"].map(len) == 1)].copy()
    rules["antecedents"] = rules["antecedents"].map(sorted)
    rules["consequent"] = rules["consequents"].map(lambda c: next(iter(c)))
    rules["role_family"] = role_family or GLOBAL
    cols = ["role_family", "antecedents", "consequent", "support", "confidence", "lift"]
    return rules[cols].sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def mine_all() -> pd.DataFrame:
    """Global rules plus rules within each role family."""
    families = sorted(load_jobs()["role_family"].dropna().unique())
    parts = [mine()] + [mine(min_support=MIN_SUPPORT_ROLE, min_lift=MIN_LIFT_ROLE, role_family=f) for f in families]
    return pd.concat(parts, ignore_index=True)


@lru_cache(maxsize=1)
def load_rules() -> pd.DataFrame:
    if not RULES.exists():
        raise FileNotFoundError("run `python -m skillpath.recsys.rules` first")
    rules = pd.read_parquet(RULES)
    rules["antecedents"] = rules["antecedents"].map(frozenset)
    return rules


def related_skills(skills, top_n: int = 10, role_family: str | None = None,
                   rules: pd.DataFrame | None = None) -> pd.DataFrame:
    """Skills the user lacks that the rules say usually go with skills they have.

    With role_family, that family's rules come first and the global rules fill in. Returns
    skill, confidence and lift (from the best rule for that skill), `because` (its
    antecedents) and `scope` (the role family or "ALL"), sorted by confidence then lift."""
    rules = load_rules() if rules is None else rules
    if "role_family" not in rules:
        rules = rules.assign(role_family=GLOBAL)
    have = set(skills)
    scopes = [role_family, GLOBAL] if role_family else [GLOBAL]
    picked = []
    for scope in scopes:
        r = rules[rules["role_family"] == scope]
        hits = r[r["antecedents"].map(lambda a: a <= have) & ~r["consequent"].isin(have)]
        picked.append(hits.sort_values(["confidence", "lift"], ascending=False))
    hits = pd.concat(picked).drop_duplicates("consequent").head(top_n)  # role-specific rules win ties
    if hits.empty:
        return pd.DataFrame(columns=["skill", "confidence", "lift", "because", "scope"])
    return pd.DataFrame({"skill": hits["consequent"], "confidence": hits["confidence"].round(3),
                         "lift": hits["lift"].round(2), "because": hits["antecedents"].map(sorted),
                         "scope": hits["role_family"]}).reset_index(drop=True)


def gap_boost(skills, role_family: str | None = None, top_n: int = 30) -> dict[str, float]:
    """Boost for skill_gap(): related skill -> its rule confidence (0-1)."""
    rel = related_skills(skills, top_n=top_n, role_family=role_family)
    return dict(zip(rel["skill"], rel["confidence"]))


def write_report(rules: pd.DataFrame, n_global: int = 20, n_role: int = 5):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    glob = rules[(rules["role_family"] == GLOBAL) & (rules["support"] >= 0.03)].head(n_global)
    header = ["| If a job asks for | it also asks for | Support | Confidence | Lift |", "|---|---|---:|---:|---:|"]

    def row(r):
        return f"| {', '.join(r.antecedents)} | {r.consequent} | {r.support:.3f} | {r.confidence:.2f} | {r.lift:.2f} |"

    lines = ["# Top skill association rules", "",
             f"FP-Growth over {len(load_jobs()):,} tech job postings. Min confidence {MIN_CONFIDENCE}, min lift "
             f"{MIN_LIFT}. Generated by `python -m skillpath.recsys.rules`.", "",
             f"## All tech jobs (min support {MIN_SUPPORT}; showing support >= 0.03, by lift)", "", *header,
             *[row(r) for r in glob.itertuples()], "",
             f"## Within each role family (min support {MIN_SUPPORT_ROLE} of the family's jobs, min lift {MIN_LIFT_ROLE}; top {n_role} by lift among rules with confidence >= 0.6)", ""]
    for fam in sorted(set(rules["role_family"]) - {GLOBAL}):
        fam_rules = rules[(rules["role_family"] == fam) & (rules["confidence"] >= 0.6)]
        top = fam_rules.sort_values(["lift", "confidence"], ascending=False).head(n_role)
        lines += [f"### {fam}", "", *header, *[row(r) for r in top.itertuples()], ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    t0 = time.time()
    rules = mine_all()
    rules.to_parquet(RULES, index=False)
    write_report(rules)
    load_rules.cache_clear()
    print(f"{len(rules):,} rules ({time.time() - t0:.0f}s) -> {RULES}")
    print(rules.groupby("role_family").agg(rules=("consequent", "size"), consequents=("consequent", "nunique")).to_string())
    for skills, fam in ((["Docker", "AWS"], None), (["Machine Learning", "Python"], "AI/ML Engineer"),
                        (["Python", "SQL"], "Data Engineer")):
        print(f"\nrelated to {skills} (role={fam}):")
        print(related_skills(skills, top_n=8, role_family=fam).to_string(index=False))


if __name__ == "__main__":
    main()
