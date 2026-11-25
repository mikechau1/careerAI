"""M1-05: build data/skills.csv from the curated seed plus the job and O*NET data.

    python -m skillpath.skills.build

For each canonical skill it records:
- aliases: seed aliases, plus O*NET technology names that map to the skill
  (e.g. "Oracle Java" -> Java, "Atlassian JIRA" -> Jira)
- job_count / job_share: how many processed tech jobs list the skill (feeds skill-gap ranking)
- onet_hot: O*NET flags the technology as a "Hot Technology"
It then reports coverage and the most frequent unmapped job skills, which are candidates
to add to seed.py.
"""

import pandas as pd

from skillpath.config import PROCESSED_DIR, RAW_DIR
from skillpath.skills.seed import EXACT_ONLY
from skillpath.skills.taxonomy import SKILLS_CSV, Taxonomy, skill_key

ONET_TECH = RAW_DIR / "onet" / "db_30_2_text" / "Technology Skills.txt"
JOBS = PROCESSED_DIR / "jobs.parquet"
# Vendor prefixes in O*NET names: "Apache Kafka" -> "kafka", "Microsoft Excel" -> "excel"
_VENDORS = ("microsoft ", "apache ", "oracle ", "amazon ", "google ", "ibm ", "atlassian ", "adobe ",
            "apple ", "the mathworks ", "red hat ", "hashicorp ", "cisco ", "sap ", "salesforce ")
# Acronyms that pass the check below but name an unrelated product
_ONET_DENY = {"Pathogen Modeling Program PMP", "SoftRisk Technologies SoftRisk SQL"}


def _trailing_acronym(name: str) -> str | None:
    """'Border Gateway Protocol BGP' -> 'BGP'. The last word must have 2+ capitals and start
    with the same letter as the name. That rejects 'Schlumberger ECLIPSE' and 'Hadoop YARN'."""
    words = name.split()
    last = words[-1].strip("()")
    if len(words) < 3 or sum(c.isupper() for c in last) < 2:
        return None
    return last if last[0].lower() == words[0][0].lower() else None


def map_onet_name(tax: Taxonomy, name: str) -> str | None:
    """Map a verbose O*NET technology name to a canonical skill."""
    if name in _ONET_DENY:
        return None
    if skill := tax.normalize(name):
        return skill
    key = skill_key(name)
    for vendor in _VENDORS:
        if key.startswith(vendor) and (skill := tax.normalize(key[len(vendor):])):
            return skill
    acronym = _trailing_acronym(name)
    if acronym and skill_key(acronym) not in EXACT_ONLY and (skill := tax.normalize(acronym)):
        return skill
    return None


def onet_matches(tax: Taxonomy) -> pd.DataFrame:
    onet = pd.read_csv(ONET_TECH, sep="\t", usecols=["Example", "Hot Technology"]).drop_duplicates()
    onet["skill"] = onet["Example"].map(lambda n: map_onet_name(tax, n))
    return onet.dropna(subset=["skill"])


def job_skill_counts(tax: Taxonomy) -> tuple[pd.Series, pd.Series, int, float]:
    """(jobs per canonical skill, unmapped raw-skill frequencies, n_jobs, mention coverage)."""
    jobs = pd.read_parquet(JOBS, columns=["job_id", "skills"])
    mentions = jobs.explode("skills").dropna(subset=["skills"])
    keys = mentions["skills"].map(skill_key)
    lookup = {k: tax.normalize(k) for k in keys.unique()}
    mentions["skill"] = keys.map(lookup)
    coverage = mentions["skill"].notna().mean()
    per_skill = mentions.dropna(subset=["skill"]).drop_duplicates(["job_id", "skill"])["skill"].value_counts()
    unmapped = keys[mentions["skill"].isna()].value_counts()
    return per_skill, unmapped, len(jobs), coverage


def build() -> pd.DataFrame:
    tax = Taxonomy.from_seed()
    onet = onet_matches(tax)
    per_skill, unmapped, n_jobs, coverage = job_skill_counts(tax)

    rows = []
    for skill in tax.skills:
        onet_rows = onet[onet["skill"] == skill]
        aliases = dict.fromkeys(tax.aliases_of[skill][1:] + [skill_key(n) for n in onet_rows["Example"]])
        aliases.pop(skill_key(skill), None)
        count = int(per_skill.get(skill, 0))
        rows.append({
            "skill": skill,
            "category": tax.category(skill),
            "aliases": "|".join(aliases),
            "job_count": count,
            "job_share": round(count / n_jobs, 4),
            "onet_hot": bool((onet_rows["Hot Technology"] == "Y").any()),
        })
    df = pd.DataFrame(rows).sort_values(["category", "job_count"], ascending=[True, False])

    print(f"{len(df)} skills ({(df.category != 'Soft Skills').sum()} technical) across {df.category.nunique()} categories")
    print(f"skill mentions in {n_jobs:,} tech jobs mapped to the taxonomy: {coverage:.1%}")
    print(f"skills seen in at least 1 job: {(df.job_count > 0).sum()}, at least 100 jobs: {(df.job_count >= 100).sum()}")
    print(f"O*NET technology names mapped: {onet['Example'].nunique():,}; skills flagged hot: {df.onet_hot.sum()}")
    print("\nmost frequent unmapped job skills (candidates for seed.py):")
    print(", ".join(f"{k} ({v})" for k, v in unmapped.head(60).items()))
    return df


def main():
    df = build()
    df.to_csv(SKILLS_CSV, index=False)
    Taxonomy.from_csv(SKILLS_CSV)  # fails loudly if mined aliases collide
    print(f"\nwrote {SKILLS_CSV}")


if __name__ == "__main__":
    main()
