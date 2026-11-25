"""Build data/processed/resumes.parquet, the tech-only resume set used for test profiles (M2-04).

    python -m skillpath.data.preprocess_resumes

Resume categories are coarse: a software developer may be filed under ENGINEERING or
BANKING. A resume counts as tech if either:
- it is in INFORMATION-TECHNOLOGY and mentions at least MIN_TECH_TERMS_IT core tech skills
  (some IT resumes are instructors or supervisors with almost no technical content), or
- it is in any other category and mentions at least MIN_TECH_TERMS distinct core tech skills.

TECH_TERMS is only a filter heuristic. The real skill taxonomy is built in M1-05.
"""

import re

import pandas as pd

from skillpath.config import PROCESSED_DIR, RAW_DIR
from skillpath.data.roles import role_family

RESUMES_CSV = RAW_DIR / "resumes" / "Resume" / "Resume.csv"
OUT = PROCESSED_DIR / "resumes.parquet"
TECH_CATEGORY = "INFORMATION-TECHNOLOGY"
MIN_TECH_TERMS_IT = 2
MIN_TECH_TERMS = 6

TECH_TERMS = {
    "python": r"\bpython\b", "java": r"\bjava\b", "javascript": r"\bjavascript\b", "typescript": r"\btypescript\b",
    "c++": r"\bc\+\+", "c#": r"\bc#", ".net": r"\.net\b", "sql": r"\bsql\b", "mysql": r"\bmysql\b",
    "postgresql": r"\bpostgres(?:ql)?\b", "oracle": r"\boracle\b", "mongodb": r"\bmongo(?:db)?\b",
    "aws": r"\baws\b|amazon web services", "azure": r"\bazure\b", "gcp": r"\bgcp\b|google cloud",
    "docker": r"\bdocker\b", "kubernetes": r"\bkubernetes\b|\bk8s\b", "linux": r"\blinux\b", "unix": r"\bunix\b",
    "git": r"\bgit(?:hub|lab)?\b", "html": r"\bhtml5?\b", "css": r"\bcss3?\b", "react": r"\breact(?:\.?js)?\b",
    "angular": r"\bangular(?:js)?\b", "node.js": r"\bnode(?:\.?js)?\b", "php": r"\bphp\b", "ruby": r"\bruby\b",
    "perl": r"\bperl\b", "scala": r"\bscala\b", "matlab": r"\bmatlab\b", "spark": r"\bspark\b", "hadoop": r"\bhadoop\b",
    "tableau": r"\btableau\b", "power bi": r"\bpower ?bi\b", "machine learning": r"machine learning",
    "tensorflow": r"\btensorflow\b", "pytorch": r"\bpytorch\b", "pandas": r"\bpandas\b", "etl": r"\betl\b",
    "rest api": r"\brest(?:ful)?\b", "jenkins": r"\bjenkins\b", "ansible": r"\bansible\b", "terraform": r"\bterraform\b",
    "vmware": r"\bvmware\b", "active directory": r"active directory", "tcp/ip": r"\btcp/?ip\b", "cisco": r"\bcisco\b",
    "firewall": r"\bfirewalls?\b", "sharepoint": r"\bsharepoint\b", "salesforce": r"\bsalesforce\b",
    "selenium": r"\bselenium\b", "jira": r"\bjira\b", "devops": r"\bdevops\b", "cybersecurity": r"cyber ?security",
    "network administration": r"network admin", "data warehouse": r"data warehous", "visual basic": r"visual basic|\bvb\.net\b",
}
_TERM_RES = {name: re.compile(rx, re.I) for name, rx in TECH_TERMS.items()}


def tech_terms(text: str) -> list[str]:
    """Distinct core tech skills mentioned in a resume."""
    return [name for name, rx in _TERM_RES.items() if rx.search(text or "")]


def resume_title(text: str) -> str:
    """The resume's headline: the first chunk of text before a run of whitespace."""
    return re.split(r"\s{2,}|\t", (text or "").strip(), maxsplit=1)[0][:80].strip()


def build() -> pd.DataFrame:
    df = pd.read_csv(RESUMES_CSV, usecols=["ID", "Resume_str", "Category"])
    df = df.rename(columns={"ID": "resume_id", "Resume_str": "text", "Category": "category"})
    df["text"] = df["text"].fillna("").str.replace(r"[ \t]+", " ", regex=True).str.strip()
    df["tech_terms"] = df["text"].map(tech_terms)
    n_terms = df["tech_terms"].str.len()
    is_tech = ((df["category"] == TECH_CATEGORY) & (n_terms >= MIN_TECH_TERMS_IT)) | (n_terms >= MIN_TECH_TERMS)
    df = df[is_tech & (df["text"].str.len() >= 200)].copy()
    df["title"] = df["text"].map(resume_title)
    df["role_family"] = df["title"].map(role_family)  # often None: resume headlines are vague
    cols = ["resume_id", "category", "title", "role_family", "tech_terms", "text"]
    return df[cols].reset_index(drop=True)


def main():
    df = build()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  ({len(df):,} resumes)")
    print("by category:", df["category"].value_counts().to_dict())
    print("with a role family from the headline:", df["role_family"].notna().sum())
    print(f"median tech terms: {df['tech_terms'].str.len().median():.0f}")


if __name__ == "__main__":
    main()
