"""M1-04: build data/processed/jobs.parquet, the tech-only, cleaned job corpus.

    python -m skillpath.data.preprocess_jobs

Steps:
1. Classify every posting title into a role family and keep only tech roles (see roles.py).
2. Join the posting's extracted skills and its description text.
3. Drop postings without a description, normalize whitespace and parse skills into a list.
4. Remove reposts: the same company, title and description listed in several cities.
"""

import hashlib
import time

import pandas as pd

from skillpath.config import PROCESSED_DIR, RAW_DIR
from skillpath.data.roles import role_families

JOBS_RAW = RAW_DIR / "jobs"
OUT = PROCESSED_DIR / "jobs.parquet"
CHUNK = 200_000

POSTING_COLS = {
    "job_link": "job_link", "job_title": "title", "company": "company", "job_location": "location",
    "search_country": "country", "job_level": "level", "job_type": "job_type", "first_seen": "first_seen",
}


def _read_filtered(path, links: set[str]) -> pd.DataFrame:
    """Read a large CSV in chunks, keeping only rows whose job_link is in `links`."""
    parts = [ch[ch["job_link"].isin(links)] for ch in pd.read_csv(path, chunksize=CHUNK)]
    return pd.concat(parts, ignore_index=True)


def _clean_text(s: pd.Series) -> pd.Series:
    return s.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()


def _parse_skills(raw) -> list[str]:
    """'Python, SQL , python, AWS' -> ['Python', 'SQL', 'AWS'] (case-insensitive dedupe, order kept)."""
    if not isinstance(raw, str):
        return []
    seen, out = set(), []
    for s in (x.strip() for x in raw.split(",")):
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def build() -> pd.DataFrame:
    t0 = time.time()
    posts = pd.read_csv(JOBS_RAW / "linkedin_job_postings.csv", usecols=list(POSTING_COLS)).rename(columns=POSTING_COLS)
    n_raw = len(posts)
    posts["role_family"] = role_families(posts["title"])
    posts = posts[posts["role_family"].notna()].drop_duplicates("job_link")
    print(f"tech postings: {len(posts):,} of {n_raw:,}  ({time.time() - t0:.0f}s)")

    links = set(posts["job_link"])
    skills = _read_filtered(JOBS_RAW / "job_skills.csv", links).drop_duplicates("job_link")
    summary = _read_filtered(JOBS_RAW / "job_summary.csv", links).drop_duplicates("job_link")
    print(f"joined skills: {len(skills):,}, descriptions: {len(summary):,}  ({time.time() - t0:.0f}s)")

    df = posts.merge(summary, on="job_link", how="inner").merge(skills, on="job_link", how="left")
    df["description"] = _clean_text(df.pop("job_summary"))
    df["title"] = _clean_text(df["title"])
    df["skills"] = df.pop("job_skills").map(_parse_skills)
    df = df[df["description"].str.len() >= 200]  # drop empty / stub descriptions

    before = len(df)
    dedupe_key = pd.DataFrame({
        "company": df["company"].fillna("").str.lower(),
        "title": df["title"].str.lower(),
        "desc": df["description"].map(lambda d: hashlib.md5(d.encode()).hexdigest()),
    })
    df = df[~dedupe_key.duplicated()]
    print(f"removed {before - len(df):,} reposts")

    df["job_id"] = df["job_link"].map(lambda u: hashlib.sha1(u.encode()).hexdigest()[:12])
    df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce")
    cols = ["job_id", "title", "role_family", "company", "location", "country", "level", "job_type",
            "first_seen", "skills", "description", "job_link"]
    return df[cols].reset_index(drop=True)


def main():
    df = build()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df):,} jobs, {OUT.stat().st_size / 1e6:.0f} MB)")
    print(df["role_family"].value_counts().to_string())
    print(f"with skills: {(df['skills'].str.len() > 0).mean():.1%}, "
          f"median skills/job: {df['skills'].str.len().median():.0f}, "
          f"median description chars: {df['description'].str.len().median():.0f}")


if __name__ == "__main__":
    main()
