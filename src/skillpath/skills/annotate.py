"""M1-07: tag every job and course with canonical skills.

    python -m skillpath.skills.annotate

Jobs (adds columns to processed/jobs.parquet):
  skills_listed     the employer-listed skills (`skills`) normalized to canonical names
  skills_text       canonical skills found in the title and description
  skills_canonical  union of both, listed first. This is what the recommenders use.

Courses (writes processed/courses.parquet from the raw Coursera CSV):
  skills            canonical skills found in the course's Skills field, title and description
  is_tech           has at least one technical skill
"""

import hashlib

import pandas as pd

from skillpath.config import PROCESSED_DIR, RAW_DIR
from skillpath.skills.extract import SkillExtractor
from skillpath.skills.taxonomy import load_taxonomy

JOBS = PROCESSED_DIR / "jobs.parquet"
COURSES_RAW = RAW_DIR / "courses" / "Coursera.csv"
COURSES = PROCESSED_DIR / "courses.parquet"


def _union(a, b) -> list[str]:
    return list(dict.fromkeys([*a, *b]))


def annotate_jobs(extractor: SkillExtractor) -> pd.DataFrame:
    tax = extractor.taxonomy
    jobs = pd.read_parquet(JOBS)
    jobs["skills_listed"] = jobs["skills"].map(lambda s: tax.normalize_list(list(s)))
    jobs["skills_text"] = (jobs["title"] + ". " + jobs["description"]).map(extractor.extract)
    jobs["skills_canonical"] = [_union(a, b) for a, b in zip(jobs["skills_listed"], jobs["skills_text"])]
    return jobs


def build_courses(extractor: SkillExtractor) -> pd.DataFrame:
    raw = pd.read_csv(COURSES_RAW).rename(columns={
        "Course Name": "title", "University": "provider", "Difficulty Level": "difficulty",
        "Course Rating": "rating", "Course URL": "url", "Course Description": "description", "Skills": "skills_raw",
    })
    raw = raw.drop_duplicates("url").drop_duplicates(["title", "provider"])
    raw["rating"] = pd.to_numeric(raw["rating"], errors="coerce")  # "Not Calibrated" -> NaN
    raw["difficulty"] = raw["difficulty"].where(raw["difficulty"].isin(["Beginner", "Intermediate", "Advanced", "Mixed"]))
    raw["description"] = raw["description"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    text = raw["skills_raw"].fillna("") + " . " + raw["title"] + " . " + raw["description"]
    raw["skills"] = text.map(extractor.extract)
    raw["is_tech"] = raw["skills"].map(lambda ss: any(extractor.taxonomy.is_technical(s) for s in ss))
    raw["course_id"] = raw["url"].map(lambda u: hashlib.sha1(str(u).encode()).hexdigest()[:12])
    cols = ["course_id", "title", "provider", "difficulty", "rating", "url", "skills", "is_tech", "description", "skills_raw"]
    return raw[cols].reset_index(drop=True)


def _coverage(name: str, skills: pd.Series):
    n = skills.str.len()
    print(f"{name}: {len(skills):,} rows | >=3 skills: {(n >= 3).mean():.1%} | >=1: {(n >= 1).mean():.1%} | median {n.median():.0f}")


def main():
    extractor = SkillExtractor(load_taxonomy())
    courses = build_courses(extractor)
    courses.to_parquet(COURSES, index=False)
    _coverage("courses (all)", courses["skills"])
    _coverage("courses (tech)", courses.loc[courses["is_tech"], "skills"])

    jobs = annotate_jobs(extractor)
    jobs.to_parquet(JOBS, index=False)
    for col in ("skills_listed", "skills_text", "skills_canonical"):
        _coverage(f"jobs.{col}", jobs[col])
    print(f"wrote {COURSES} and updated {JOBS}")


if __name__ == "__main__":
    main()
