"""M2-06: recommend courses that close a user's skill gaps.

    from skillpath.recsys.courses import recommend_courses
    gaps = prioritized_gaps(profile, "Data Engineer")       # M2-03
    recommend_courses(gaps)                                  # top 3 per gap, no course repeated

For each gap, in priority order, the candidates are the tech courses tagged with that skill
(M1-07). Candidates are scored on:
  relevance    +1.5 if the skill is in the course title, +0.25 per mention in the description
               (up to 4). Without this, a 5-star course that mentions AWS once beats an AWS course.
  coverage     1 for the target gap, plus 0.4 for each other gap the course covers (at most 3)
  rating       Coursera rating, centered on 4.5 (missing ratings count as 4.5), clipped to [-1, +0.5]
  level fit    0-1: Beginner or Mixed for Missing skills; Intermediate or Advanced for Improve
  focus        0-0.5: a course about a few skills beats a 30-skill specialization that mentions it once
Each course is assigned to at most one gap (the first one, in priority order, that picks it),
and lists every gap it covers in `matched_skills`. If fewer than `per_skill` courses are
tagged with a skill, semantic search over course descriptions fills the rest, but only with
courses at least MIN_SEMANTIC_SCORE similar. A skill the catalog doesn't cover (it's from
2021: no Snowflake or LangChain courses) gets no course rather than an unrelated one.
"""

from functools import lru_cache

import pandas as pd

from skillpath.config import PROCESSED_DIR
from skillpath.skills.extract import SkillExtractor

COURSES = PROCESSED_DIR / "courses.parquet"
MIN_SEMANTIC_SCORE = 0.45   # all-MiniLM cosine; below this, matches were unrelated in spot checks
TITLE_BONUS, MENTION_BONUS, MAX_MENTIONS = 1.5, 0.25, 4
LEVEL_FIT = {"Missing": {"Beginner": 1.0, "Mixed": 0.7, "Intermediate": 0.3, "Advanced": 0.0},
             "Improve": {"Beginner": 0.2, "Mixed": 0.6, "Intermediate": 1.0, "Advanced": 0.8}}
OUT_COLUMNS = ["gap_skill", "course_id", "title", "provider", "difficulty", "rating", "url",
               "matched_skills", "match", "score"]


def add_relevance_columns(courses: pd.DataFrame) -> pd.DataFrame:
    """Add title_skills (skills named in the title) and desc_counts (mentions per skill in the description)."""
    ext = SkillExtractor()
    courses = courses.copy()
    courses["title_skills"] = courses["title"].map(lambda t: set(ext.extract(t)))
    desc = courses["description"] if "description" in courses else pd.Series("", index=courses.index)
    courses["desc_counts"] = desc.fillna("").map(lambda d: dict(ext.extract_counts(d)))
    return courses


@lru_cache(maxsize=1)
def load_courses() -> pd.DataFrame:
    c = pd.read_parquet(COURSES)
    c = c[c["is_tech"]].copy()
    c["skill_set"] = c["skills"].map(set)
    return add_relevance_columns(c).reset_index(drop=True)


def _as_gap_frame(gaps) -> pd.DataFrame:
    if isinstance(gaps, pd.DataFrame):
        return gaps[["skill", "status"]].reset_index(drop=True)
    return pd.DataFrame({"skill": list(gaps), "status": "Missing"})


EXTRA_GAP_BONUS, MAX_EXTRA_GAPS = 0.4, 3


def _score(cands: pd.DataFrame, skill: str, status: str, gap_set: set[str]) -> pd.Series:
    relevance = (TITLE_BONUS * cands["title_skills"].map(lambda t: skill in t)
                 + MENTION_BONUS * cands["desc_counts"].map(lambda d: min(d.get(skill, 0), MAX_MENTIONS)))
    # 1 for the target gap + a capped bonus per other gap, so a broad course can't win on coverage alone
    coverage = 1 + EXTRA_GAP_BONUS * cands["skill_set"].map(lambda s: min(len(s & gap_set) - 1, MAX_EXTRA_GAPS))
    rating = (cands["rating"].fillna(4.5) - 4.5).clip(-1, 0.5)
    level = cands["difficulty"].map(LEVEL_FIT.get(status, LEVEL_FIT["Missing"])).fillna(0.5)
    focus = 1 / cands["skill_set"].map(len).clip(lower=1)
    return (relevance + coverage + rating + level + 0.5 * focus).astype(float)


def _semantic_fallback(skill: str, k: int, exclude: set[str]) -> pd.DataFrame:
    try:
        from skillpath.recsys.semantic import SemanticRecommender
        hits = SemanticRecommender.search_courses(f"{skill} course", k=k + len(exclude))
    except Exception:  # Chroma index not built: no fallback
        return pd.DataFrame()
    hits = hits[(hits["score"] >= MIN_SEMANTIC_SCORE) & ~hits["course_id"].isin(exclude)]
    return hits.head(k)


def recommend_courses(gaps, per_skill: int = 3, courses: pd.DataFrame | None = None,
                      semantic_fallback: bool = True) -> pd.DataFrame:
    """Top `per_skill` courses per gap skill, deduplicated across gaps.

    `gaps` is prioritized_gaps() output (skill + status) or a plain list of skills (treated as Missing)."""
    courses = load_courses() if courses is None else courses
    if "title_skills" not in courses:
        courses = add_relevance_columns(courses)
    gap_df = _as_gap_frame(gaps)
    gap_set = set(gap_df["skill"])
    used, rows = set(), []
    for skill, status in gap_df.itertuples(index=False):
        cands = courses[courses["skill_set"].map(lambda s: skill in s) & ~courses["course_id"].isin(used)]
        if not cands.empty:
            cands = cands.assign(score=_score(cands, skill, status, gap_set)).nlargest(per_skill, "score")
        for c in cands.itertuples():
            used.add(c.course_id)
            rows.append({"gap_skill": skill, "course_id": c.course_id, "title": c.title, "provider": c.provider,
                         "difficulty": c.difficulty, "rating": c.rating, "url": c.url,
                         "matched_skills": sorted(c.skill_set & gap_set), "match": "skill", "score": round(c.score, 3)})
        missing = per_skill - len(cands)
        if missing > 0 and semantic_fallback:
            meta = courses.set_index("course_id")
            for h in _semantic_fallback(skill, missing, used).itertuples():
                if h.course_id not in meta.index:
                    continue
                c = meta.loc[h.course_id]
                used.add(h.course_id)
                rows.append({"gap_skill": skill, "course_id": h.course_id, "title": c["title"], "provider": c["provider"],
                             "difficulty": c["difficulty"], "rating": c["rating"], "url": c["url"],
                             "matched_skills": sorted(c["skill_set"] & gap_set), "match": "semantic",
                             "score": round(float(h.score), 3)})
    return pd.DataFrame(rows, columns=OUT_COLUMNS)
