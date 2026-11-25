"""M2-03: skill gap analysis. What does a target role need that the user doesn't have yet?

    from skillpath.recsys.skill_gap import skill_gap, prioritized_gaps
    skill_gap(profile, "Data Engineer")        # one row per required skill, with a status
    prioritized_gaps(profile, "Data Engineer")  # just the gaps, most important first

A skill's importance for a role is the share of that role's postings that require it
(e.g. 68% of AI/ML Engineer jobs list Python). Only technical skills count by default,
because soft skills like communication are in almost every posting and aren't useful gaps.
Umbrella terms that restate the role ("Software Development", "Data Engineering") are
excluded too.

Status:
  Matched   the user has the skill
  Improve   the user has it, but marked their level as beginner (Profile.skill_levels)
  Missing   the user doesn't have it

Priority = importance x (1 + boost). `boost` is an optional per-skill bonus in [0, 1].
M2-05 passes skills that are strongly associated with what the user already knows,
since those are the most natural next skills to learn.
"""

from functools import lru_cache

import pandas as pd

from skillpath.profile import Profile
from skillpath.recsys.base import load_jobs
from skillpath.skills.taxonomy import load_taxonomy

GAP_STATUSES = ("Missing", "Improve")
# Umbrella terms that restate the role rather than name a learnable skill
# ("learn data engineering" is not actionable advice for a Data Engineer)
UMBRELLA = frozenset({"Software Development", "Data Engineering", "Data Science"})


@lru_cache(maxsize=32)
def role_requirements(role_family: str, technical_only: bool = True) -> pd.DataFrame:
    """Skills required by a role family, as the share of its postings that list each skill."""
    jobs = load_jobs()
    role_jobs = jobs[jobs["role_family"] == role_family]
    if role_jobs.empty:
        raise ValueError(f"no jobs for role family {role_family!r}")
    tax = load_taxonomy()
    counts = role_jobs["skills_canonical"].explode().dropna().value_counts()
    df = pd.DataFrame({"skill": counts.index, "share": (counts.to_numpy() / len(role_jobs)).round(4)})
    df["category"] = df["skill"].map(tax.category)
    if technical_only:
        df = df[df["skill"].map(tax.is_technical)]
    return df.reset_index(drop=True)


def _status(skill: str, profile: Profile, have: set[str]) -> str:
    if skill not in have:
        return "Missing"
    return "Improve" if profile.skill_levels.get(skill, "").lower() == "beginner" else "Matched"


def skill_gap(profile: Profile, role_family: str | None = None, top_n: int = 20, min_share: float = 0.05,
              boost: dict[str, float] | None = None, exclude: frozenset[str] = UMBRELLA) -> pd.DataFrame:
    """The top_n most required skills for the role, each with the user's status and a priority."""
    role = role_family or profile.target_role
    if not role:
        raise ValueError("no target role: pass role_family or set profile.target_role")
    req = role_requirements(role)
    req = req[(req["share"] >= min_share) & ~req["skill"].isin(exclude)].head(top_n).copy()
    have = set(profile.skills)
    boost = boost or {}
    req["status"] = req["skill"].map(lambda s: _status(s, profile, have))
    req["boost"] = req["skill"].map(lambda s: float(boost.get(s, 0.0)))
    req["priority"] = (req["share"] * (1 + req["boost"])).where(req["status"].isin(GAP_STATUSES), 0.0).round(4)
    return req[["skill", "category", "share", "status", "boost", "priority"]].reset_index(drop=True)


def prioritized_gaps(profile: Profile, role_family: str | None = None, n: int = 8, **kwargs) -> pd.DataFrame:
    """The user's n most important gaps (Missing or Improve), highest priority first."""
    gap = skill_gap(profile, role_family, **kwargs)
    return gap[gap["status"].isin(GAP_STATUSES)].sort_values("priority", ascending=False).head(n).reset_index(drop=True)


def match_rate(profile: Profile, role_family: str | None = None, **kwargs) -> float:
    """Importance-weighted share of the role's top skills the user already has (0-1), for the dashboard."""
    gap = skill_gap(profile, role_family, **kwargs)
    return float(gap.loc[gap["status"] == "Matched", "share"].sum() / gap["share"].sum()) if len(gap) else 0.0
