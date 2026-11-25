"""Heuristic relevance judge for (profile, job) pairs. These are proxy labels for M2-04.

grade = role relation + skill fit, from 0 to 3:
  role relation  2 if the job's role family is the profile's target role,
                 1 if it's an adjacent family (data.roles.ADJACENT), else 0
  skill fit      +1 if the profile has at least FIT_THRESHOLD of the job's
                 employer-listed technical skills
A job counts as relevant (binary metrics) when grade >= RELEVANT_GRADE: same family, or
an adjacent family with a good skill fit.

Why these signals: the role family comes from the job *title* (M1-04), and fit uses the
employer's *listed* skills. Models 1 and 2 rank on the full job text, so the labels don't
simply reward whatever a model looks at. Model 3 does use skill overlap, which is
correlated with skill fit; see M2-07 for the ablation.

These labels are a stand-in for human judgments. Hand-labeled rows in
eval/qrels_manual.csv override them (see eval/README.md).
"""

from skillpath.data.roles import role_relation
from skillpath.profile import Profile
from skillpath.skills.taxonomy import load_taxonomy

FIT_THRESHOLD = 0.3
RELEVANT_GRADE = 2


def skill_fit(profile_skills: set[str], job_skills: list[str]) -> float:
    """Share of the job's technical skills that the profile has (0-1)."""
    tax = load_taxonomy()
    tech = {s for s in job_skills if tax.is_technical(s)}
    return len(tech & profile_skills) / len(tech) if tech else 0.0


def grade(profile: Profile, job_family: str | None, job_listed_skills: list[str], job_all_skills: list[str]) -> int:
    """Relevance grade 0-3. Uses the listed skills, or all skills if the employer listed fewer than 3."""
    skills = job_listed_skills if len(job_listed_skills) >= 3 else job_all_skills
    fit = skill_fit(set(profile.skills), list(skills))
    return role_relation(profile.target_role, job_family) + int(fit >= FIT_THRESHOLD)
