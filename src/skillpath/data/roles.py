"""Decide whether a job title is a tech role, and which role family it belongs to.

Families are checked in order and the first match wins. A "Software Test Engineer"
therefore lands in QA, not Software, and an "ML Data Engineer" lands in AI/ML.
"""

import re

import pandas as pd

# (family, regex) in priority order
ROLE_FAMILIES: list[tuple[str, str]] = [
    ("AI/ML Engineer", r"machine learning|\bml\b|\bmlops\b|\bai\b|artificial intelligence|\bllm\b|\bnlp\b"
                       r"|computer vision|deep learning|applied scientist|\bgen(?:erative)? ?ai\b"),
    ("Data Scientist", r"data scien"),
    ("Data Engineer", r"data engineer|\betl\b|analytics engineer|big data|data architect|data platform"
                      r"|data warehouse|\bdatabricks\b|\bsnowflake\b"),
    ("Data Analyst", r"data analy|business intelligence|\bbi (?:developer|analyst|engineer)|\btableau\b|\bpower ?bi\b"),
    ("Security Engineer", r"\bcyber|information security|\binfosec\b|security (?:engineer|analyst|architect|operations)"
                          r"|\bsoc analyst|penetration test|\bappsec\b"),
    ("Cloud/DevOps Engineer", r"\bcloud\b|\bdevops\b|\bdevsecops\b|\bsre\b|site reliability|platform engineer"
                              r"|infrastructure engineer|\bkubernetes\b|\baws\b|\bazure\b|\bgcp\b"),
    ("QA/Test Engineer", r"\bqa\b|\bsdet\b|test automation|automation test|quality assurance (?:engineer|analyst)"
                         r"|software test|test engineer"),
    ("Software Engineer", r"\bsoftware\b|\bdeveloper\b|\bprogrammer\b|front[- ]?end|back[- ]?end|full[- ]?stack"
                          r"|\bweb (?:engineer|application)|\bmobile engineer|\bios\b|\bandroid\b|\bjava\b|\bpython\b"
                          r"|\.net\b|\bjavascript\b|\btypescript\b|\breact\b|\bnode(?:\.?js)?\b|\bgolang\b|\bc\+\+"
                          r"|\bsalesforce (?:engineer|admin)|\bservicenow\b"),
    ("IT/Systems", r"\bit (?:support|specialist|manager|analyst|technician|director|administrator|engineer|operations)"
                   r"|information technology|help ?desk|service desk|desktop support|technical support"
                   r"|systems? (?:engineer|administrator|analyst)|\bsysadmin\b|network (?:engineer|administrator|architect)"
                   r"|database administrator|\bdba\b|solutions? architect|\blinux\b"),
]

# Titles that match a family pattern but are not tech jobs
EXCLUDE = (r"\bsales\b|account executive|business develop|real estate|property|mechanical|civil|electrical"
           r"|chemical|structural|manufacturing|hvac|\bnurse\b|data entry|\bdriver\b|recruit|talent acquisition"
           r"|teacher|instructor|professor|tutor|\btrainer\b|annotat|\bwriter\b|marketing"
           r"|counsel|attorney|paralegal|human resource|\bhr\b")

_FAMILY_RES = [(name, re.compile(rx, re.I)) for name, rx in ROLE_FAMILIES]
_EXCLUDE_RE = re.compile(EXCLUDE, re.I)


def role_family(title: str | None) -> str | None:
    """Return the role family for a job title, or None if it is not a tech job."""
    if not title or _EXCLUDE_RE.search(title):
        return None
    for name, rx in _FAMILY_RES:
        if rx.search(title):
            return name
    return None


def role_families(titles: pd.Series) -> pd.Series:
    """Vectorized role_family() for a Series of titles (fast on 1M+ rows)."""
    titles = titles.fillna("")
    out = pd.Series(pd.NA, index=titles.index, dtype="string")
    excluded = titles.str.contains(EXCLUDE, case=False, regex=True)
    for name, rx in reversed(ROLE_FAMILIES):  # reversed so the highest-priority family is written last
        out[titles.str.contains(rx, case=False, regex=True)] = name
    out[excluded] = pd.NA
    return out
