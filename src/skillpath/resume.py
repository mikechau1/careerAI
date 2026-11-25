"""M1-06: turn an uploaded resume (PDF or text) into a structured Profile.

    from skillpath.resume import read_resume, parse_resume
    profile = parse_resume(read_resume("resume.pdf"), target_role="Data Engineer")

Everything is rule-based (skill dictionary, regexes), so it works offline with no API key.
An LLM fallback could fill gaps later, but the dictionary match is what the recommenders
rely on, because it maps straight onto the canonical skill list.
"""

import io
import re
from datetime import date
from pathlib import Path

from pypdf import PdfReader

from skillpath.data.roles import role_family
from skillpath.profile import Profile
from skillpath.skills.extract import SkillExtractor

_DEGREES = [  # highest first
    ("PhD", r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctor of philosophy\b"),
    ("Master's", r"\bmaster'?s?\b|\bm\.s\.|\bm\.sc\b|\bmsc\b|\bmba\b|\bm\.eng\b|\bmeng\b"),
    ("Bachelor's", r"\bbachelor'?s?\b|\bb\.s\.|\bb\.sc\b|\bbsc\b|\bb\.a\.|\bb\.eng\b|\bbeng\b|\bb\.tech\b|\bbtech\b"),
    ("Associate", r"\bassociate'?s? (?:degree|of)\b|\ba\.a\.s\b"),
]
_EXPLICIT_YEARS = re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)\.?\s+(?:of\s+)?(?:\w+\s+){0,3}?experience", re.I)
_MONTH = r"(?:(?:0?[1-9]|1[0-2])/|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+)?"
_RANGE = re.compile(
    rf"{_MONTH}((?:19|20)\d{{2}})\s*(?:-|–|—|to|until)\s*(?:{_MONTH}((?:19|20)\d{{2}})|(current|present|now|today))", re.I)

_extractor: SkillExtractor | None = None


def _get_extractor() -> SkillExtractor:
    global _extractor
    if _extractor is None:
        _extractor = SkillExtractor()
    return _extractor


def read_resume(source: str | Path | bytes, filename: str | None = None) -> str:
    """Extract text from a .pdf, .txt or .md resume, given a path or raw bytes plus a filename."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        data, filename = path.read_bytes(), filename or path.name
    else:
        data = source
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf" or data[:5] == b"%PDF-":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if suffix in ("", ".txt", ".md"):
        return data.decode("utf-8", errors="replace").strip()
    raise ValueError(f"unsupported resume format {suffix!r}; use PDF or plain text")


def highest_degree(text: str) -> str | None:
    for name, rx in _DEGREES:
        if re.search(rx, text, re.I):
            return name
    return None


def years_of_experience(text: str, today: date | None = None) -> float | None:
    """An explicit 'N years of experience' claim if there is one; otherwise the total span of
    the dated job ranges ('01/2015 to Current'), with overlapping ranges merged."""
    explicit = [int(m.group(1)) for m in _EXPLICIT_YEARS.finditer(text)]
    if explicit:
        return float(max(explicit))
    now = (today or date.today()).year
    spans = []
    for m in _RANGE.finditer(text):
        start = int(m.group(1))
        end = now if m.group(3) else int(m.group(2))
        if 1960 <= start <= end <= now:
            spans.append((start, end))
    if not spans:
        return None
    total, cur_start, cur_end = 0, *sorted(spans)[0]
    for s, e in sorted(spans)[1:]:
        if s > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    total += cur_end - cur_start
    return float(min(total, 50))


def headline(text: str) -> str:
    """The first meaningful line of the resume, e.g. 'SOFTWARE ENGINEER'."""
    for line in re.split(r"\n|\s{3,}|\t", text.strip()):
        line = line.strip(" |•-")
        if len(line) >= 3:
            return line[:100]
    return ""


def parse_resume(text: str, target_role: str | None = None, profile_id: str | None = None) -> Profile:
    """Build a Profile from resume text. If target_role isn't given, guess it from the headline."""
    title = headline(text)
    return Profile(
        text=text,
        skills=_get_extractor().extract(text),
        target_role=target_role or role_family(title),
        title=title or None,
        highest_degree=highest_degree(text),
        years_experience=years_of_experience(text),
        profile_id=profile_id,
    )
