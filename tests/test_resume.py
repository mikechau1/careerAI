from datetime import date
from pathlib import Path

import pytest

from skillpath.config import RAW_DIR
from skillpath.resume import headline, highest_degree, parse_resume, read_resume, years_of_experience
from skillpath.skills.extract import SkillExtractor, tokenize

SAMPLE = """JANE DOE
Senior Data Engineer
jane@example.com | (555) 123-4567

Summary: 6 years of experience building data pipelines on AWS with Python and Apache Spark.

Experience
Data Engineer, Acme Corp   01/2019 to Current
- Built ETL pipelines with Airflow and dbt into Snowflake; deployed services with Docker/Kubernetes.
- Wrote RESTful APIs in Python (FastAPI). Strong communication skills.
Software Developer, Beta Inc   2016 - 2019
- Java, SQL, CI/CD with Jenkins. Go to market planning with sales.

Education
B.S. Computer Science, State University
"""


@pytest.fixture(scope="module")
def extractor():
    return SkillExtractor()


def test_tokenize_keeps_tech_punctuation():
    assert tokenize("C#, .NET and Node.js; CI/CD.") == ["c#", ".net", "and", "node.js", "ci/cd"]


def test_extract_finds_multiword_and_alias_skills(extractor):
    skills = extractor.extract(SAMPLE)
    for expected in ["Python", "AWS", "Apache Spark", "ETL", "Apache Airflow", "dbt", "Snowflake",
                     "Docker", "Kubernetes", "FastAPI", "Java", "SQL", "CI/CD", "Jenkins"]:
        assert expected in skills, expected


def test_extract_skips_ambiguous_words_in_prose(extractor):
    skills = extractor.extract("Go to market with a strong sales team. R and D budget. Rest of the team.")
    assert "Go" not in skills and "R" not in skills and "REST APIs" not in skills


def test_extract_splits_slash_and_hyphen_joined_tokens(extractor):
    assert extractor.extract("python-based tooling with docker/kubernetes") == ["Python", "Docker", "Kubernetes"]


def test_technical_only_drops_soft_skills(extractor):
    assert extractor.extract("Python and problem solving", technical_only=True) == ["Python"]


def test_parse_resume_builds_profile():
    p = parse_resume(SAMPLE)
    assert p.highest_degree == "Bachelor's"
    assert p.years_experience == 6.0          # the explicit claim wins over date ranges
    assert p.title == "JANE DOE"              # headline is the first line; target role not guessable
    assert parse_resume(SAMPLE, target_role="Data Engineer").target_role == "Data Engineer"
    assert "Python" in p.skills


def test_years_from_date_ranges_merges_overlaps():
    text = "Analyst 2010 - 2014. Engineer 2013 to 2016. Lead 01/2018 to Current"
    assert years_of_experience(text, today=date(2024, 6, 1)) == 6 + 6


def test_degree_and_headline():
    assert highest_degree("MBA, 2015; B.S. 2010") == "Master's"
    assert highest_degree("High school diploma") is None
    assert headline("  SOFTWARE ENGINEER   Summary  ...") == "SOFTWARE ENGINEER"


def test_read_resume_text_and_bad_format(tmp_path):
    f = tmp_path / "cv.txt"
    f.write_text("Python developer", encoding="utf-8")
    assert read_resume(f) == "Python developer"
    assert read_resume(b"SQL analyst", filename="cv.md") == "SQL analyst"
    with pytest.raises(ValueError):
        read_resume(b"...", filename="cv.docx")


PDF_DIR = RAW_DIR / "resumes" / "data" / "data" / "INFORMATION-TECHNOLOGY"


@pytest.mark.skipif(not PDF_DIR.exists(), reason="resume dataset not downloaded")
def test_real_pdf_resumes_parse():
    pdfs = sorted(Path(PDF_DIR).glob("*.pdf"))[:5]
    profiles = [parse_resume(read_resume(p)) for p in pdfs]
    assert all(len(p.text) > 200 for p in profiles)
    assert sum(len(p.skills) >= 5 for p in profiles) >= 3
