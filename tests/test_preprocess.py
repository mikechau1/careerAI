from skillpath.data.preprocess_jobs import _parse_skills
from skillpath.data.preprocess_resumes import resume_title, tech_terms


def test_parse_skills_dedupes_case_insensitively_and_keeps_order():
    assert _parse_skills("Python, SQL , python,, AWS") == ["Python", "SQL", "AWS"]
    assert _parse_skills(None) == []


def test_tech_terms_uses_word_boundaries():
    found = tech_terms("Built REST APIs in Python and Java; deployed with Docker on AWS. Javanese speaker.")
    assert {"python", "java", "docker", "aws", "rest api"} <= set(found)
    assert "javascript" not in found


def test_tech_terms_ignores_non_tech_text():
    assert tech_terms("Managed a bakery team and handled customer orders.") == []


def test_resume_title_takes_headline():
    assert resume_title("  SOFTWARE ENGINEER   Summary  Experienced developer") == "SOFTWARE ENGINEER"
