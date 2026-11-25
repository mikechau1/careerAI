import pandas as pd
import pytest

from skillpath.data.roles import role_families, role_family

CASES = [
    ("Senior Machine Learning Engineer", "AI/ML Engineer"),
    ("AI Engineer - LLM Applications", "AI/ML Engineer"),
    ("Data Scientist II", "Data Scientist"),
    ("Senior Data Engineer (Spark)", "Data Engineer"),
    ("Business Intelligence Analyst", "Data Analyst"),
    ("Cybersecurity Analyst", "Security Engineer"),
    ("Site Reliability Engineer", "Cloud/DevOps Engineer"),
    ("AWS Cloud Engineer", "Cloud/DevOps Engineer"),
    ("Software Test Engineer", "QA/Test Engineer"),
    ("Full Stack Developer", "Software Engineer"),
    ("Sr. Software Engineer, Backend", "Software Engineer"),
    ("iOS Developer", "Software Engineer"),
    ("IT Support Technician", "IT/Systems"),
    ("Network Engineer", "IT/Systems"),
    # not tech
    ("Mechanical Engineer", None),
    ("Registered Nurse", None),
    ("Account Executive", None),
    ("Business Development Manager", None),
    ("Data Entry Clerk", None),
    ("Software Sales Representative", None),
    ("Real Estate Developer", None),
    ("Python Instructor", None),
    ("Counsel, Data Analytics", None),
    ("Sr. Human Resource Business Partner (Technology)", None),
    ("Cashier", None),
    ("", None),
    (None, None),
]


@pytest.mark.parametrize("title,expected", CASES)
def test_role_family(title, expected):
    assert role_family(title) == expected


def test_vectorized_matches_scalar():
    titles = pd.Series([t for t, _ in CASES])
    got = role_families(titles).tolist()
    assert [None if pd.isna(g) else g for g in got] == [e for _, e in CASES]
