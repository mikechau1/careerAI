import pandas as pd
import pytest

from skillpath.config import PROCESSED_DIR
from skillpath.skills.annotate import _union
from skillpath.skills.extract import SkillExtractor


def test_union_keeps_order_and_dedupes():
    assert _union(["Python", "SQL"], ["SQL", "Docker"]) == ["Python", "SQL", "Docker"]


def test_space_separated_course_skills_are_split_by_the_extractor():
    # Coursera's Skills field runs skills together with spaces (M1-03 caveat)
    skills = SkillExtractor().extract("Machine Learning  Python Programming  Data Visualization  peering")
    assert skills == ["Machine Learning", "Python", "Data Visualization"]


@pytest.mark.skipif(not (PROCESSED_DIR / "courses.parquet").exists(), reason="run skillpath.skills.annotate first")
def test_annotated_outputs_have_skill_columns():
    jobs = pd.read_parquet(PROCESSED_DIR / "jobs.parquet", columns=["skills_listed", "skills_text", "skills_canonical"])
    assert (jobs["skills_canonical"].str.len() >= 3).mean() > 0.9
    courses = pd.read_parquet(PROCESSED_DIR / "courses.parquet")
    assert {"course_id", "title", "provider", "url", "skills", "skills_field", "is_tech"} <= set(courses.columns)
    assert courses["course_id"].is_unique
    onboarding = courses[courses["title"] == "Create Online Employee Onboarding Course with Eduflow"]
    assert onboarding.empty or "Deep Learning" not in list(onboarding.iloc[0]["skills"])  # noisy Skills field ignored
