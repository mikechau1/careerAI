import pandas as pd
import pytest

from skillpath.recsys.courses import COURSES, recommend_courses

TOY_COURSES = pd.DataFrame({
    "course_id": ["c1", "c2", "c3", "c4"],
    "title": ["Spark Basics", "Spark + Airflow Pipelines", "Advanced Spark", "Intro to Snowflake"],
    "provider": ["U1", "U2", "U3", "U4"], "difficulty": ["Beginner", "Beginner", "Advanced", "Beginner"],
    "rating": [4.7, 4.8, 4.2, None], "url": ["u1", "u2", "u3", "u4"],
    "skill_set": [{"Apache Spark"}, {"Apache Spark", "Apache Airflow"}, {"Apache Spark"}, {"Snowflake"}],
})


def test_courses_dedupe_and_prefer_multi_gap_courses():
    out = recommend_courses(["Apache Spark", "Apache Airflow"], per_skill=2, courses=TOY_COURSES, semantic_fallback=False)
    spark = out[out["gap_skill"] == "Apache Spark"]
    assert spark.iloc[0]["course_id"] == "c2"                # covers both gaps
    assert spark.iloc[0]["matched_skills"] == ["Apache Airflow", "Apache Spark"]
    assert out["course_id"].is_unique                         # c2 not repeated under Airflow


def test_courses_level_fit_for_missing_vs_improve():
    def first(status):
        gaps = pd.DataFrame({"skill": ["Apache Spark"], "status": [status]})
        return recommend_courses(gaps, per_skill=1, courses=TOY_COURSES.drop(1), semantic_fallback=False).iloc[0]
    assert first("Missing")["difficulty"] == "Beginner"
    assert first("Improve")["difficulty"] == "Advanced"


def test_broad_low_rated_course_does_not_win_on_coverage_alone():
    broad = pd.DataFrame({"course_id": ["broad", "good"], "title": ["Everything AI", "Intro to TensorFlow"],
                          "provider": ["X", "Y"], "difficulty": ["Advanced", "Beginner"], "rating": [3.4, 4.7],
                          "url": ["a", "b"],
                          "skill_set": [{"TensorFlow", "PyTorch", "Deep Learning", "NLP", "AI"}, {"TensorFlow"}]})
    out = recommend_courses(["TensorFlow", "PyTorch", "Deep Learning", "NLP", "AI"], per_skill=1, courses=broad,
                            semantic_fallback=False)
    assert out.iloc[0]["course_id"] == "good"


def test_course_named_after_the_skill_beats_a_passing_mention():
    courses = pd.DataFrame({"course_id": ["passing", "focused"],
                            "title": ["Getting Started with ProfitBooks", "AWS Elastic Beanstalk: Deploy a Flask App"],
                            "provider": ["X", "Y"], "difficulty": ["Beginner", "Beginner"], "rating": [5.0, 4.6],
                            "url": ["a", "b"], "description": ["Accounting software, hosted on AWS.", "Deploy to AWS."],
                            "skill_set": [{"AWS"}, {"AWS", "Flask"}]})
    out = recommend_courses(["AWS"], per_skill=1, courses=courses, semantic_fallback=False)
    assert out.iloc[0]["course_id"] == "focused"


def test_gap_with_no_courses_returns_nothing_without_fallback():
    assert recommend_courses(["Haskell"], courses=TOY_COURSES, semantic_fallback=False).empty


@pytest.mark.skipif(not COURSES.exists(), reason="run skillpath.skills.annotate first")
def test_real_courses_for_common_gaps():
    out = recommend_courses(["Docker", "Kubernetes", "Apache Spark"], semantic_fallback=False)
    assert set(out["gap_skill"]) == {"Docker", "Kubernetes", "Apache Spark"}
    assert out["course_id"].is_unique and out["url"].notna().all()
