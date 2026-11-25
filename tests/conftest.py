import pandas as pd
import pytest

# A tiny job corpus shared by the recommender tests
TOY_JOBS = pd.DataFrame({
    "job_id": ["j1", "j2", "j3", "j4"],
    "title": ["Data Engineer", "Data Engineer", "Frontend Developer", "Data Engineer"],
    "role_family": ["Data Engineer", "Data Engineer", "Software Engineer", "Data Engineer"],
    "company": ["A", "B", "C", "D"], "location": ["x"] * 4, "level": ["Mid"] * 4,
    "description": ["Build Spark pipelines in Python with Airflow on AWS"] * 2
                   + ["React and TypeScript web apps"] + ["SQL and Python ETL into Snowflake"],
    "skills_listed": [["Python", "Apache Spark"], ["Python", "SQL"], ["React"], ["SQL", "Snowflake"]],
    "skills_canonical": [["Python", "Apache Spark", "Apache Airflow", "AWS", "Communication"],
                         ["Python", "SQL", "Apache Spark"], ["React", "TypeScript"], ["SQL", "Python", "Snowflake"]],
})


@pytest.fixture
def toy_jobs():
    return TOY_JOBS.copy()
